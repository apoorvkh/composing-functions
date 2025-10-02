import random
from dataclasses import dataclass
from statistics import mean
from typing import Literal, Sequence, TypedDict

import torch
from tango.integrations.torch import TorchFormat
from tqdm import tqdm

from ..models.transformers import NNsightModel, TransformersModelT
from ..prompting import InContextQuery
from ..tasks import CompositionalTask, Task, TaskT
from .lens import IntermediateVariable, LensExperiment
from .residual_stream import select_words_from_query
from .utils import Experiment, Step, Sweep, step


class Patch(TypedDict):
    x: str
    activation: torch.Tensor


@step(cacheable=True, version="004", format=TorchFormat())
def select_patches_from_source(
    queries_src: list[InContextQuery],
    queries_dst: list[InContextQuery],
    intermediate_vars_src: list[IntermediateVariable],
    vars_prob_range: tuple[float, float],
    layer_pos_src: tuple[int, float] | Literal["max"],
    query_tokens_residual_stream_src: list[torch.Tensor],
) -> list[Patch]:
    """Select patches from source task for activation patching experiments."""
    # which x are solved completely correctly (all hops, composition) in both tasks
    x_src_list = [icq.query.x for icq in queries_src]
    x_dst_list = [icq.query.x for icq in queries_dst]
    common_x_values = set(x_src_list) & set(x_dst_list)

    results = []
    for icq_src, var_src, qtrs_src in zip(queries_src, intermediate_vars_src, query_tokens_residual_stream_src):
        x_src = icq_src.query.x
        if x_src in common_x_values and (vars_prob_range[0] <= var_src["prob"] <= vars_prob_range[1]):
            if layer_pos_src == "max":
                pos, layer = var_src["position"], var_src["layer"]
            else:
                num_query_tokens = qtrs_src.shape[0]
                layer, pos = layer_pos_src
                pos = round(pos * num_query_tokens) - num_query_tokens
            activation = qtrs_src[pos, layer]
            results.append(Patch(x=x_src, activation=activation))

    return results


@step(cacheable=True, version="002")
def get_x_GFx_map(task_name: TaskT, model_name: TransformersModelT, dataset: list[CompositionalTask]) -> dict[str, int]:
    """Get mapping from input x to target g(f(x)) token IDs."""
    task = Task(task_name=task_name)
    tokenizer = NNsightModel.get_tokenizer(model_name=model_name)
    dataset_x_GFx = {d.x: select_words_from_query(task, tokenizer, d)["GFx"]["token_ids"][0] for d in dataset}
    return dataset_x_GFx


class QueryWithPatch(TypedDict):
    icq: InContextQuery
    patch: Patch
    dst_position: int
    dst_layer: int
    GF_dst_x_dst: int
    GF_src_x_dst: int
    GF_dst_x_src: int
    GF_src_x_src: int


@step(cacheable=True, version="003", format=TorchFormat())
def sample_patch_per_query(
    queries: list[InContextQuery],
    tasks_are_same: bool,
    x_GFx_src: dict[str, int],
    x_GFx_dst: dict[str, int],
    intermediate_vars: list[IntermediateVariable],
    intermediate_prob_range: tuple[float, float],
    layer_pos_dst: tuple[int, float] | Literal["max"],
    query_tokens_residual_stream: list[torch.Tensor],
    patches: list[Patch],
    seed: int = 0,
) -> list[QueryWithPatch]:
    """Sample one patch per destination query for patching experiments."""
    random.seed(seed)

    results = []
    for icq, var, qtrs in zip(queries, intermediate_vars, query_tokens_residual_stream):
        if not (intermediate_prob_range[0] <= var["prob"] <= intermediate_prob_range[1]):
            continue

        x_dst = icq.query.x

        patches_for_sampling = [
            p
            for p in patches
            if len({x_GFx_dst.get(x_dst), x_GFx_src.get(x_dst), x_GFx_dst.get(p["x"]), x_GFx_src.get(p["x"])} - {None})
            == (2 if tasks_are_same else 4)
        ]

        if len(patches_for_sampling) == 0:
            continue

        sampled_patch = random.sample(patches_for_sampling, 1)[0]

        num_query_tokens = qtrs.shape[0]
        if layer_pos_dst == "max":
            dst_pos = var["position"] - num_query_tokens
            dst_layer = var["layer"]
        else:
            dst_layer, dst_pos = layer_pos_dst
            # supposing pos is float: index (among query tokens) / num_query_tokens
            dst_pos = round(dst_pos * num_query_tokens) - num_query_tokens

        x_src = sampled_patch["x"]
        GF_dst_x_dst = x_GFx_dst[x_dst]
        GF_src_x_dst = x_GFx_src[x_dst]
        GF_dst_x_src = x_GFx_dst[x_src]
        GF_src_x_src = x_GFx_src[x_src]

        results.append(
            QueryWithPatch(
                icq=icq,
                patch=sampled_patch,
                dst_position=dst_pos,
                dst_layer=dst_layer,
                GF_dst_x_dst=GF_dst_x_dst,
                GF_src_x_dst=GF_src_x_dst,
                GF_dst_x_src=GF_dst_x_src,
                GF_src_x_src=GF_src_x_src,
            )
        )

    return results


class TokenProbs(TypedDict):
    GF_dst_x_dst: float
    GF_src_x_dst: float
    GF_dst_x_src: float
    GF_src_x_src: float


@step(cacheable=True, version="004")
def patch_residual_stream(
    model_name: TransformersModelT,
    task_name: TaskT,
    queries_with_patches: list[QueryWithPatch],
    metric: Literal["prob", "reciprocal_rank"] = "reciprocal_rank",
    patch: bool = False,
) -> list[TokenProbs]:
    """Run model with or without activation patching and return token probabilities."""
    model = NNsightModel(model_name=model_name)
    task = Task(task_name=task_name)
    trailing_space = task.trailing_space_in_query(pred_type="GFx")

    results = []

    for qwp in tqdm(queries_with_patches):
        prompt = qwp["icq"].get_prompt("x", "GFx", trailing_space=trailing_space)

        GF_dst_x_dst = qwp["GF_dst_x_dst"]
        GF_src_x_dst = qwp["GF_src_x_dst"]
        GF_dst_x_src = qwp["GF_dst_x_src"]
        GF_src_x_src = qwp["GF_src_x_src"]

        with torch.no_grad():
            if patch is False:
                with model.nnsight_model.trace(prompt):
                    logits = model.lm_head.output[0][-1].cpu().save()
            else:
                dst_layer, dst_position = qwp["dst_layer"], qwp["dst_position"]
                activation = qwp["patch"]["activation"]
                with model.nnsight_model.trace(prompt):
                    model.layers[dst_layer].output[0][0, dst_position, :] = activation
                    logits = model.lm_head.output[0][-1].cpu().save()

        match metric:
            case "prob":
                probs = torch.softmax(logits, dim=0)
                probs = TokenProbs(
                    GF_dst_x_dst=probs[GF_dst_x_dst].item(),
                    GF_src_x_dst=probs[GF_src_x_dst].item(),
                    GF_dst_x_src=probs[GF_dst_x_src].item(),
                    GF_src_x_src=probs[GF_src_x_src].item(),
                )
            case "reciprocal_rank":

                def reciprocal_rank(sort_indices: torch.Tensor, token: int) -> float:
                    rank = (sort_indices == token).nonzero(as_tuple=False)[0, 0].item()
                    return 1 / (rank + 1)

                sort_indices = torch.argsort(logits, descending=True)
                probs = TokenProbs(
                    GF_dst_x_dst=reciprocal_rank(sort_indices, GF_dst_x_dst),
                    GF_src_x_dst=reciprocal_rank(sort_indices, GF_src_x_dst),
                    GF_dst_x_src=reciprocal_rank(sort_indices, GF_dst_x_src),
                    GF_src_x_src=reciprocal_rank(sort_indices, GF_src_x_src),
                )

        results.append(probs)

    return results


@step(cacheable=True, version="002")
def mean_token_probs(token_probs: list[TokenProbs]) -> TokenProbs | None:
    """Compute mean token probabilities across all patching experiments."""
    if len(token_probs) == 0:
        return None

    return TokenProbs(
        GF_dst_x_dst=mean([p["GF_dst_x_dst"] for p in token_probs]),
        GF_src_x_dst=mean([p["GF_src_x_dst"] for p in token_probs]),
        GF_dst_x_src=mean([p["GF_dst_x_src"] for p in token_probs]),
        GF_src_x_src=mean([p["GF_src_x_src"] for p in token_probs]),
    )


@dataclass(kw_only=True)
class PatchingExperiment(Experiment):
    """Activation patching experiment to study causal role of intermediate representations."""

    model_name: TransformersModelT
    task_src: TaskT
    task_dst: TaskT
    var_prob_range_src: tuple[float, float] = (0.5, 1.0)
    var_prob_range_dst: tuple[float, float] = (0.5, 1.0)
    layer_pos_src: tuple[int, float] | Literal["max"] = "max"
    layer_pos_dst: tuple[int, float] | Literal["max"] = "max"
    lens_method: Literal["logit_lens", "token_identity"] = "logit_lens"
    metric: Literal["prob", "reciprocal_rank"] = "reciprocal_rank"
    patch: bool

    @property
    def src_experiment(self) -> LensExperiment:
        """Source logit lens experiment to extract patches from."""
        return LensExperiment(
            model_name=self.model_name,
            task_name=self.task_src,
            lens_method=self.lens_method,
            metric=self.metric,
        )

    @property
    def dst_experiment(self) -> LensExperiment:
        """Destination logit lens experiment to patch into."""
        return LensExperiment(
            model_name=self.model_name,
            task_name=self.task_dst,
            lens_method=self.lens_method,
            metric=self.metric,
        )

    @property
    def dependencies(self) -> Sequence[Experiment]:
        return [self.src_experiment, self.dst_experiment]

    @property
    def step_dict(self) -> dict[str, Step]:
        """Activation patching pipeline combining source and destination logit lens pipelines."""
        src_experiment = self.src_experiment
        dst_experiment = self.dst_experiment
        src_task_steps = {f"src_{k}": v for k, v in src_experiment.step_dict.items()}
        dst_task_steps = {f"dst_{k}": v for k, v in dst_experiment.step_dict.items()}
        steps = src_task_steps | dst_task_steps

        # Step: Extract patches from source task (list[Patch])
        steps["patches_from_src"] = select_patches_from_source(
            queries_src=steps["src_selected_queries"],
            queries_dst=steps["dst_selected_queries"],
            intermediate_vars_src=steps["src_intermediate_vars"],
            vars_prob_range=self.var_prob_range_src,
            layer_pos_src=self.layer_pos_src,
            query_tokens_residual_stream_src=steps["src_query_tokens_residual_stream"],
        )

        # Step: Get x to g(f(x)) token mappings (dict[str, int])
        steps["src_x_GFx"] = get_x_GFx_map(
            task_name=src_experiment.task_name,
            model_name=src_experiment.model_name,
            dataset=steps["src_dataset"],
        )

        steps["dst_x_GFx"] = get_x_GFx_map(
            task_name=dst_experiment.task_name,
            model_name=dst_experiment.model_name,
            dataset=steps["dst_dataset"],
        )

        # Step: Sample patches for destination queries (list[QueryWithPatch])
        steps["queries_with_patches"] = sample_patch_per_query(
            queries=steps["dst_selected_queries"],
            tasks_are_same=(src_experiment.task_name == dst_experiment.task_name),
            x_GFx_src=steps["src_x_GFx"],
            x_GFx_dst=steps["dst_x_GFx"],
            intermediate_vars=steps["dst_intermediate_vars"],
            intermediate_prob_range=self.var_prob_range_dst,
            layer_pos_dst=self.layer_pos_dst,
            query_tokens_residual_stream=steps["dst_query_tokens_residual_stream"],
            patches=steps["patches_from_src"],
        )

        # Step: Run patching experiment (list[TokenProbs])
        steps["token_probabilities"] = patch_residual_stream(
            model_name=self.model_name,
            task_name=self.task_dst,
            queries_with_patches=steps["queries_with_patches"],
            metric=self.metric,
            patch=self.patch,
        )

        # Step: Compute mean probabilities (TokenProbs | None)
        steps["mean_probabilities"] = mean_token_probs(token_probs=steps["token_probabilities"])

        return steps

    def results(self) -> dict:
        return {
            **(self.step_result("mean_probabilities") or {}),
            "num_patches": len(self.step_result("patches_from_src")),
            "num_queries": len(self.step_result("queries_with_patches")),
        }


@dataclass(kw_only=True)
class PatchingSweep(Sweep[PatchingExperiment]):
    tasks_src_dst: list[tuple[TaskT, TaskT]]
    var_prob_range_src: tuple[float, float] = (0.5, 1.0)
    var_prob_range_dst: tuple[float, float] = (0.5, 1.0)
    layer_pos_dst: tuple[int, float] | Literal["max"] = "max"
    model: TransformersModelT

    @property
    def experiments(self) -> Sequence[PatchingExperiment]:
        return [
            PatchingExperiment(
                task_src=task_src,
                task_dst=task_dst,
                var_prob_range_src=self.var_prob_range_src,
                var_prob_range_dst=self.var_prob_range_dst,
                layer_pos_dst=self.layer_pos_dst,
                model_name=self.model,
                patch=patch,
            )
            for task_src, task_dst in self.tasks_src_dst
            for patch in [False, True]
        ]


if __name__ == "__main__":
    PatchingExperiment.cli()
