from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Literal, Sequence, TypedDict

import polars as pl
import torch
from tqdm import tqdm

from .. import lens
from ..models.transformers import NNsightModel, TransformersModelT
from ..tasks import NodeT, Task, TaskT
from .residual_stream import ResidualStreamExperiment, Word
from .utils import Step, Sweep, step


@step(cacheable=True, version="001")
def get_top_k_tokens(
    model_name: TransformersModelT,
    residual_stream: list[torch.Tensor],
    k: int = 5,
    lens_method: Literal["logit_lens", "token_identity"] = "logit_lens",
) -> list[pl.DataFrame]:
    """Extract top-k token predictions at each token position and layer."""
    model = NNsightModel(model_name=model_name)
    tokenizer = model.tokenizer

    topk_tokens_per_query = []

    for activations in tqdm(residual_stream):
        # logits: [positions, layers, vocab_size]
        match lens_method:
            case "logit_lens":
                logits = lens.logit_lens(
                    model=model,
                    residual_stream_acts=activations,
                )
            case "token_identity":
                logits = lens.token_identity(
                    model=model,
                    residual_stream_acts=activations,
                )
        probs = torch.softmax(logits, dim=2)

        # shapes: (positions, layers, k)
        topk_probs, topk_token_ids = torch.topk(probs, k=k, dim=2, largest=True, sorted=True)

        topk_tokens_per_query.append(
            pl.DataFrame(
                [
                    {
                        "position": pos,
                        "layer": layer,
                        "top_k": k_idx,
                        "token": tokenizer.decode(topk_token_ids[pos, layer, k_idx].item()),
                        "prob": topk_probs[pos, layer, k_idx].item(),
                    }
                    for pos in range(topk_probs.shape[0])
                    for layer in range(topk_probs.shape[1])
                    for k_idx in range(k)
                ]
            )
        )

    return topk_tokens_per_query


@step(cacheable=True, version="002")
def get_query_token_probs(
    model_name: TransformersModelT,
    residual_stream: list[torch.Tensor],
    selected_words: list[dict[NodeT, Word]],
    lens_method: Literal["logit_lens", "token_identity"] = "logit_lens",
) -> list[pl.DataFrame]:
    """Compute probabilities and ranks of variables at each token position and layer."""
    model = NNsightModel(model_name=model_name)

    probs_per_query = []

    for example_idx in tqdm(range(len(residual_stream))):
        activations = residual_stream[example_idx]
        selections = selected_words[example_idx]

        # logits: [positions, layers, vocab_size]
        match lens_method:
            case "logit_lens":
                logits = lens.logit_lens(
                    model=model,
                    residual_stream_acts=activations,
                )
            case "token_identity":
                logits = lens.token_identity(
                    model=model,
                    residual_stream_acts=activations,
                )

        P, L, V = logits.shape

        probs = torch.softmax(logits, dim=2).detach().cpu()
        sort_indices = lens.argsort_logits(logits)

        probs_for_selected_words = []
        for node_type, word in selections.items():
            first_token_id = word["token_ids"][0]
            rr = lens.reciprocal_rank(sort_indices=sort_indices, shape=(P, L), token=first_token_id)

            for pos in range(P):
                for layer in range(L):
                    probs_for_selected_words.append(
                        {
                            "position": pos,
                            "layer": layer,
                            "node_type": node_type,
                            "prob": probs[pos, layer, first_token_id].item(),
                            "reciprocal_rank": rr[pos, layer].item(),
                        }
                    )

        probs_per_query.append(pl.DataFrame(probs_for_selected_words))

    return probs_per_query


@step(cacheable=True, version="001")
def compute_processing_signatures(
    query_token_probs: list[pl.DataFrame],
    metric: Literal["prob", "reciprocal_rank"] = "reciprocal_rank",
) -> list[pl.DataFrame]:
    """Compute processing signatures for each query by taking max across positions."""
    return [df.group_by("layer", "node_type").agg(pl.col(metric).max()) for df in query_token_probs]


@step(cacheable=True, version="001")
def aggregate_processing_signatures(
    processing_signatures: list[pl.DataFrame],
    metric: Literal["prob", "reciprocal_rank"] = "reciprocal_rank",
) -> pl.DataFrame | None:
    """Aggregate processing signatures across all queries for task-level analysis."""
    if len(processing_signatures) == 0:
        return None
    return pl.concat(processing_signatures).group_by("layer", "node_type").agg(pl.col(metric).mean())


@step(cacheable=True, version="001")
def max_probs_per_node(
    query_token_probs: list[pl.DataFrame],
    metric: Literal["prob", "reciprocal_rank"] = "reciprocal_rank",
) -> list[pl.DataFrame]:
    """Find the position and layer where each variable has maximum representation."""
    return [
        (
            df.sort(metric, descending=True)
            .group_by("node_type")
            .first()
            .select("node_type", "position", "layer", metric)
        )
        for df in query_token_probs
    ]


class IntermediateVariable(TypedDict):
    node_type: str
    position: int
    layer: int
    prob: float


@step(cacheable=True, version="002")
def intermediate_vars(
    task_name: TaskT, probs_per_node: list[pl.DataFrame], metric: Literal["prob", "reciprocal_rank"]
) -> list[IntermediateVariable]:
    """Extract the strongest intermediate variable representation for each query."""
    intermediate_variables = Task(task_name).correct_intermediate_nodes
    results = []
    for df in probs_per_node:
        node_type, position, layer, prob = (
            df.select("node_type", "position", "layer", metric)
            .filter(pl.col("node_type").is_in(intermediate_variables))
            .sort(metric, descending=True)
            .row(0)
        )
        results.append(IntermediateVariable(node_type=node_type, position=position, layer=layer, prob=prob))
    return results


@step(cacheable=True, version="001")
def aggregate_intermediate_var_probs(vars: list[IntermediateVariable]) -> float | None:
    """Compute mean intermediate variable signal strength across queries."""
    if len(vars) == 0:
        return None
    return mean([var["prob"] for var in vars])


@dataclass
class LensExperiment(ResidualStreamExperiment):
    lens_method: Literal["logit_lens", "token_identity"] = "logit_lens"
    top_k: int = 5
    metric: Literal["prob", "reciprocal_rank"] = "reciprocal_rank"

    @property
    def step_dict(self) -> dict[str, Step]:
        """Interpretability analysis pipeline extending activation extraction pipeline."""
        steps = super().step_dict

        # Step 12: Extract top-k tokens for interpretability overview (list[pl.DataFrame])
        steps["top_k_tokens"] = get_top_k_tokens(
            model_name=self.model_name,
            residual_stream=steps["query_tokens_residual_stream"],
            k=self.top_k,
            lens_method=self.lens_method,
        )

        # Step 13: Track specific task variables across layers (list[pl.DataFrame])
        steps["query_token_probs"] = get_query_token_probs(
            model_name=self.model_name,
            residual_stream=steps["query_tokens_residual_stream"],
            selected_words=steps["selected_words"],
            lens_method=self.lens_method,
        )

        # Step 14: Create processing signatures per query (list[pl.DataFrame])
        steps["processing_signatures"] = compute_processing_signatures(query_token_probs=steps["query_token_probs"])

        # Step 15: Aggregate signatures across queries (pl.DataFrame | None)
        steps["aggregate_processing_signature"] = aggregate_processing_signatures(
            processing_signatures=steps["processing_signatures"]
        )

        # Step 16: Find strongest representation locations (list[pl.DataFrame])
        steps["max_probs_per_node"] = max_probs_per_node(query_token_probs=steps["query_token_probs"])

        # Step 17: Extract intermediate variable metrics (list[IntermediateVariable])
        steps["intermediate_vars"] = intermediate_vars(
            task_name=self.task_name,
            probs_per_node=steps["max_probs_per_node"],
            metric=self.metric,
        )

        # Step 18: Compute task-level compositional score (float | None)
        steps["aggregate_intermediate_var_prob"] = aggregate_intermediate_var_probs(vars=steps["intermediate_vars"])

        return steps

    def results(self) -> dict:
        intermediate_vars = self.step_result("intermediate_vars")

        if len(intermediate_vars) == 0:
            intermediate_node_per_query = None
            intermediate_metric_per_query = None
        else:
            intermediate_node_per_query = [var["node_type"] for var in intermediate_vars]
            intermediate_metric_per_query = [var["prob"] for var in intermediate_vars]

        return {
            **asdict(self),
            "count": len(intermediate_vars),
            "intermediate_node_per_query": intermediate_node_per_query,
            "intermediate_metric_per_query": intermediate_metric_per_query,
            "intermediate_metric_aggregated": self.step_result("aggregate_intermediate_var_prob"),
        }


@dataclass
class LensSweep(Sweep[LensExperiment]):
    tasks: list[TaskT]
    models: list[TransformersModelT]
    lens_method: Literal["logit_lens", "token_identity"] = "logit_lens"
    top_k: int = 5
    metric: Literal["prob", "reciprocal_rank"] = "reciprocal_rank"
    icl_examples: int = 10
    seed: int = 0

    @property
    def experiments(self) -> Sequence[LensExperiment]:
        """Lens analysis experiments for all task–model–correctness combinations."""
        return [
            LensExperiment(
                task_name=task_name,
                model_name=model_name,
                lens_method=self.lens_method,
                top_k=self.top_k,
                metric=self.metric,
                icl_examples=self.icl_examples,
                seed=self.seed,
                correctness=correctness,
            )
            for task_name in self.tasks
            for model_name in self.models
            for correctness in [True, False]
        ]

    def median_intermediate_var_location(self, prob_range: tuple[float, float] = (0.5, 1.0)) -> tuple[float, int]:
        """Compute median location of intermediate variable representations."""
        positions, layers = [], []
        for e in self.experiments:
            if not e.correctness:
                continue
            vars: list[IntermediateVariable] = e.step_result("intermediate_vars")
            query_tokens_residual_stream: list[torch.Tensor] = e.step_result("query_tokens_residual_stream")
            for var, qtrs in zip(vars, query_tokens_residual_stream):
                if prob_range[0] <= var["prob"] <= prob_range[1]:
                    positions.append(var["position"] / qtrs.shape[0])  # position (in query tokens) / num query tokens
                    layers.append(var["layer"])
        return median(positions), median(layers)

    def overall_processing_signature(self, correctness: bool):
        """Create overall processing signature across all experiments."""
        return (
            pl.concat(
                [
                    signature
                    for e in self.experiments
                    for signature in e.step_result("processing_signatures")
                    if e.correctness == correctness
                ]
            )
            .group_by("layer", "node_type")
            .agg(pl.col(self.metric).mean())
        )


if __name__ == "__main__":
    LensExperiment.cli()
