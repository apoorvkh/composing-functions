from dataclasses import dataclass
from typing import Sequence, TypedDict

import torch
from tango.integrations.torch import TorchFormat
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from .. import lens
from ..models.transformers import NNsightModel, TransformersModelT
from ..prompting import InContextQuery
from ..tasks import CompositionalTask, NodeT, Task, TaskT
from .evaluate_task import EvaluateTaskExperiment
from .utils import Step, Sweep, step


class Word(TypedDict):
    word: str
    token_ids: list[int]
    tokens: list[str]


def select_words_from_query(
    task: Task,
    tokenizer: PreTrainedTokenizerFast | PreTrainedTokenizer,
    query: CompositionalTask,
) -> dict[NodeT, Word]:
    """Extract tokenization info for all variables in a compositional task."""
    selected_words = {}
    for node_type in task.nodes:
        leading_space = task.leading_space(node_type=node_type)
        field_value = query.get(node_type, leading_space=leading_space)
        token_ids = tokenizer.encode(text=field_value, add_special_tokens=False)
        tokens = [tokenizer.decode(t) for t in token_ids]
        selected_words[node_type] = Word(word=field_value, token_ids=token_ids, tokens=tokens)
    return selected_words


@step(cacheable=True, version="001")
def filter_out_first_token_overlaps(
    model_name: TransformersModelT,
    task_name: TaskT,
    in_context_queries: list[InContextQuery],
) -> list[InContextQuery]:
    """Filter queries to avoid confounding first-token overlaps."""
    tokenizer = NNsightModel.get_tokenizer(model_name=model_name)
    task = Task(task_name=task_name)

    selected_queries = []

    for icq in in_context_queries:
        selected_words: dict[NodeT, Word] = select_words_from_query(task, tokenizer, icq.query)

        # skip if any fields have the same first token
        first_tokens = [word["token_ids"][0] for word in selected_words.values()]
        if len(first_tokens) != len(set(first_tokens)):
            continue

        # skip if any first tokens are in x
        first_tokens = [word["token_ids"][0] for node_type, word in selected_words.items() if node_type != "x"]
        x_tokens = selected_words["x"]["token_ids"]
        if len(set(first_tokens) & set(x_tokens)) > 0:
            continue

        selected_queries.append(icq)

    return selected_queries


@step(cacheable=True, version="001")
def select_words(
    model_name: TransformersModelT,
    task_name: TaskT,
    in_context_queries: list[InContextQuery],
) -> list[dict[NodeT, Word]]:
    """Extract tokenization info for all variables across all queries."""
    tokenizer = NNsightModel.get_tokenizer(model_name=model_name)
    task = Task(task_name=task_name)
    return [select_words_from_query(task, tokenizer, icq.query) for icq in in_context_queries]


@step(cacheable=True, version="001", format=TorchFormat())
def get_residual_stream_activations(
    model_name: TransformersModelT,
    task_name: TaskT,
    in_context_queries: list[InContextQuery],
    query_type: NodeT = "x",
    pred_type: NodeT = "GFx",
) -> list[torch.Tensor]:
    """Extract residual stream activations (torch.Tensor: [positions, layers, hidden_dim]) for each query."""
    model = NNsightModel(model_name=model_name)
    task = Task(task_name=task_name)
    trailing_space = task.trailing_space_in_query(pred_type=pred_type)
    prompts = [icq.get_prompt(query_type, pred_type, trailing_space=trailing_space) for icq in in_context_queries]
    return [lens.residual_stream(model=model, prompt=p) for p in prompts]


@step(cacheable=True, version="001", format=TorchFormat())
def residual_stream_for_query_tokens(
    model_name: TransformersModelT,
    task_name: TaskT,
    residual_stream: list[torch.Tensor],
    in_context_queries: list[InContextQuery],
    query_type: NodeT = "x",
    pred_type: NodeT = "GFx",
) -> list[torch.Tensor]:
    """Select residual stream activations of query tokens (where directed computation occurs)."""
    tokenizer = NNsightModel.get_tokenizer(model_name=model_name)
    trailing_space = Task(task_name=task_name).trailing_space_in_query(pred_type=pred_type)

    def first_query_token_index(example_idx: int) -> int:
        """Find where the query starts in the full prompt."""
        icq = in_context_queries[example_idx]
        query_len = len(f"{icq.query.x}\nA:")

        prompt = icq.get_prompt(
            query_type,
            pred_type,
            trailing_space=trailing_space,
        )
        prompt_tokens = tokenizer.encode(text=prompt, add_special_tokens=False)

        # Count characters in prompt tokens (right to left) until query is "covered"
        for i in range(len(prompt_tokens) - 1, -1, -1):
            query_len -= len(tokenizer.decode(prompt_tokens[i]))
            if query_len <= 0:
                return i
        return 0

    return [rs[first_query_token_index(i) :, :, :] for i, rs in enumerate(residual_stream)]


@dataclass
class ResidualStreamExperiment(EvaluateTaskExperiment):
    query_type: NodeT = "x"
    pred_type: NodeT = "GFx"
    correctness: bool = True

    use_nnsight: bool = True
    num_queries: int | None = None

    @property
    def step_dict(self) -> dict[str, Step]:
        """Activation extraction pipeline extending evaluation pipeline."""
        steps = super().step_dict

        # Step 8: Filter queries to avoid token overlap confounds (list[InContextQuery])
        steps["selected_queries"] = filter_out_first_token_overlaps(
            model_name=self.model_name,
            task_name=self.task_name,
            in_context_queries=steps["correct_queries" if self.correctness else "incorrect_queries"],
        )

        # Step 9: Extract variable tokenization info (list[dict[NodeT, Word]])
        steps["selected_words"] = select_words(
            model_name=self.model_name,
            task_name=self.task_name,
            in_context_queries=steps["selected_queries"],
        )

        # Step 10: Extract residual stream activations (list[torch.Tensor])
        steps["residual_stream"] = get_residual_stream_activations(
            model_name=self.model_name,
            task_name=self.task_name,
            in_context_queries=steps["selected_queries"],
            query_type=self.query_type,
            pred_type=self.pred_type,
        )

        # Step 11: Focus on query tokens where computation occurs (list[torch.Tensor])
        steps["query_tokens_residual_stream"] = residual_stream_for_query_tokens(
            model_name=self.model_name,
            task_name=self.task_name,
            residual_stream=steps["residual_stream"],
            in_context_queries=steps["selected_queries"],
            query_type=self.query_type,
            pred_type=self.pred_type,
        )

        return steps

    def results(self) -> dict:
        return {
            "num_selected_queries": len(self.step_result("selected_queries")),
        }


@dataclass
class ResidualStreamSweep(Sweep[ResidualStreamExperiment]):
    tasks: list[TaskT]
    models: list[TransformersModelT]
    icl_examples: int = 10
    seed: int = 0

    @property
    def experiments(self) -> Sequence[ResidualStreamExperiment]:
        return [
            ResidualStreamExperiment(
                task_name=task_name,
                model_name=model_name,
                icl_examples=self.icl_examples,
                seed=self.seed,
                correctness=correctness,
            )
            for task_name in self.tasks
            for model_name in self.models
            for correctness in [True, False]
        ]


if __name__ == "__main__":
    ResidualStreamExperiment.cli()
