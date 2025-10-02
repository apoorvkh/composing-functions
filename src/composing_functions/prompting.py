"""In-context learning and prompting utilities for compositional tasks.

This module provides tools for creating prompts and evaluating model performance
on compositional reasoning tasks using in-context learning.
"""

from dataclasses import dataclass

from .models import TextGenerationModel
from .models.api import APIModel
from .tasks import CompositionalTask, NodeT, Task, TaskT


@dataclass
class InContextQuery:
    """Container for in-context learning queries with examples (context) and target query.

    Args:
        context: List of example tasks to provide as context
        query: Target task to be evaluated
    """

    context: list[CompositionalTask]
    query: CompositionalTask

    def get_prompt(
        self,
        query_type: NodeT = "x",
        pred_type: NodeT = "GFx",
        prompt_prefix: str = "",
        prompt_sep: str = "\n\n",
        trailing_space: bool = False,
    ) -> str:
        """Generate formatted prompt string for in-context learning."""
        prompt = prompt_prefix
        for example in self.context:
            prompt += f"Q: {example.get(query_type)}\nA: {example.get(pred_type)}{prompt_sep}"
        prompt += f"Q: {self.query.get(query_type)}\nA:"
        if trailing_space:
            prompt += " "
        return prompt


@dataclass
class Prediction:
    """Result with prompt, model prediction, and ground truth label"""

    prompt: str
    pred: str
    label: str


def get_prediction(
    model: TextGenerationModel,
    task_name: TaskT,
    icq: InContextQuery,
    query_type: NodeT = "x",
    pred_type: NodeT = "GFx",
) -> Prediction:
    prompt_sep = "\n\n"

    if isinstance(model, APIModel):
        # API models don't always return text with leading spaces
        # We str.lstrip() the prediction for consistency
        trailing_space_in_query = False
        leading_space_in_label = False
    else:
        task = Task(task_name=task_name)
        trailing_space_in_query = task.trailing_space_in_query(pred_type=pred_type)
        leading_space_in_label = task.leading_space(node_type=pred_type)

    prompt = icq.get_prompt(
        query_type=query_type,
        pred_type=pred_type,
        prompt_sep=prompt_sep,
        trailing_space=trailing_space_in_query,
    )
    label = icq.query.get(pred_type, leading_space=leading_space_in_label)
    pred = model.generate_continuation(text=prompt, max_new_tokens=20, stop_seq=prompt_sep)

    if isinstance(model, APIModel):
        pred = pred.rpartition("A: ")[2]

    return Prediction(prompt=prompt, pred=pred, label=label)
