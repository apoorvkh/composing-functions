from dataclasses import dataclass
from statistics import mean
from typing import Literal, Sequence

from ..models.transformers import TransformersModelT
from ..tasks import NodeT, Task, TaskT
from .lens import LensExperiment
from .linear_task_embedding import LinearTaskEmbeddingExperiment
from .utils import Experiment, Step, Sweep


@dataclass(kw_only=True)
class HopLensCorrelationExperiment(LensExperiment):
    """Individual hop linearity and presence of variable for one task and specified intermediate variable."""

    model_name: TransformersModelT  # pyright: ignore [reportIncompatibleVariableOverride]
    # linear task embedding parameters
    train_examples: int = 100
    intermediate_node: NodeT = "Fx"

    @property
    def first_hop_linearity_experiment(self) -> LinearTaskEmbeddingExperiment:
        """Linear embedding analysis for x to intermediate hop."""
        return LinearTaskEmbeddingExperiment(
            task_name=self.task_name,
            model_name=self.model_name,
            train_examples=self.train_examples,
            query_type="x",
            pred_type=self.intermediate_node,
        )

    @property
    def second_hop_linearity_experiment(self) -> LinearTaskEmbeddingExperiment:
        """Linear embedding analysis for intermediate to g(f(x)) hop."""
        return LinearTaskEmbeddingExperiment(
            task_name=self.task_name,
            model_name=self.model_name,
            train_examples=self.train_examples,
            query_type=self.intermediate_node,
            pred_type="GFx",
        )

    @property
    def dependencies(self) -> Sequence[Experiment]:
        return [
            self.first_hop_linearity_experiment,
            self.second_hop_linearity_experiment,
        ]

    @property
    def step_dict(self) -> dict[str, Step]:
        """Same as logit lens pipeline."""
        return super().step_dict

    def results(self) -> dict:
        results = super().results()

        intermediate_metric_per_query = [
            metric
            for node, metric in zip(
                results["intermediate_node_per_query"] or [],
                results["intermediate_metric_per_query"] or [],
            )
            if node == self.intermediate_node
        ]
        intermediate_metric = mean(intermediate_metric_per_query) if len(intermediate_metric_per_query) > 0 else None

        first_hop_linearity = self.first_hop_linearity_experiment.step_result("test_similarity")
        second_hop_linearity = self.second_hop_linearity_experiment.step_result("test_similarity")

        return {
            **super().results(),
            "intermediate_metric": intermediate_metric,
            "first_hop_linearity": first_hop_linearity,
            "second_hop_linearity": second_hop_linearity,
            "min_hop_linearity": min(first_hop_linearity, second_hop_linearity),
            "max_hop_linearity": max(first_hop_linearity, second_hop_linearity),
            "accuracy": self.step_result("results_x_GFx")["accuracy"],
        }


@dataclass
class HopLensCorrelationSweep(Sweep[HopLensCorrelationExperiment]):
    """Correlation of hop linearity and presence of intermediate variable across tasks."""

    tasks: list[TaskT]
    models: list[TransformersModelT]
    # linear task embedding parameters
    train_examples: int = 100
    # lens parameters
    lens_method: Literal["logit_lens", "token_identity"] = "logit_lens"

    @property
    def experiments(self) -> Sequence[HopLensCorrelationExperiment]:
        return [
            HopLensCorrelationExperiment(
                task_name=task_name,
                model_name=model_name,
                train_examples=self.train_examples,
                lens_method=self.lens_method,
                intermediate_node=intermediate_node,
            )
            for task_name in self.tasks
            for model_name in self.models
            for intermediate_node in Task(task_name).correct_intermediate_nodes
        ]


if __name__ == "__main__":
    HopLensCorrelationExperiment.cli()
