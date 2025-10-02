from dataclasses import dataclass
from typing import Literal, Sequence

from ..models.transformers import TransformersModelT
from ..tasks import TaskT
from .lens import LensExperiment
from .linear_task_embedding import LinearTaskEmbeddingExperiment
from .utils import Experiment, Step, Sweep


@dataclass(kw_only=True)
class TaskLensCorrelationExperiment(LensExperiment):
    """Embedding-space task linearity and aggregate presence of intermediate variable for one task."""

    model_name: TransformersModelT  # pyright: ignore [reportIncompatibleVariableOverride]
    train_examples: int = 100

    @property
    def task_linearity_experiment(self) -> LinearTaskEmbeddingExperiment:
        """Linear embedding analysis for x to g(f(x)) mapping."""
        return LinearTaskEmbeddingExperiment(
            task_name=self.task_name,
            model_name=self.model_name,
            train_examples=self.train_examples,
            query_type="x",
            pred_type="GFx",
        )

    @property
    def dependencies(self) -> Sequence[Experiment]:
        return [self.task_linearity_experiment]

    @property
    def step_dict(self) -> dict[str, Step]:
        """Same as logit lens pipeline."""
        return super().step_dict

    def results(self) -> dict:
        return {
            **super().results(),
            "task_linearity": self.task_linearity_experiment.step_result("test_similarity"),
            "accuracy": self.step_result("results_x_GFx")["accuracy"],
        }


@dataclass
class TaskLensCorrelationSweep(Sweep[TaskLensCorrelationExperiment]):
    """Correlation across tasks of linearity and aggregate presence of intermediate variables."""

    tasks: list[TaskT]
    models: list[TransformersModelT]
    # linear task embedding parameters
    train_examples: int = 100
    # lens parameters
    lens_method: Literal["logit_lens", "token_identity"] = "logit_lens"

    @property
    def experiments(self) -> Sequence[TaskLensCorrelationExperiment]:
        return [
            TaskLensCorrelationExperiment(
                task_name=task_name,
                model_name=model_name,
                train_examples=self.train_examples,
                lens_method=self.lens_method,
            )
            for task_name in self.tasks
            for model_name in self.models
        ]


if __name__ == "__main__":
    TaskLensCorrelationExperiment.cli()
