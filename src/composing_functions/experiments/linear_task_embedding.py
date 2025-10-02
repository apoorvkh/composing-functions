import dataclasses
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F

from ..models.transformers import TransformersModel, TransformersModelT
from ..tasks import CompositionalTask, NodeT, Task, TaskT
from .generate_data import GenerateDataExperiment
from .utils import Step, Sweep, step


@step(cacheable=False, version="001")
def get_embeddings(
    task_name: TaskT,
    model_name: TransformersModelT,
    query_type: NodeT,
    pred_type: NodeT,
    dataset: list[CompositionalTask],
    range: tuple[int, int | None] = (0, None),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Get mean embedding for each query word and first-token unembedding for each pred word."""
    task = Task(task_name=task_name)
    model = TransformersModel(model_name=model_name)
    tokenizer = model.tokenizer
    dataset = dataset[range[0] : range[1]]

    # Mean embeddings of all tokens for query words

    model_embeddings: torch.Tensor = (
        model.model.get_input_embeddings().weight.detach().cpu()  # type: ignore
    )
    query_words = [
        d.get(
            node_type=query_type,
            leading_space=task.leading_space(node_type=query_type),
        )
        for d in dataset
    ]
    query_embeddings = torch.stack(
        [
            torch.stack([model_embeddings[t] for t in tokenizer.encode(text=w, add_special_tokens=False)]).mean(dim=0)
            for w in query_words
        ]
    )

    # Unembeddings of first tokens for pred words

    model_unembeddings: torch.Tensor = (
        model.model.get_output_embeddings().weight.detach().cpu()  # type: ignore
    )
    pred_words = [d.get(node_type=pred_type, leading_space=task.leading_space(node_type=pred_type)) for d in dataset]
    pred_first_tokens = [tokenizer.encode(text=w, add_special_tokens=False)[0] for w in pred_words]
    pred_embeddings = torch.stack([model_unembeddings[t] for t in pred_first_tokens])

    return query_embeddings, pred_embeddings


@step(cacheable=False, version="001")
def compute_transformation(
    embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Fit linear transformation from query embeddings to pred embeddings."""
    x, y = embeddings
    x = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)  # bias term
    return torch.linalg.lstsq(x, y).solution


@step(cacheable=True, version="003")
def transformation_similarity(
    embeddings: tuple[torch.Tensor, torch.Tensor],
    transformation: torch.Tensor,
) -> float:
    """Compute cosine similarity between transformed inputs and target embeddings."""
    x, y = embeddings
    x = torch.cat([x, torch.ones(x.shape[0], 1, device=x.device)], dim=1)  # bias term
    transformed_inputs = x @ transformation
    similarities = F.cosine_similarity(transformed_inputs, y, dim=1)
    similarities = (similarities + 1) / 2  # normalize to [0, 1]
    return similarities.mean().item()


@dataclass(kw_only=True)
class LinearTaskEmbeddingExperiment(GenerateDataExperiment):
    model_name: TransformersModelT
    train_examples: int = 100
    query_type: NodeT = "x"
    pred_type: NodeT = "GFx"

    @property
    def step_dict(self) -> dict[str, Step]:
        """Extends data generation pipeline."""
        steps = super().step_dict

        # Step 3: Get embeddings for training examples (tuple[torch.Tensor, torch.Tensor])
        steps["train_embeddings"] = get_embeddings(
            task_name=self.task_name,
            model_name=self.model_name,
            query_type=self.query_type,
            pred_type=self.pred_type,
            dataset=steps["dataset"],
            range=(0, self.train_examples),
        )

        # Step 4: Get embeddings for test examples (tuple[torch.Tensor, torch.Tensor])
        steps["test_embeddings"] = get_embeddings(
            task_name=self.task_name,
            model_name=self.model_name,
            query_type=self.query_type,
            pred_type=self.pred_type,
            dataset=steps["dataset"],
            range=(self.train_examples, None),
        )

        # Step 5: Fit linear transformation on training data (torch.Tensor)
        steps["transformation"] = compute_transformation(embeddings=steps["train_embeddings"])

        # Step 6-7: Compute similarity on train and test sets (float)
        steps["train_similarity"] = transformation_similarity(
            embeddings=steps["train_embeddings"],
            transformation=steps["transformation"],
        )
        steps["test_similarity"] = transformation_similarity(
            embeddings=steps["test_embeddings"],
            transformation=steps["transformation"],
        )

        return steps

    def results(self) -> dict:
        return {
            **dataclasses.asdict(self),
            "train_similarity": self.step_result("train_similarity"),
            "test_similarity": self.step_result("test_similarity"),
        }


@dataclass
class LinearTaskEmbeddingSweep(Sweep[LinearTaskEmbeddingExperiment]):
    tasks: list[TaskT]
    models: list[TransformersModelT]
    train_examples: int = 100

    @property
    def experiments(self) -> Sequence[LinearTaskEmbeddingExperiment]:
        return [
            LinearTaskEmbeddingExperiment(
                task_name=task_name,
                model_name=model_name,
                train_examples=self.train_examples,
                query_type=query_type,
                pred_type=pred_type,
            )
            for task_name in self.tasks
            for model_name in self.models
            for query_type, pred_type in [*Task(task_name).correct_hops, ("x", "GFx")]
        ]


if __name__ == "__main__":
    LinearTaskEmbeddingExperiment.cli()
