import random
from dataclasses import dataclass
from statistics import mean
from typing import Sequence, TypedDict

from tqdm import tqdm

from ..models import ModelT, get_model
from ..prompting import (
    InContextQuery,
    Prediction,
    get_prediction,
)
from ..tasks import HopT, NodeT, Task, TaskT
from .generate_data import GenerateDataExperiment
from .utils import Step, Sweep, step


@step(cacheable=False, version="001")
def sample_queries(
    in_context_queries: list[InContextQuery],
    num_examples: int | None = None,
    seed: int = 0,
) -> list[InContextQuery]:
    """Sample subset of queries for evaluation."""
    num_examples = num_examples or len(in_context_queries)
    random.seed(seed)
    return random.sample(in_context_queries, num_examples)


class EvaluationResults(TypedDict):
    predictions: list[Prediction]
    accuracy: float


@step(cacheable=True, version="001")
def evaluate(
    task_name: TaskT,
    in_context_queries: list[InContextQuery],
    model_name: ModelT,
    use_nnsight: bool = False,
    query_type: NodeT = "x",
    pred_type: NodeT = "GFx",
) -> EvaluationResults:
    """
    Evaluate model on specific hop type (e.g., x to f(x) or x to g(f(x))).
    Returns predictions and accuracy for compositionality gap analysis.
    """
    model = get_model(model_name=model_name, use_nnsight=use_nnsight)
    predictions = [
        get_prediction(model, task_name, icq, query_type=query_type, pred_type=pred_type)
        for icq in tqdm(in_context_queries)
    ]
    accuracy = mean([float(prediction.pred == prediction.label) for prediction in predictions])
    return EvaluationResults(predictions=predictions, accuracy=accuracy)


class PercentHopsCorrect(TypedDict):
    at_least_one_primitive: float
    all_primitives: float
    all_primitives_and_composition: float


@step(cacheable=True, version="001")
def percent_hops_correct(
    composition_results: EvaluationResults,
    **results_for_hops: EvaluationResults,
) -> PercentHopsCorrect:
    """Compute compositionality gap statistics."""
    indices_for_hops = [
        {i for i, p in enumerate(hop_results["predictions"]) if p.pred == p.label}
        for hop_results in results_for_hops.values()
    ]

    at_least_one_correct = set.union(*indices_for_hops)
    all_primitives_correct = set.intersection(*indices_for_hops)

    indices_x_GFx = {i for i, p in enumerate(composition_results["predictions"]) if p.pred == p.label}
    all_correct = all_primitives_correct & indices_x_GFx

    total = len(composition_results["predictions"])

    return PercentHopsCorrect(
        at_least_one_primitive=len(at_least_one_correct) / total,
        all_primitives=len(all_primitives_correct) / total,
        all_primitives_and_composition=len(all_correct) / total,
    )


@step(cacheable=True, version="001")
def filter_queries_by_correctness(
    queries: list[InContextQuery],
    return_correct_compositions: bool,
    composition_results: EvaluationResults,
    **results_for_hops: EvaluationResults,
) -> list[InContextQuery]:
    """Get queries where all hops are correct; and the composition is either correct or incorrect."""
    indices_all_hops_correct = set.intersection(
        *[
            {i for i, p in enumerate(hop_results["predictions"]) if p.pred == p.label}
            for hop_results in results_for_hops.values()
        ]
    )

    indices_x_GFx_correct = {i for i, p in enumerate(composition_results["predictions"]) if p.pred == p.label}

    if return_correct_compositions:
        indices = indices_all_hops_correct & indices_x_GFx_correct
    else:
        indices = indices_all_hops_correct - indices_x_GFx_correct

    return [queries[i] for i in indices]


@dataclass(kw_only=True)
class EvaluateTaskExperiment(GenerateDataExperiment):
    model_name: ModelT
    use_nnsight: bool = False
    num_queries: int | None = 250

    @property
    def hops(self) -> list[HopT]:
        return Task(task_name=self.task_name).correct_hops + [("x", "GFx")]

    @property
    def step_dict(self) -> dict[str, Step]:
        """Evaluation pipeline extending data generation pipeline."""
        steps = super().step_dict

        # Step 3: Sample queries for evaluation (list[InContextQuery])
        steps["sampled_queries"] = sample_queries(
            in_context_queries=steps["in_context_queries"],
            num_examples=self.num_queries,
            seed=self.seed,
        )

        # Step 4: Evaluate all hop types (EvaluationResults for each hop)
        for query_type, pred_type in self.hops:
            steps[f"results_{query_type}_{pred_type}"] = evaluate(
                task_name=self.task_name,
                in_context_queries=steps["sampled_queries"],
                model_name=self.model_name,
                use_nnsight=self.use_nnsight,
                query_type=query_type,
                pred_type=pred_type,
            )

        hop_steps = {
            f"results_{query_type}_{pred_type}": steps[f"results_{query_type}_{pred_type}"]
            for query_type, pred_type in self.hops
            if (query_type, pred_type) != ("x", "GFx")
        }

        # Step 5: Compute compositionality gap statistics (PercentHopsCorrect)
        steps["percent_hops_correct"] = percent_hops_correct(
            composition_results=steps["results_x_GFx"],
            **hop_steps,
        )

        # Step 6-7: Filter queries by correctness for downstream analysis (list[InContextQuery])
        steps["correct_queries"] = filter_queries_by_correctness(
            queries=steps["sampled_queries"],
            return_correct_compositions=True,
            composition_results=steps["results_x_GFx"],
            **hop_steps,
        )

        steps["incorrect_queries"] = filter_queries_by_correctness(
            queries=steps["sampled_queries"],
            return_correct_compositions=False,
            composition_results=steps["results_x_GFx"],
            **hop_steps,
        )

        return steps

    def results(self) -> dict:
        return {
            **{
                f"{query_type}_{pred_type}": self.step_result(f"results_{query_type}_{pred_type}")["accuracy"]
                for query_type, pred_type in self.hops
            },
            **self.step_result("percent_hops_correct"),
            "num_queries": len(self.step_result("sampled_queries")),
            "num_correct_primitives": len(self.step_result("correct_queries"))
            + len(self.step_result("incorrect_queries")),
            "num_all_correct": len(self.step_result("correct_queries")),
        }


@dataclass
class EvaluateTaskSweep(Sweep[EvaluateTaskExperiment]):
    models: list[ModelT]
    tasks: list[TaskT]
    icl_examples: int = 10
    seed: int = 0

    use_nnsight: bool = False
    num_queries: int | None = 250

    @property
    def experiments(self) -> Sequence[EvaluateTaskExperiment]:
        return [
            EvaluateTaskExperiment(
                model_name=model_name,
                task_name=task_name,
                icl_examples=self.icl_examples,
                seed=self.seed,
                use_nnsight=self.use_nnsight,
                num_queries=self.num_queries,
            )
            for task_name in self.tasks
            for model_name in self.models
        ]


if __name__ == "__main__":
    EvaluateTaskExperiment.cli()
