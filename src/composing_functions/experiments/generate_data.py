import random
from dataclasses import dataclass

from ..prompting import InContextQuery
from ..tasks import CompositionalTask, Task, TaskT
from .utils import Experiment, Step, Sweep, step


@step(cacheable=True, version="001")
def generate_dataset(task_name: TaskT, seed: int = 0) -> list[CompositionalTask]:
    """Generate and shuffle the complete dataset for a task."""
    dataset = Task(task_name).build_dataset()
    random.seed(seed)
    random.shuffle(dataset)
    return dataset


@step(cacheable=True, version="001")
def generate_in_context_queries(
    dataset: list[CompositionalTask],
    icl_examples: int = 10,
    seed: int = 0,
) -> list[InContextQuery]:
    """
    Create in-context learning queries.
    Each dataset example is used as a query and a number of examples are sampled as context.
    Context examples do not overlap in any variables with the query.
    """
    random.seed(seed)
    in_context_queries = []
    for query in dataset:
        context = []
        while len(context) < icl_examples:
            example = random.choice(dataset)
            if example not in context and example != query and not CompositionalTask.overlap(example, query):
                context.append(example)
        in_context_queries.append(InContextQuery(context=context, query=query))
    return in_context_queries


@dataclass
class GenerateDataExperiment(Experiment):
    task_name: TaskT
    icl_examples: int = 10
    seed: int = 0

    @property
    def step_dict(self) -> dict[str, Step]:
        """Data generation pipeline."""
        steps = {}

        # Step 1: Create dataset for task (list[CompositionalTask])
        steps["dataset"] = generate_dataset(task_name=self.task_name, seed=self.seed)

        # Step 2: Create in-context queries (list[InContextQuery])
        steps["in_context_queries"] = generate_in_context_queries(
            dataset=steps["dataset"],
            icl_examples=self.icl_examples,
            seed=self.seed,
        )

        return steps

    def results(self) -> dict:
        """Return random sample from generated dataset."""
        dataset = self.step_result("dataset")
        random.seed(0)
        random_sample = random.sample(dataset, 1)[0]
        return {
            "x": random_sample.x,
            "Fx": random_sample.Fx,
            "Gx": random_sample.Gx,
            "GFx": random_sample.GFx,
            "FGx": random_sample.FGx,
            "dataset_size": len(dataset),
        }


@dataclass
class GenerateDataSweep(Sweep[GenerateDataExperiment]):
    tasks: list[TaskT]
    icl_examples: int = 10
    seed: int = 0

    @property
    def experiments(self) -> list[GenerateDataExperiment]:
        return [
            GenerateDataExperiment(
                task_name=task_name,
                icl_examples=self.icl_examples,
                seed=self.seed,
            )
            for task_name in self.tasks
        ]


if __name__ == "__main__":
    GenerateDataExperiment.cli()
