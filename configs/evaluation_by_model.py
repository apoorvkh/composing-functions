from typing import Literal

import tyro

from composing_functions.experiments.evaluate_task import EvaluateTaskSweep

sweep = EvaluateTaskSweep(
    models=[
        "llama-3-1b",
        "llama-3-3b",
        "llama-3-8b",
        "llama-3-70b",
        "llama-3-405b",
        #
        "olmo-2-1b",
        "olmo-2-7b",
        "olmo-2-13b",
        "olmo-2-32b",
        #
        "deepseek-v3",
        "deepseek-r1",  # reasoning
        #
        "gpt-4o",
        "o4-mini",  # reasoning
    ],
    tasks=[
        "antonym-spanish",
        "plus-hundred-times-two",
        "park-country-capital",
        "book-author-birthyear",
    ],
    num_queries=100,
)


def main(cmd: Literal["run", "count", "print-incomplete", "print-results"] = "run"):
    sweep.run(experiment_sweep=sweep, cmd=cmd)


if __name__ == "__main__":
    tyro.cli(main)
