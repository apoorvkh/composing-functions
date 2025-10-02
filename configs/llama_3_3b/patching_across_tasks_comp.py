from typing import Literal

import tyro

from composing_functions.experiments.patching import PatchingSweep

sweep = PatchingSweep(
    tasks_src_dst=[
        ("antonym-spanish", "antonym-german"),
        ("antonym-spanish", "antonym-french"),
        ("antonym-german", "antonym-spanish"),
        ("antonym-german", "antonym-french"),
        ("antonym-french", "antonym-spanish"),
        ("antonym-french", "antonym-german"),
        ("product-company-ceo", "product-company-hq"),
        ("product-company-hq", "product-company-ceo"),
    ],
    var_prob_range_src=(0.5, 1.0),
    var_prob_range_dst=(0.5, 1.0),
    layer_pos_dst=(18, 0.71),  # from LensSweep.median_intermediate_var_location()
    model="llama-3-3b",
)


def main(cmd: Literal["run", "count", "print-incomplete", "print-results"] = "run"):
    sweep.run(experiment_sweep=sweep, cmd=cmd)


if __name__ == "__main__":
    tyro.cli(main)
