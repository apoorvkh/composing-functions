from typing import Literal

import tyro

from composing_functions.experiments.residual_stream import ResidualStreamSweep

sweep = ResidualStreamSweep(
    tasks=[
        # antonym translation
        "antonym-spanish",
        "antonym-german",
        "antonym-french",
        # wikidata relations
        "book-author-birthyear",
        "song-artist-birthyear",
        "landmark-country-capital",
        "park-country-capital",
        "movie-director-birthyear",
        "person-university-year",
        "person-university-founder",
        "product-company-ceo",
        "product-company-hq",
        # arithmetic
        "plus-ten-times-two",
        "plus-hundred-times-two",
        "mod-twenty-times-two",
        "word-int-times-two",
        "word-substring-reverse",
        "rgb-rot120-name",
    ],
    models=[
        "llama-3-3b",
    ],
)


def main(cmd: Literal["run", "count", "print-incomplete", "print-results"] = "run"):
    sweep.run(experiment_sweep=sweep, cmd=cmd)


if __name__ == "__main__":
    tyro.cli(main)
