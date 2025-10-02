from pathlib import Path

import polars as pl
from configs.generate_data import sweep
from plotting import plotting_utils as pu

OUTPUT_DIR = Path("artifacts")

dataset_table = (
    sweep.results()
    .drop("icl_examples", "seed")
    .with_columns(
        f=pl.col("task_name").replace(pu.f_titles),
        g=pl.col("task_name").replace(pu.g_titles),
        example=(pl.col("x") + " → " + pl.col("Fx") + " → " + pl.col("GFx")),
        FGx=pl.when(pl.col("FGx") == pl.col("GFx")).then(None).otherwise(pl.col("FGx")),
    )
)


def generate_table(df: pl.DataFrame) -> str:
    latex_table = df.fill_null("---").style.as_latex()
    latex_table = latex_table.replace(r"`\$`", "$").replace(r"`\\`", "\\")
    latex_table = latex_table.replace(r"\addlinespace[2.5pt]", "")
    latex_table = latex_table[latex_table.find(r"\midrule") + 9 : latex_table.rfind(r"\bottomrule") - 1]
    return latex_table


with open(OUTPUT_DIR / "tasks.tex", "w") as f:
    _df = dataset_table.select(["f", "g", "dataset_size", "example"])
    f.write(generate_table(_df))

with open(OUTPUT_DIR / "tasks_appendix.tex", "w") as f:
    _df = dataset_table.select(["f", "g", "x", "Gx", "FGx"]).filter(~(pl.col("Gx").is_null() & pl.col("FGx").is_null()))
    f.write(generate_table(_df))
