from pathlib import Path

import altair as alt
import polars as pl
from configs.llama_3_3b.evaluation import sweep
from plotting import plotting_utils as pu

OUTPUT_DIR = Path("artifacts")

results = sweep.results()
results = results.with_columns(
    all_primitives=-pl.col("all_primitives"),
    all_primitives_rel=(pl.col("all_primitives") + 1e-10) / (pl.col("all_primitives") + 1e-10),
    all_primitives_and_composition_rel=pl.col("all_primitives_and_composition") / (pl.col("all_primitives") + 1e-10),
)

print(
    f"{results.select(pl.corr('all_primitives', 'all_primitives_and_composition_rel')).item() ** 2:.2f}",
    end="",
    file=open(OUTPUT_DIR / "corr" / "compositionality_gap.tex", "w"),
)

results = pu.merge_columns(
    results,
    cols=["all_primitives", "all_primitives_rel", "all_primitives_and_composition_rel"],
    key="Hops",
    value="Proportion Correct",
).with_columns(
    Tasks=pu.map_column(col="task_name", map=pu.tasks_shorthand),
    Hops=pu.map_column(
        col="Hops",
        map={
            "all_primitives_rel": "All Hops (Relative)",
            "all_primitives_and_composition_rel": "All Hops + Composition (Relative)",
            "all_primitives": "All Hops (Absolute)",
        },
    ),
)

sorted_tasks = list(
    results.filter(pl.col("Hops") == "All Hops + Composition (Relative)")
    .select("Tasks", "Proportion Correct")
    .sort("Proportion Correct", descending=True)
    .get_column("Tasks")
)

_chart = (
    alt.Chart(results)
    .mark_bar()
    .encode(
        x=alt.X("Tasks", sort=sorted_tasks),
        y=alt.Y(
            "Proportion Correct",
            stack=None,
            axis=alt.Axis(labelExpr="datum.value < 0 ? -datum.value : datum.value"),
        ),
        color=alt.Color("Hops").scale(range=["#4269D0", "#EFB118", "#FF725C"]),
    )
    .properties(width=1650, height=(1650 // (16 / 9)), autosize="fit", padding=0)
    .configure_axis(titleFontSize=48, labelLimit=1000)
    .configure_axisX(labelFontSize=34, titlePadding=110, labelAngle=-45)
    .configure_axisY(labelFontSize=36, titlePadding=10)
    .configure_legend(labelFontSize=36, titleFontSize=48, orient="top", labelLimit=1000)
)

_chart.save(OUTPUT_DIR / "compositionality_gap.pdf")
