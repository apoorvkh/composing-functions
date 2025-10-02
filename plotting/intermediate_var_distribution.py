from pathlib import Path

import altair as alt
import plotting.plotting_utils as pu
import polars as pl
from configs.llama_3_3b.task_lens_corr import sweep

OUTPUT_DIR = Path("artifacts")

results = sweep.results()
results = results.with_columns(task_name=pu.map_column(col="task_name", map=pu.tasks_shorthand))
results = results.drop_nulls("intermediate_metric_aggregated").filter(pl.col("count") >= 10)

task_order = list(results.sort("intermediate_metric_aggregated", descending=True)["task_name"])

bin_size = 0.1

hist = (
    results.explode(columns=["intermediate_metric_per_query"])
    .with_columns(
        intermediate_metric=((pl.col("intermediate_metric_per_query") / bin_size).round(0) * bin_size).round(1)
    )
    .group_by(["task_name", "intermediate_metric"])
    .len()
    .rename({"len": "count"})
    .with_columns(normalized_count=(pl.col("count") / pl.col("count").sum().over("task_name")))
)

_chart = (
    alt.Chart(hist)
    .mark_bar()
    .encode(
        x=alt.X("normalized_count", title=""),
        y=alt.Y(
            "intermediate_metric:N",
            title=None,
            scale=alt.Scale(reverse=True),
        ),
        color=alt.Color(
            "task_name",
            legend=None,
            sort=task_order,
            scale=alt.Scale(range=pu.color_range),
        ),
    )
    .properties(width=200, height=180)
    .facet(
        facet=alt.Facet(
            "task_name",
            header=alt.Header(title="Presence of Intermediate Variables", titleOrient="left"),
            sort=task_order,
        ),
        columns=4,
        spacing=10,
    )
    .properties(title=alt.Title("Proportion of Examples", orient="bottom", anchor="middle"))
    .configure(padding=0)
    .configure_title(fontSize=40)
    .configure_axis(labelFontSize=30)
    .configure_header(labelFontSize=30, titleFontSize=40)
)

_chart.save(OUTPUT_DIR / "intermediate_distribution.pdf")


bimodal_density = (
    hist.group_by("intermediate_metric")
    .agg(pl.col("normalized_count").mean())
    .with_columns(pl.col("normalized_count") / pl.col("normalized_count").sum())
    .filter(
        pl.col("intermediate_metric").is_between(0, 0.1, closed="left")
        | pl.col("intermediate_metric").is_between(0.5, 1, closed="both")
    )["normalized_count"]
    .sum()
)

print(
    f"{bimodal_density:.0%}".replace("%", "\\%"),
    end="",
    file=open(OUTPUT_DIR / "intermediate_var_bimodal_density.tex", "w"),
)
