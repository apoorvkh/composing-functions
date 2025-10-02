import os
from pathlib import Path

import altair as alt
import plotting.plotting_utils as pu
import polars as pl
from configs.llama_3_3b.task_lens_corr_token_identity import sweep

OUTPUT_DIR = Path("artifacts")

os.makedirs(OUTPUT_DIR / "token_identity", exist_ok=True)
os.makedirs(OUTPUT_DIR / "token_identity" / "corr", exist_ok=True)

results = sweep.results()
results = results.with_columns(task_name=pu.map_column(col="task_name", map=pu.tasks_shorthand))
results = results.drop_nulls("intermediate_metric_aggregated").filter(pl.col("count") >= 10)

task_order = list(results.sort(by="intermediate_metric_aggregated", descending=True)["task_name"])

chart = (
    alt.Chart(results)
    .transform_regression("intermediate_metric_aggregated", "task_linearity", method="linear", extent=[0, 1])
    .mark_line(color="#EFB118")
    .encode(x="task_linearity:Q", y="intermediate_metric_aggregated:Q")
)

chart += (
    alt.Chart(results)
    .mark_point(filled=True, opacity=1)
    .encode(
        x=alt.X(
            "task_linearity:Q",
            scale=alt.Scale(domain=[0.55, 0.95]),
            axis=alt.Axis(title="Embedding Space Task Linearity"),
        ),
        y=alt.Y(
            "intermediate_metric_aggregated:Q",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(title="Presence of Intermediate Variables"),
        ),
        color=alt.Color(
            "task_name",
            scale=alt.Scale(range=(pu.observable10 + pu.dark2)),
            sort=task_order,
        ),
    )
)

chart = (
    chart.properties(width=900, height=660, autosize="fit", padding=0)
    .configure_point(size=400)
    .configure_line(strokeDash=(16, 10), strokeWidth=8)
    .configure_axis(labelFontSize=32, titleFontSize=36)
    .configure_axisX(tickCount=4)
    .configure_axisY(format=".1f")
    .configure_legend(labelFontSize=28, title=None, labelLimit=1000)
)

chart.save(OUTPUT_DIR / "token_identity" / "task_lens_correlation.pdf")

print(
    "{:.2f}".format(results.select(pl.corr("intermediate_metric_aggregated", "task_linearity")).item() ** 2),
    end="",
    file=open(OUTPUT_DIR / "token_identity" / "corr" / "lens_task.tex", "w"),
)

print(
    "{:.2f}".format(results.select(pl.corr("accuracy", "task_linearity")).item() ** 2),
    end="",
    file=open(OUTPUT_DIR / "token_identity" / "corr" / "acc_task.tex", "w"),
)

print(
    "{:.2f}".format(results.select(pl.corr("accuracy", "intermediate_metric_aggregated")).item() ** 2),
    end="",
    file=open(OUTPUT_DIR / "token_identity" / "corr" / "acc_lens.tex", "w"),
)
