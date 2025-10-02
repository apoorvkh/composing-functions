import os
from pathlib import Path
from typing import Literal

import altair as alt
import polars as pl
import tyro
from plotting import plotting_utils as pu

OUTPUT_DIR = Path("artifacts")

os.makedirs(OUTPUT_DIR / "patching", exist_ok=True)


def save_plot(config: Literal["comp", "direct"]):
    match config:
        case "comp":
            from configs.llama_3_3b.patching_across_tasks_comp import sweep
        case "direct":
            from configs.llama_3_3b.patching_across_tasks_direct import sweep

    results = sweep.results().with_columns(
        task_src=pl.col("task_src").replace(pu.tasks_shorthand),
        task_dst=pl.col("task_dst").replace(pu.tasks_shorthand),
    )

    _token_cols = ["GF_dst_x_dst", "GF_src_x_dst", "GF_dst_x_src", "GF_src_x_src"]

    data = results.pivot(
        index=[c for c in results.columns if c not in {"patch", *_token_cols}],
        on="patch",
    ).with_columns(
        tasks=pl.col("task_src") + " → " + pl.col("task_dst"),
        GF_dst_x_dst=(pl.col("GF_dst_x_dst_true") - pl.col("GF_dst_x_dst_false")),
        GF_src_x_dst=(pl.col("GF_src_x_dst_true") - pl.col("GF_src_x_dst_false")),
        GF_dst_x_src=(pl.col("GF_dst_x_src_true") - pl.col("GF_dst_x_src_false")),
        GF_src_x_src=(pl.col("GF_src_x_src_true") - pl.col("GF_src_x_src_false")),
    )

    data = data.melt(
        id_vars=[c for c in data.columns if c not in _token_cols],
        value_vars=_token_cols,
        variable_name="token",
        value_name="delta",
    ).with_columns(
        token=pu.map_column(
            col="token",
            map={
                "GF_dst_x_dst": "g(f(x))",
                "GF_dst_x_src": "g(f(x'))",
                "GF_src_x_dst": "g'(f(x))",
                "GF_src_x_src": "g'(f(x'))",
            },
        )
    )

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("token", axis=None),
            y=alt.Y("delta", title="Δ Rec. Rank", scale=alt.Scale(domain=[-1, 1])),
            color=alt.Color("token", legend=alt.Legend(title="", orient="bottom")).scale(range=pu.color_range),
        )
        .properties(width=300, height=300)
        .facet(facet=alt.Facet("tasks", header=alt.Header(title="", labelFontSize=24)), columns=3)
        .configure_axis(titleFontSize=28, labelFontSize=24, tickCount=6)
        .configure_legend(labelFontSize=28)
    )

    chart.save(f"artifacts/patching/patching_across_tasks_{config}.pdf")


if __name__ == "__main__":
    tyro.cli(save_plot)
