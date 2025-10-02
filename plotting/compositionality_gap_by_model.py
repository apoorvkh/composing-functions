from pathlib import Path

import altair as alt
import polars as pl
from configs.evaluation_by_model import sweep
from plotting import plotting_utils as pu

OUTPUT_DIR = Path("artifacts")

results = sweep.results().with_columns(
    model_families=pl.col("model_name").replace(pu.model_families),
    Models=pu.map_column(col="model_name", map=pu.model_labels),
    relative_gap=((pl.col("all_primitives") - pl.col("all_primitives_and_composition")) / pl.col("all_primitives")),
    absolute_gap=(pl.col("all_primitives") - pl.col("all_primitives_and_composition")),
)

results = pu.merge_columns(
    results,
    cols=["at_least_one_primitive", "all_primitives", "all_primitives_and_composition"],
    key="Hops",
    value="Proportion Correct",
).with_columns(
    Hops=pu.map_column(col="Hops", map=pu.hops_labels),
)

_base = alt.Chart(results).encode(x=alt.X("Models"), detail="model_families")

chart = (
    _base.mark_line(point={"shape": "square"}).encode(
        y=alt.Y("mean(Proportion Correct)"),
        color=alt.Color("Hops").scale(range=pu.color_range),
    )
) + (
    _base.mark_errorband(extent="iqr").encode(
        y=alt.Y("Proportion Correct"),
        color=alt.Color("Hops").scale(range=pu.color_range),
    )
)

_c = pu.color_range[6]

gap = _base.mark_line(
    point={"shape": "triangle", "color": _c},
    strokeDash=[8, 8],
    color=_c,
).encode(
    y=alt.Y(
        "mean(relative_gap)",
        title="Compositionality Gap",
        scale=alt.Scale(domain=[0.0, 1.0]),
        axis=alt.Axis(titleColor=_c, labelColor=_c),
    )
)

chart = alt.layer(chart, gap).resolve_scale(y="independent")

chart = (
    chart.properties(width=1650, height=(1650 // (16 / 9)), autosize="fit", padding=0)
    .configure_line(strokeWidth=8)
    .configure_point(size=400)
    .configure_axis(labelFontSize=38, titleFontSize=48, labelLimit=1000)
    .configure_axisX(labelAngle=-45, titlePadding=20)
    .configure_axisY(format=".1f", titlePadding=10)
    .configure_legend(labelFontSize=36, titleFontSize=48, orient="top", labelLimit=1000)
)

chart.save(OUTPUT_DIR / "compositionality_gap_by_model.pdf")
