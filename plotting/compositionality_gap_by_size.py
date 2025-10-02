from pathlib import Path

import altair as alt
import polars as pl
from configs.evaluation_by_model import sweep
from plotting import plotting_utils as pu

OUTPUT_DIR = Path("artifacts")

results = (
    sweep.results()
    .with_columns(
        model_families=pl.col("model_name").replace(pu.model_families),
        Models=pu.map_column(col="model_name", map=pu.model_labels),
        relative_gap=((pl.col("all_primitives") - pl.col("all_primitives_and_composition")) / pl.col("all_primitives")),
        absolute_gap=(pl.col("all_primitives") - pl.col("all_primitives_and_composition")),
    )
    .filter(pl.col("model_families").is_in(["Llama 3", "OLMo 2"]))
    .with_columns(
        size=pl.col("model_name").replace(pu.model_parameters).cast(pl.Int32),
        layers=pl.col("model_name").replace(pu.model_layers).cast(pl.Int32),
    )
)

_base = (
    alt.Chart(results)
    .mark_line(point=alt.OverlayMarkDef(size=100), strokeWidth=3)
    .encode(
        y=alt.Y("mean(relative_gap)", title="Compositionality Gap", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color(
            "model_families",
            title="",
            scale=alt.Scale(range=pu.dark2),
        ),
    )
    .properties(width=int(1650 * 0.25), height=int(1650 * 0.25), autosize="fit")
    .configure_axis(labelFontSize=20, titleFontSize=24)
    .configure_legend(labelFontSize=20, orient="top-right", fillColor="white")
)

chart_params = _base.encode(x=alt.X("size", title="Parameters (billion)"))
chart_layers = _base.encode(x=alt.X("layers", title="Layers"))

chart_params.save(OUTPUT_DIR / "compositionality_gap_by_params.pdf")
chart_layers.save(OUTPUT_DIR / "compositionality_gap_by_layers.pdf")
