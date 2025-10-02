from pathlib import Path

import altair as alt
from configs.llama_3_3b.lens import sweep
from plotting import plotting_utils as pu

OUTPUT_DIR = Path("artifacts")


def build_plot(sweep, correctness):
    aggregate_df = sweep.overall_processing_signature(correctness=correctness)
    aggregate_df = aggregate_df.with_columns(node_type=pu.map_column(col="node_type", map=pu.node_labels))

    node_labels, node_colors, node_dash = pu.node_properties(task_name="plus-ten-times-two")

    return (
        alt.Chart(
            aggregate_df,
            title=f"{'Correct' if correctness else 'Incorrect'} Examples",
        )
        .mark_line()
        .encode(
            x=alt.X("layer", axis=alt.Axis(title="Layer")),
            y=alt.Y("reciprocal_rank", axis=alt.Axis(title="Reciprocal Rank")),
            color=alt.Color(
                "node_type",
                sort=node_labels,
                title="",
                scale=alt.Scale(domain=node_labels, range=node_colors),
            ),
            strokeDash=alt.StrokeDash(
                "node_type",
                sort=node_labels,
                title="",
                scale=alt.Scale(domain=node_labels, range=node_dash),
            ),
        )
        .properties(width=500, height=500, autosize="fit", padding=0)
        .configure_title(fontSize=36)
        .configure_line(strokeWidth=8)
        .configure_axis(labelFontSize=28, titleFontSize=34)
        .configure_legend(labelFontSize=29, orient="top", symbolOffset=-50)
    )


build_plot(sweep=sweep, correctness=True).save(OUTPUT_DIR / "lens_correct.pdf")
build_plot(sweep=sweep, correctness=False).save(OUTPUT_DIR / "lens_incorrect.pdf")
