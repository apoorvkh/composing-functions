import os
from pathlib import Path

import altair as alt
from configs.llama_3_3b.lens import sweep
from plotting import plotting_utils as pu

OUTPUT_DIR = Path("artifacts")


def build_chart_per_task(experiment):
    _df = experiment.step_result("aggregate_processing_signature")
    _df = _df.with_columns(node_type=pu.map_column(col="node_type", map=pu.node_labels))

    node_labels, node_colors, node_dash = pu.node_properties(task_name=experiment.task_name)

    _title = pu.tasks_shorthand[experiment.task_name]

    _chart = (
        alt.Chart(_df, title=_title)
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
        .configure_title(fontSize=35)
        .configure_line(strokeWidth=8)
        .configure_axis(labelFontSize=28, titleFontSize=34)
        .configure_legend(labelFontSize=30, orient="top", symbolOffset=-50)
    )

    return _chart


os.makedirs(OUTPUT_DIR / "lens", exist_ok=True)
for e in sweep.experiments:
    if len(e.step_result("processing_signatures")) >= 10:
        build_chart_per_task(e).save(
            OUTPUT_DIR / "lens" / f"{e.task_name}_{'in' if not e.correctness else ''}correct.pdf"
        )


os.makedirs(OUTPUT_DIR / "lens_excluded", exist_ok=True)
for e in sweep.experiments:
    if 0 < len(e.step_result("processing_signatures")) < 10:
        build_chart_per_task(e).save(
            OUTPUT_DIR / "lens_excluded" / f"{e.task_name}_{'in' if not e.correctness else ''}correct.pdf"
        )
