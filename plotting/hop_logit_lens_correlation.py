from pathlib import Path

import altair as alt
import plotting.plotting_utils as pu
import polars as pl
from configs.llama_3_3b.hop_lens_corr import sweep

OUTPUT_DIR = Path("artifacts")

hop_results = sweep.results()
hop_results = hop_results.with_columns(task_name=pu.map_column(col="task_name", map=pu.tasks_shorthand))
hop_results = hop_results.filter(pl.col("count") >= 10)
hop_results = hop_results.with_columns(intermediate_node=pu.map_column("intermediate_node", pu.node_labels))

task_order = list(hop_results.sort(by="intermediate_metric", descending=True)["task_name"])

hop_results = hop_results.rename(
    {
        "first_hop_linearity": "First Hop Linearity",
        "second_hop_linearity": "Second Hop Linearity",
        "min_hop_linearity": "Min Hop Linearity",
        "max_hop_linearity": "Max Hop Linearity",
    }
)

linearity_columns = [
    "First Hop Linearity",
    "Second Hop Linearity",
    "Min Hop Linearity",
    "Max Hop Linearity",
]

hop_results_long = hop_results.unpivot(
    index=[c for c in hop_results.columns if c not in linearity_columns],
    on=linearity_columns,
    variable_name="linearity_type",
    value_name="linearity",
)

base = (
    alt.Chart(hop_results_long)
    .mark_point(filled=True, opacity=1)
    .encode(
        x=alt.X(
            "linearity:Q",
            scale=alt.Scale(domain=[0.5, 1.0]),
            axis=alt.Axis(title="Emb. Space Linearity"),
        ),
        y=alt.Y(
            "intermediate_metric:Q",
            scale=alt.Scale(domain=[0, 1]),
            axis=alt.Axis(title="Intermediate Variable"),
        ),
        color=alt.Color(
            "task_name",
            scale=alt.Scale(range=pu.color_range),
            sort=task_order,
            legend=alt.Legend(title="Tasks", labelLimit=1000),
        ),
        shape=alt.Shape(
            "intermediate_node",
            legend=alt.Legend(title="Intermediate Node"),
        ),
    )
    .facet(
        facet=alt.Facet(
            "linearity_type:N",
            header=alt.Header(labelOrient="top", title=None, labelFontSize=36),
        ),
        columns=2,
    )
)

base = (
    base.configure_point(size=400)
    .configure_line(strokeDash=(16, 10), strokeWidth=8)
    .configure_axis(labelFontSize=32, titleFontSize=28)
    .configure_axisX(tickCount=4)
    .configure_axisY(format=".1f")
    .configure_legend(labelFontSize=28, titleFontSize=32, titleLimit=1000)
)

base.save(OUTPUT_DIR / "lens_hop_correlation.pdf")

###

print(
    "{:.2f}".format(hop_results.select(pl.corr("intermediate_metric", "First Hop Linearity")).item() ** 2),
    end="",
    file=open(OUTPUT_DIR / "corr" / "lens_first_hop.tex", "w"),
)

print(
    "{:.2f}".format(hop_results.select(pl.corr("intermediate_metric", "Second Hop Linearity")).item() ** 2),
    end="",
    file=open(OUTPUT_DIR / "corr" / "lens_second_hop.tex", "w"),
)

print(
    "{:.2f}".format(hop_results.select(pl.corr("intermediate_metric", "Min Hop Linearity")).item() ** 2),
    end="",
    file=open(OUTPUT_DIR / "corr" / "lens_min_hop.tex", "w"),
)

print(
    "{:.2f}".format(hop_results.select(pl.corr("intermediate_metric", "Max Hop Linearity")).item() ** 2),
    end="",
    file=open(OUTPUT_DIR / "corr" / "lens_max_hop.tex", "w"),
)
