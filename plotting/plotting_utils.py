from typing import Sequence

import polars as pl

from composing_functions.models import ModelT
from composing_functions.tasks import HopT, NodeT, Task, TaskT

node_labels: dict[NodeT, str] = {
    "x": "x",
    "Fx": "f(x)",
    "Gx": "g(x)",
    "GFx": "g(f(x))",
    "FGx": "f(g(x))",
}


def node_properties(task_name: TaskT) -> tuple[list[str], list[str], list[tuple[int, int]]]:
    task = Task(task_name=task_name)

    def get_color(n: NodeT) -> str:
        match n:
            case "x":
                return "#4269D0"  # blue
            case "GFx":
                return "#3CA951"  # green
            case "FGx":
                return "#FF725C"  # red
            case "Fx":
                return "#EFB118"  # orange
            case "Gx":
                if "Gx" in task.correct_intermediate_nodes:
                    return "#A463F2"  # purple
                return "#FF725C"  # red

    def get_stroke_dash(n: NodeT) -> tuple[int, int]:
        match n:
            case "x" | "GFx" | "FGx":
                return (1, 0)  # solid line
            case "Fx" | "Gx":
                return (8, 8)  # dashed line

    nodes = task.nodes
    labels = [node_labels[node] for node in nodes]
    colors = [get_color(node) for node in nodes]
    stroke_dash = [get_stroke_dash(node) for node in nodes]

    return labels, colors, stroke_dash


hop_labels: dict[HopT, str] = {
    ("x", "Fx"): "x → f(x)",
    ("Fx", "GFx"): "f(x) → g(f(x))",
    ("x", "GFx"): "x → g(f(x))",
    ("x", "Gx"): "x → g(x)",
    ("Gx", "FGx"): "g(x) → f(g(x))",
    ("Gx", "GFx"): "g(x) → g(f(x))",
}

hops_labels = {
    "at_least_one_primitive": "≥1 Hop",
    "all_primitives": "All Hops",
    "all_primitives_and_composition": "All Hops + Composition",
}


model_labels: dict[ModelT, str] = {
    "llama-3-1b": "Llama 3 (1B)",
    "llama-3-3b": "Llama 3 (3B)",
    "llama-3-8b": "Llama 3 (8B)",
    "llama-3-70b": "Llama 3 (70B)",
    "llama-3-405b": "Llama 3 (405B-I)",
    "olmo-2-1b": "OLMo 2 (1B)",
    "olmo-2-7b": "OLMo 2 (7B)",
    "olmo-2-13b": "OLMo 2 (13B)",
    "olmo-2-32b": "OLMo 2 (32B)",
    "deepseek-v3": "DeepSeek-V3",
    "deepseek-r1": "DeepSeek-R1",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gpt-4o": "GPT-4o",
    "o4-mini": "o4-mini",
}

model_families: dict[ModelT, str] = {
    "llama-3-1b": "Llama 3",
    "llama-3-3b": "Llama 3",
    "llama-3-8b": "Llama 3",
    "llama-3-70b": "Llama 3",
    "llama-3-405b": "Llama 3",
    "olmo-2-1b": "OLMo 2",
    "olmo-2-7b": "OLMo 2",
    "olmo-2-13b": "OLMo 2",
    "olmo-2-32b": "OLMo 2",
    "deepseek-v3": "DeepSeek",
    "deepseek-r1": "DeepSeek",
    "gpt-4o": "GPT",
    "o4-mini": "GPT",
}

model_parameters: dict[ModelT, str] = {
    "olmo-2-1b": "1",
    "olmo-2-7b": "7",
    "olmo-2-13b": "13",
    "olmo-2-32b": "32",
    "llama-3-1b": "1",
    "llama-3-3b": "3",
    "llama-3-8b": "8",
    "llama-3-70b": "70",
    "llama-3-405b": "405",
}

model_layers: dict[ModelT, str] = {
    "olmo-2-1b": "16",
    "olmo-2-7b": "32",
    "olmo-2-13b": "40",
    "olmo-2-32b": "64",
    "llama-3-1b": "16",
    "llama-3-3b": "28",
    "llama-3-8b": "32",
    "llama-3-70b": "80",
    "llama-3-405b": "126",
}

tasks_shorthand: dict[TaskT, str] = {
    "antonym-spanish": "antonym-spanish",
    "antonym-german": "antonym-german",
    "antonym-french": "antonym-french",
    "book-author-birthyear": "book-author-birthyear",
    "song-artist-birthyear": "song-artist-birthyear",
    "landmark-country-capital": "landmark-country-capital",
    "park-country-capital": "park-country-capital",
    "movie-director-birthyear": "movie-director-birthyear",
    "person-university-year": "person-university-year",
    "person-university-founder": "person-university-founder",
    "product-company-ceo": "product-company-ceo",
    "product-company-hq": "product-company-hq",
    "plus-ten-times-two": "plus-10-times-2",
    "plus-hundred-times-two": "plus-100-times-2",
    "mod-twenty-times-two": "mod-20-times-2",
    "word-int-times-two": "word-int-times-2",
    "word-substring-reverse": "word-truncate-reverse",
    "rgb-rot120-name": "rgb-rotate-name",
}

f_latex_titles: dict[TaskT, str] = {
    "antonym-spanish": "Word → Antonym",
    "antonym-german": "Word → Antonym",
    "antonym-french": "Word → Antonym",
    "book-author-birthyear": "Book → Author",
    "song-artist-birthyear": "Song → Artist",
    "landmark-country-capital": "Landmark → Country",
    "park-country-capital": "Park → Country",
    "movie-director-birthyear": "Movie → Director",
    "person-university-year": "Person → University",
    "person-university-founder": "Person → University",
    "product-company-ceo": "Product → Company",
    "product-company-hq": "Product → Company",
    "plus-ten-times-two": "`$`x + 10`$`",
    "plus-hundred-times-two": "`$`x + 100`$`",
    "mod-twenty-times-two": r"`$`x `\`mod 20`$`",
    "word-int-times-two": "Word → Numeric",
    "word-substring-reverse": "Word[:-1]",
    "rgb-rot120-name": "Rotate(RGB, 120°)",
}

g_latex_titles: dict[TaskT, str] = {
    "antonym-spanish": "English → Spanish",
    "antonym-german": "English → German",
    "antonym-french": "English → French",
    "book-author-birthyear": "Author → Birth Year",
    "song-artist-birthyear": "Artist → Birth Year",
    "landmark-country-capital": "Country → Capital",
    "park-country-capital": "Country → Capital",
    "movie-director-birthyear": "Director → Birth Year",
    "person-university-year": "University → Year",
    "person-university-founder": "University → Founder",
    "product-company-ceo": "Company → CEO",
    "product-company-hq": "Company → HQ",
    "plus-ten-times-two": "`$`2x`$`",
    "plus-hundred-times-two": "`$`2x`$`",
    "mod-twenty-times-two": "`$`2x`$`",
    "word-int-times-two": "`$`2x`$`",
    "word-substring-reverse": "Word[::-1]",
    "rgb-rot120-name": "RGB → Name",
}

f_titles = {k: v.replace("`$`", "").replace(r"`\`", "") for k, v in f_latex_titles.items()}
g_titles = {k: v.replace("`$`", "").replace(r"`\`", "") for k, v in g_latex_titles.items()}


def map_column(col: str, map: dict) -> pl.Expr:
    return pl.col(col).replace(map).cast(pl.Enum(map.values()))


def merge_columns(df: pl.DataFrame, cols: Sequence[str], key: str, value: str) -> pl.DataFrame:
    return df.unpivot(
        index=[c for c in df.columns if c not in cols],
        on=[c for c in df.columns if c in cols],
        variable_name=key,
        value_name=value,
    )


observable10 = [
    "#4269D0",
    "#EFB118",
    "#FF725C",
    "#6CC5B0",
    "#3CA951",
    "#FF8AB7",
    "#A463F2",
    "#97BBF5",
    "#9C6B4E",
    "#9498A0",
]
dark2 = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"]
color_range = observable10 + dark2
