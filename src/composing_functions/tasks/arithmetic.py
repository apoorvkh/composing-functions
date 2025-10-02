import colorsys
import random

import requests
import webcolors
from num2words import num2words

from . import CompositionalTask


def plus_ten_times_two(num_examples: int = 1000) -> list[CompositionalTask]:
    """Functions: A [+10], B [x2]"""
    examples = []
    for x in range(num_examples):
        examples.append(
            CompositionalTask(
                x=str(x),
                Fx=str(x + 10),
                Gx=str(x * 2),
                GFx=str((x + 10) * 2),
                FGx=str((x * 2) + 10),
            )
        )
    return examples


def plus_hundred_times_two(num_examples: int = 1000) -> list[CompositionalTask]:
    """Functions: A [+100], B [x2]"""
    examples = []
    for x in range(num_examples):
        examples.append(
            CompositionalTask(
                x=str(x),
                Fx=str(x + 100),
                Gx=str(x * 2),
                GFx=str((x + 100) * 2),
                FGx=str((x * 2) + 100),
            )
        )
    return examples


def mod_twenty_times_two(num_examples: int = 1000) -> list[CompositionalTask]:
    """Functions: A [%20], B [x2]"""
    return [
        CompositionalTask(
            x=str(x),
            Fx=str(x % 20),
            Gx=str(x * 2),
            GFx=str((x % 20) * 2),
            FGx=None,
        )
        for x in range(num_examples)
    ]


def word_int_times_two(num_examples: int = 1000) -> list[CompositionalTask]:
    """Functions: A [word-to-int], B [x2]"""
    return [
        CompositionalTask(
            x=num2words(x),
            Fx=str(x),
            Gx=num2words(x * 2),
            GFx=str(x * 2),
            FGx=None,
        )
        for x in range(num_examples)
    ]


def word_list() -> list[str]:
    antonyms = requests.get(
        "https://raw.githubusercontent.com/ericwtodd/function_vectors/5b5b88caf241f5a0dc576cbfdb4f1cfede8a096e/dataset_files/abstractive/antonym.json"
    ).json()
    return list({i["input"] for i in antonyms} | {i["output"] for i in antonyms})


def word_substr_reverse() -> list[CompositionalTask]:
    """Functions: A [remove last letter], B [reverse]"""
    return [
        CompositionalTask(
            x=x,
            Fx=x[:-1],
            Gx=x[::-1],
            GFx=x[:-1][::-1],
            FGx=x[::-1][:-1],
        )
        for x in word_list()
    ]


def rotate_hue(rgb: tuple[int, int, int], degrees: float) -> tuple[int, int, int]:
    r, g, b = [c / 255.0 for c in rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + degrees / 360.0) % 1.0
    r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
    return (int(r2 * 255), int(g2 * 255), int(b2 * 255))


def rgb_to_name(rgb: tuple[int, int, int]) -> str:
    def _dist(name):
        r, g, b = webcolors.name_to_rgb(name)
        return (r - rgb[0]) ** 2 + (g - rgb[1]) ** 2 + (b - rgb[2]) ** 2

    return min(webcolors.names(), key=_dist)


def rgb_rotation_name(num_examples: int = 1000, degrees: int = 120) -> list[CompositionalTask]:
    """Functions: A [rgb->rotation], B [rgb->name]"""
    examples = []
    for _ in range(num_examples):
        rgb = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
        hex = webcolors.rgb_to_hex(rgb).lstrip("#")

        rgb_rotated = rotate_hue(rgb, degrees)
        hex_rotated = webcolors.rgb_to_hex(rgb_rotated).lstrip("#")

        name = rgb_to_name(rgb)
        name_rotated = rgb_to_name(rgb_rotated)

        examples.append(
            CompositionalTask(
                x=hex,
                Fx=hex_rotated,
                Gx=name,
                GFx=name_rotated,
                FGx=None,
            )
        )

    return examples
