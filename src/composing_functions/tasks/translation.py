from __future__ import annotations

from typing import Literal

import nltk
import requests
from easynmt import EasyNMT
from tqdm import tqdm

from . import CompositionalTask


def get_translation_model() -> EasyNMT:
    nltk.download("punkt_tab")
    return EasyNMT("opus-mt")


def translate(model: EasyNMT, word: str, language: Literal["es", "de", "fr"]) -> str:
    output: str = model.translate(word, source_lang="en", target_lang=language)  # type: ignore
    return output.split(" ")[0].lower()


def get_antonyms() -> dict[str, str]:
    """Antonym data from 'Function Vectors in Large Language Models' (Todd, ICLR 2024)"""
    antonyms = requests.get(
        "https://raw.githubusercontent.com/ericwtodd/function_vectors/5b5b88caf241f5a0dc576cbfdb4f1cfede8a096e/dataset_files/abstractive/antonym.json"
    ).json()
    return {i["input"]: i["output"] for i in antonyms}


def antonym_translation(language: Literal["es", "de", "fr"]) -> list[CompositionalTask]:
    """Functions: A [antonym], B [translation]"""
    antonyms = get_antonyms()
    antonyms = list(antonyms.items())

    model = get_translation_model()

    examples = [
        CompositionalTask(
            x=word,
            Fx=antonym,
            Gx=translate(model, word, language),
            GFx=translate(model, antonym, language),
            FGx=None,
        )
        for word, antonym in tqdm(antonyms)
    ]
    return examples
