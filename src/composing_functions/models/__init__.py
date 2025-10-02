"""Model implementations and interface for language model evaluation.

This module provides a unified interface for working with various language models,
including open-source transformer models and API-based models.
"""

from abc import ABC, abstractmethod
from typing import Literal, cast

ModelT = Literal[
    # open source models
    "llama-3-1b",
    "llama-3-3b",
    "llama-3-8b",
    "llama-3-70b",
    "llama-3-405b",
    #
    "olmo-2-1b",
    "olmo-2-7b",
    "olmo-2-13b",
    "olmo-2-32b",
    #
    "deepseek-v3",
    "deepseek-r1",  # reasoning
    #
    "gpt-4o",
    "o4-mini",  # reasoning
]


class TextGenerationModel(ABC):
    """Abstract base class for all text generation models."""

    @abstractmethod
    def generate_continuation(self, text: str, max_new_tokens: int | None = None, stop_seq: str | None = None) -> str:
        """Generate continuation text from the given prompt.

        Args:
            text: Input prompt text
            max_new_tokens: Maximum number of new tokens to generate
            stop_seq: Stop generation when this sequence is encountered

        Returns:
            Generated continuation text
        """
        raise NotImplementedError


def get_model(model_name: ModelT, use_nnsight: bool = False) -> TextGenerationModel:
    """Factory function to instantiate models.

    Args:
        model_name: Name of the model to load
        use_nnsight: Whether to use NNsight wrapper for interpretability

    Returns:
        Model instance
    """
    match model_name:
        case (
            "llama-3-1b"
            | "llama-3-3b"
            | "llama-3-8b"
            | "llama-3-70b"
            | "olmo-2-1b"
            | "olmo-2-7b"
            | "olmo-2-13b"
            | "olmo-2-32b"
        ):
            from .transformers import NNsightModel, TransformersModel, TransformersModelT

            model_name = cast(TransformersModelT, model_name)
            if use_nnsight:
                return NNsightModel(model_name=model_name)
            return TransformersModel(model_name=model_name)
        case "llama-3-405b" | "deepseek-v3" | "deepseek-r1" | "gpt-4o" | "o4-mini":
            from .api import APIModel, APIModelT

            model_name = cast(APIModelT, model_name)
            return APIModel(model_name=model_name)


__all__ = ["TextGenerationModel", "get_model", "ModelT"]
