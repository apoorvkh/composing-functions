from __future__ import annotations

from typing import Literal

import nnsight
import torch
from nnsight import Envoy, LanguageModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from . import TextGenerationModel

TransformersModelT = Literal[
    "llama-3-1b",
    "llama-3-3b",
    "llama-3-8b",
    "llama-3-70b",
    "olmo-2-1b",
    "olmo-2-7b",
    "olmo-2-13b",
    "olmo-2-32b",
]


def model_path(model_name: TransformersModelT) -> str:
    match model_name:
        case "llama-3-1b":
            return "meta-llama/Llama-3.2-1B"
        case "llama-3-3b":
            return "meta-llama/Llama-3.2-3B"
        case "llama-3-8b":
            return "meta-llama/Llama-3.1-8B"
        case "llama-3-70b":
            return "meta-llama/Llama-3.1-70B"
        case "olmo-2-1b":
            return "allenai/OLMo-2-0425-1B"
        case "olmo-2-7b":
            return "allenai/OLMo-2-1124-7B"
        case "olmo-2-13b":
            return "allenai/OLMo-2-1124-13B"
        case "olmo-2-32b":
            return "allenai/OLMo-2-0325-32B"


class PreTrainedModelForCausalLM(PreTrainedModel, GenerationMixin):
    pass


class TransformersModel(TextGenerationModel):
    @staticmethod
    def get_tokenizer(model_name: TransformersModelT) -> PreTrainedTokenizerFast | PreTrainedTokenizer:
        return AutoTokenizer.from_pretrained(model_path(model_name), padding_side="left")

    def __init__(self, model_name: TransformersModelT):
        self.model_name: TransformersModelT = model_name
        self.model: PreTrainedModelForCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path(self.model_name), device_map="auto"
        )
        self.tokenizer = self.get_tokenizer(model_name)

    def generate_continuation(self, text: str, max_new_tokens: int | None = None, stop_seq: str | None = None) -> str:
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_tokens = self.model.generate(
            **model_inputs,  # type: ignore
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            stop_strings=stop_seq,
            pad_token_id=(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id),
        )
        generated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        continuation = generated_text[len(text) :]
        if stop_seq is not None:
            continuation = continuation.split(stop_seq, maxsplit=1)[0]
        return continuation


class NNsightModel(TextGenerationModel):
    @staticmethod
    def get_tokenizer(model_name: TransformersModelT) -> PreTrainedTokenizerFast | PreTrainedTokenizer:
        tokenizer = TransformersModel.get_tokenizer(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def __init__(self, model_name: TransformersModelT):
        self.model_name: TransformersModelT = model_name
        self.nnsight_model = LanguageModel(
            model_path(self.model_name),
            device_map="auto",
            dispatch=True,
            tokenizer=self.get_tokenizer(model_name),  # type: ignore
        )
        self.nnsight_model.eval()
        self.tokenizer = self.nnsight_model.tokenizer

    @property
    def device(self) -> torch.device:
        return self.nnsight_model.device  # type: ignore

    @property
    def layers(self) -> Envoy:
        match self.model_name:
            case "llama-3-1b" | "llama-3-3b" | "llama-3-8b" | "llama-3-70b" | "llama-3-405b":
                return self.nnsight_model.model.layers  # type: ignore
            case "olmo-2-1b" | "olmo-2-7b" | "olmo-2-13b" | "olmo-2-32b":
                return self.nnsight_model.model.layers  # type: ignore

    @property
    def norm(self) -> Envoy:
        match self.model_name:
            case "llama-3-1b" | "llama-3-3b" | "llama-3-8b" | "llama-3-70b" | "llama-3-405b":
                return self.nnsight_model.model.norm  # type: ignore
            case "olmo-2-1b" | "olmo-2-7b" | "olmo-2-13b" | "olmo-2-32b":
                return self.nnsight_model.model.norm  # type: ignore

    @property
    def lm_head(self) -> Envoy:
        match self.model_name:
            case "llama-3-1b" | "llama-3-3b" | "llama-3-8b" | "llama-3-70b" | "llama-3-405b":
                return self.nnsight_model.lm_head
            case "olmo-2-1b" | "olmo-2-7b" | "olmo-2-13b" | "olmo-2-32b":
                return self.nnsight_model.lm_head

    def generate_continuation(self, text: str, max_new_tokens: int | None = None, stop_seq: str | None = None) -> str:
        max_new_tokens = max_new_tokens or 20

        with self.nnsight_model.generate(text, max_new_tokens=max_new_tokens):
            new_tokens = nnsight.list().save()  # type: ignore
            for _ in range(max_new_tokens):
                new_tokens.append(self.lm_head.output.argmax(dim=-1)[0][-1])
                self.lm_head.next()

        new_tokens = [self.tokenizer.decode(_t) for _t in new_tokens]

        new_tokens = "".join(new_tokens)
        if stop_seq is not None and stop_seq in new_tokens:
            new_tokens = new_tokens[: new_tokens.index(stop_seq)]

        return new_tokens
