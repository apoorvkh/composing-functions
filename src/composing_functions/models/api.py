from __future__ import annotations

import os
from typing import Literal

import requests

from . import TextGenerationModel

APIModelT = Literal["llama-3-405b", "deepseek-v3", "deepseek-r1", "gpt-4o", "o4-mini"]


class APIModel(TextGenerationModel):
    def __init__(self, model_name: APIModelT):
        self.model_name: APIModelT = model_name

    @property
    def model_path(self) -> str:
        match self.model_name:
            case "llama-3-405b":
                return "meta-llama/llama-3.1-405b-instruct"
            case "deepseek-v3":
                return "deepseek/deepseek-chat-v3-0324"
            case "deepseek-r1":
                return "deepseek/deepseek-r1-0528"
            case "gpt-4o":
                return "openai/gpt-4o-2024-11-20"
            case "o4-mini":
                return "openai/o4-mini"

    @property
    def model_quantization(self) -> Literal["fp4", "fp8", "fp16", "bf16", "fp32"] | None:
        match self.model_name:
            case "llama-3-405b":
                return "bf16"
            case "deepseek-v3" | "deepseek-r1":
                return "fp8"
            case "gpt-4o" | "o4-mini":
                return None

    @property
    def reasoning_budget(self) -> int:
        match self.model_name:
            case "deepseek-r1" | "o4-mini":
                return 2000
            case _:
                return 0

    def generate_continuation(self, text: str, max_new_tokens: int | None = None, stop_seq: str | None = None) -> str:
        max_new_tokens = max_new_tokens or 20
        payload = {
            "model": self.model_path,
            "prompt": text,
            "max_tokens": max_new_tokens,
            "temperature": 0.0,
        }

        if self.model_quantization is not None:
            payload["provider"] = {"quantizations": [self.model_quantization]}

        if stop_seq is not None:
            payload["stop"] = [stop_seq]

        if self.reasoning_budget > 0:
            payload["max_tokens"] += self.reasoning_budget
            payload["reasoning"] = {"max_tokens": self.reasoning_budget}

        response = requests.post(
            "https://openrouter.ai/api/v1/completions",
            headers={
                "Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

        return response.json()["choices"][0]["text"].lstrip()
