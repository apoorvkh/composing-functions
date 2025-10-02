"""Mechanistic interpretability tools for analyzing language model internals.

This module provides methods for extracting and analyzing model activations,
including logit lens, token identity analysis, and residual stream extraction.
"""

import random

import torch
from einops import rearrange

from .models.transformers import NNsightModel


def residual_stream(model: NNsightModel, prompt: str) -> torch.Tensor:
    """
    Extract residual stream activations from all layers of a transformer model.

    Returns:
        torch.Tensor: Activations with shape (positions, layers, hidden_dim)
    """

    outputs = []
    with torch.no_grad():
        with model.nnsight_model.trace(prompt):
            for layer in model.layers:
                _output = layer.output[0][0].cpu().save()
                outputs.append(_output)

    return rearrange(torch.stack(outputs), "layers positions hidden_dim -> positions layers hidden_dim")


def logit_lens(model: NNsightModel, residual_stream_acts: torch.Tensor) -> torch.Tensor:
    """
    Apply logit lens to project residual stream activations to vocabulary space.

    Returns:
        torch.Tensor: Logit scores with shape (positions, layers, vocab_size)
            Each entry [pos, layer, token] represents the logit score for that token
            at the specified position and layer.
    """
    residual_stream_acts = residual_stream_acts.to(device=model.device)

    with torch.no_grad():
        logits = model.lm_head._module(
            model.norm._module(
                rearrange(
                    residual_stream_acts,
                    "positions layers hidden_dim -> (positions layers) hidden_dim",
                )
            )
        ).cpu()

    return rearrange(
        logits,
        "(positions layers) vocab -> positions layers vocab",
        positions=residual_stream_acts.shape[0],
        layers=residual_stream_acts.shape[1],
    )


def token_identity(
    model: NNsightModel,
    residual_stream_acts: torch.Tensor,
    seed: int = 0,
) -> torch.Tensor:
    """
    Decode residual stream activations using token identity patchscope (Ghandeharioun 2024; Sec. 4.1)
        - Identity prompt : "[token1] [token1] [;] [token2] [token2] [;] ... [?]"
        - Randomly samples 10 tokens for the identity context

    Args:
        model (NNsightModel)
        residual_stream_acts (torch.Tensor): Activations with shape (positions, layers, hidden_dim)
        seed (int, optional)

    Returns:
        torch.Tensor: Logit scores with shape (positions, layers, vocab_size)
            Each entry represents the model's confidence that the activation
            corresponds to each vocabulary token in the identity context.
    """
    tokenizer = model.nnsight_model.tokenizer
    random.seed(seed)
    sampled_tokens = list(random.sample(range(tokenizer.vocab_size), 10))
    sep_token = tokenizer.encode(";", add_special_tokens=False)[0]
    query_token = tokenizer.encode("?", add_special_tokens=False)[0]

    identity_prompt = []
    if tokenizer.bos_token_id is not None:
        identity_prompt.append(tokenizer.bos_token_id)
    for t in sampled_tokens:
        identity_prompt += [t, t, sep_token]
    identity_prompt.append(query_token)

    P, L = residual_stream_acts.shape[:2]

    logits = []
    with torch.no_grad():
        for layer_idx in range(L):  # IMPORTANT (nnsight): must patch in order of layers
            with model.nnsight_model.trace([identity_prompt] * P):
                # patching from some activation to (same layer, final token) patchscope
                model.layers[layer_idx].output[0][:, -1, :] = residual_stream_acts[:, layer_idx, :]
                # save logits at final token
                _logits = model.lm_head.output[:, -1, :].cpu().save()
                logits.append(_logits)

    return rearrange(torch.stack(logits), "layers positions vocab -> positions layers vocab")


def argsort_logits(logits: torch.Tensor) -> torch.Tensor:
    """
    Sort vocabulary indices by logit scores in descending order.

    Args:
        logits: Logit scores with shape (positions, layers, vocab_size)

    Returns:
        torch.Tensor: Sorted token indices with shape (positions*layers, vocab_size)
    """
    return torch.argsort(
        rearrange(logits, "positions layers vocab -> (positions layers) vocab"),
        descending=True,
        dim=1,
    )


def reciprocal_rank(sort_indices: torch.Tensor, shape: tuple[int, ...], token: int) -> torch.Tensor:
    """
    Calculate reciprocal rank of a specific token across positions and layers.

    Args:
        sort_indices: Token indices sorted by logit score, shape (positions*layers, vocab_size)
        shape: Tuple specifying the original (positions, layers) dimensions
        token: Vocabulary index of the token to analyze

    Returns:
        torch.Tensor: Reciprocal ranks with shape (positions, layers)
    """
    ranks = (sort_indices == token).nonzero(as_tuple=False)[:, 1].reshape(*shape)
    return 1 / (ranks + 1)
