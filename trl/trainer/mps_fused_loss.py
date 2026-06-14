# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fused loss functions optimized for Apple Silicon MPS.

Solves the "Logit Bottleneck": standard TRL/PyTorch projects hidden states
through the full vocabulary to create a ``[B, T, V]`` logit tensor before
computing cross-entropy loss. For a 4B model with a 152K vocabulary, this
tensor alone consumes ~600MB per batch.

This module provides:

1. **Chunked Fused Linear+CE**: Processes the vocabulary in slices of
   ``chunk_size`` (default 4096), computing cross-entropy incrementally.
   Peak memory is ``[B, T, chunk_size]`` instead of ``[B, T, V]``.

2. **KL/JS Divergence**: PyTorch-native implementations equivalent to
   MLX's Metal kernels from ``mlx-lm/mlx_lm/tuner/losses.py``.

All functions work on any device (CPU/CUDA/MPS) — MPS-specific
synchronization is handled by the caller via ``mps_utils``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# Default chunk size tuned for M4 Pro dispatch latency vs memory tradeoff.
# 65536 reduces Python MPS dispatch iterations from 19 to 3 for 152K vocab,
# recovering ~5s/step while keeping peak memory well below monolithic logits.
DEFAULT_CHUNK_SIZE: int = 65536


class ChunkedFusedLinearCrossEntropy(torch.autograd.Function):
    """Custom autograd function that fuses lm_head projection + CE loss.

    Standard flow (wasteful)::

        logits = lm_head(hidden)     # [B, T, V] — 600MB for 4B model
        loss = CE(logits, labels)    # scalar
        # logits stays in memory until backward completes

    Fused flow (ours) — two-pass for exact results::

        Pass 1: Compute global logsumexp over all vocab chunks (online algorithm).
        Pass 2: Gather target logits from the correct chunk, compute loss.
        CE = -target_logit + logsumexp,  averaged over valid tokens.

    The backward pass recomputes chunks on-the-fly and uses the saved
    global logsumexp to compute correct global softmax probabilities.
    """

    @staticmethod
    def forward(
        ctx,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        bias: torch.Tensor | None = None,
        ignore_index: int = -100,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> torch.Tensor:
        """Forward pass: two-pass chunked matmul + CE.

        Pass 1 computes the global log-sum-exp across all vocabulary tokens
        using the numerically stable online algorithm:
            max_so_far, sum_exp_so_far  (updated chunk by chunk)
            logsumexp = max + log(sum_exp)

        Pass 2 gathers ``logits[b, t, labels[b, t]]`` from the chunk that
        contains each label, then computes::

            CE = mean_valid(-target_logit + logsumexp)

        Args:
            hidden_states: ``[B, T, D]`` — last hidden state from model.
            weight: ``[V, D]`` — lm_head weight matrix.
            labels: ``[B, T]`` — target token IDs.
            bias: ``[V]`` — optional lm_head bias.
            ignore_index: Label value to ignore in CE. Default -100.
            chunk_size: Number of vocabulary tokens per chunk. Default 4096.

        Returns:
            Scalar loss tensor.
        """
        B, T, D = hidden_states.shape
        V = weight.shape[0]

        valid_mask = labels != ignore_index
        valid_count = valid_mask.sum().float().clamp(min=1.0)

        # ------------------------------------------------------------------
        # Pass 1: Online logsumexp over all vocabulary chunks
        # ------------------------------------------------------------------
        # We maintain running (max, sum-of-exp) per position [B, T, 1]:
        #   For each new chunk of logits x_c:
        #     new_max = max(running_max, chunk_max)
        #     sum_exp = sum_exp * exp(old_max - new_max) + sum(exp(x_c - new_max))
        #     running_max = new_max
        #   Final result: logsumexp = running_max + log(sum_exp)
        # CRITICAL: accumulators MUST be FP32 to prevent overflow.
        # BFloat16 has only 7 mantissa bits — summing exp() across 65536
        # elements causes catastrophic precision loss → inf → NaN.
        acc_dtype = torch.float32
        running_max = torch.full(
            (B, T, 1), -float("inf"), device=hidden_states.device, dtype=acc_dtype
        )
        running_sum_exp = torch.zeros(
            (B, T, 1), device=hidden_states.device, dtype=acc_dtype
        )

        for v_start in range(0, V, chunk_size):
            v_end = min(v_start + chunk_size, V)
            chunk_logits = torch.matmul(hidden_states, weight[v_start:v_end].t())
            if bias is not None:
                chunk_logits = chunk_logits + bias[v_start:v_end]

            chunk_max = chunk_logits.max(dim=-1, keepdim=True).values.to(acc_dtype)  # [B, T, 1]
            new_max = torch.maximum(running_max, chunk_max)

            # Rescale previous sum_exp to the new max, then add this chunk.
            # All exp/sum arithmetic in FP32 to prevent BF16 overflow.
            running_sum_exp = (
                running_sum_exp * torch.exp(running_max - new_max)
                + torch.exp(chunk_logits.to(acc_dtype) - new_max).sum(dim=-1, keepdim=True)
            )
            running_max = new_max

            del chunk_logits, chunk_max

        logsumexp = (running_max + torch.log(running_sum_exp)).squeeze(-1)  # [B, T], FP32
        del running_max, running_sum_exp

        # ------------------------------------------------------------------
        # Pass 2: Gather target logits and compute loss
        # ------------------------------------------------------------------
        # For each valid position (b, t), we need logits[b, t, labels[b, t]].
        # We only compute the chunk that contains the target label.
        target_logits = torch.zeros(B, T, device=hidden_states.device, dtype=acc_dtype)

        for v_start in range(0, V, chunk_size):
            v_end = min(v_start + chunk_size, V)
            in_range = valid_mask & (labels >= v_start) & (labels < v_end)
            # REMOVED: `if not in_range.any(): continue` — .any() forces CPU-GPU sync stall
            # PyTorch handles empty masks correctly, so this guard is pure overhead on MPS.

            chunk_logits = torch.matmul(hidden_states, weight[v_start:v_end].t()).to(acc_dtype)
            if bias is not None:
                chunk_logits = chunk_logits + bias[v_start:v_end]

            # Remap labels to chunk-local indices  [0, chunk_size)
            local_labels = (labels - v_start).clamp(min=0, max=v_end - v_start - 1)
            gathered = chunk_logits.gather(-1, local_labels.unsqueeze(-1)).squeeze(-1)

            # Only accumulate for positions whose label actually falls in this chunk
            target_logits = target_logits + gathered * in_range.float()

            del chunk_logits, gathered

        # CE = mean_valid( -target_logit + logsumexp )  — all in FP32
        per_token_loss = -target_logits + logsumexp  # [B, T], FP32
        loss = (per_token_loss * valid_mask.float()).sum() / valid_count

        # Save for backward — logsumexp is small [B, T], NOT [B, T, V]
        ctx.save_for_backward(hidden_states, weight, labels, logsumexp)
        ctx._bias = bias
        ctx.ignore_index = ignore_index
        ctx.chunk_size = chunk_size
        ctx._valid_count = valid_count

        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass: recompute chunks on-the-fly with global softmax.

        Uses the saved ``logsumexp`` to compute correct global softmax
        probabilities for each chunk::

            softmax_chunk = exp(chunk_logits - logsumexp)

        The CE gradient is then ``softmax - one_hot(label)`` for each chunk,
        masked to valid positions and accumulated into parameter gradients.
        """
        hidden_states, weight, labels, logsumexp = ctx.saved_tensors
        bias = ctx._bias
        ignore_index = ctx.ignore_index
        chunk_size = ctx.chunk_size
        valid_count = ctx._valid_count

        B, T, D = hidden_states.shape
        V = weight.shape[0]

        valid_mask = labels != ignore_index
        scale = grad_output / valid_count

        grad_hidden = torch.zeros_like(hidden_states)
        grad_weight = torch.zeros_like(weight)
        grad_bias = torch.zeros_like(bias) if bias is not None else None

        for v_start in range(0, V, chunk_size):
            v_end = min(v_start + chunk_size, V)
            chunk_weight = weight[v_start:v_end]

            # Recompute chunk logits — cast to FP32 to match saved logsumexp precision
            chunk_logits = torch.matmul(hidden_states, chunk_weight.t()).float()
            if bias is not None:
                chunk_logits = chunk_logits + bias[v_start:v_end].float()

            # Global softmax for this chunk: exp(logits_chunk - logsumexp)
            # logsumexp is already FP32 from forward; chunk_logits must match
            chunk_probs = torch.exp(chunk_logits - logsumexp.unsqueeze(-1))  # [B, T, chunk_size], FP32

            # CE gradient = probs - one_hot(label)
            in_range = valid_mask & (labels >= v_start) & (labels < v_end)
            # REMOVED: `if in_range.any():` — .any() forces CPU-GPU sync stall on MPS.
            # Dense operations handle empty masks correctly without the guard.
            local_labels = (labels - v_start).clamp(min=0, max=v_end - v_start - 1)
            one_hot = torch.zeros_like(chunk_probs)
            one_hot.scatter_(-1, local_labels.unsqueeze(-1), 1.0)
            
            # Mask invalid positions natively without .float() casts
            in_range_expanded = in_range.unsqueeze(-1)
            grad_logits = torch.where(in_range_expanded, chunk_probs - one_hot, chunk_probs)

            # Mask valid bounds directly and apply scale factor
            valid_mask_expanded = valid_mask.unsqueeze(-1)
            grad_logits = torch.where(valid_mask_expanded, grad_logits * scale, 0.0)

            # Cast back to model dtype for parameter gradient accumulation
            grad_logits_native = grad_logits.to(hidden_states.dtype)

            # Accumulate parameter gradients
            grad_hidden += torch.matmul(grad_logits_native, chunk_weight)
            grad_weight[v_start:v_end] += torch.matmul(
                grad_logits_native.reshape(-1, v_end - v_start).t(),
                hidden_states.reshape(-1, D),
            )
            if grad_bias is not None:
                grad_bias[v_start:v_end] += grad_logits.reshape(-1, v_end - v_start).sum(0)

            del chunk_logits, chunk_probs, grad_logits, grad_logits_native

        return grad_hidden, grad_weight, None, grad_bias, None, None


def mps_fused_cross_entropy(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    lm_head_bias: torch.Tensor | None = None,
    ignore_index: int = -100,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> torch.Tensor:
    """Compute cross-entropy loss without materializing the full logit tensor.

    This is the primary entry point for the fused loss. When the vocabulary
    is large (e.g. 152K for Qwen3-VL), this saves ~600MB of peak memory
    per batch compared to the standard ``logits = lm_head(hidden); CE(logits, labels)``
    pattern.

    Falls back to standard cross-entropy when chunk_size >= vocab_size
    (no benefit from chunking for small vocabularies).

    Args:
        hidden_states: ``[B, T, D]`` from the last transformer layer.
        lm_head_weight: ``[V, D]`` weight matrix of the language model head.
        labels: ``[B, T]`` target token IDs.
        lm_head_bias: Optional ``[V]`` bias of the language model head.
        ignore_index: Label value to ignore in CE loss. Default -100.
        chunk_size: Vocabulary chunk size. Default 4096 (optimized for M4 Pro).

    Returns:
        Scalar loss tensor with gradients.
    """
    V = lm_head_weight.shape[0]

    if chunk_size >= V:
        # No benefit from chunking for small vocabularies
        logits = F.linear(hidden_states, lm_head_weight, lm_head_bias)
        return F.cross_entropy(
            logits.reshape(-1, V),
            labels.reshape(-1),
            ignore_index=ignore_index,
        )

    return ChunkedFusedLinearCrossEntropy.apply(
        hidden_states, lm_head_weight, labels, lm_head_bias, ignore_index, chunk_size
    )


# ============================================================================
# KL and JS Divergence Losses
# ============================================================================
# PyTorch equivalents of MLX's Metal kernels from
# mlx-lm/mlx_lm/tuner/losses.py (_make_kl_forward_kernel, etc.)


def kl_div_loss(
    logits_q: torch.Tensor,
    logits_p: torch.Tensor,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """KL divergence from log-softmax of logits.

    Computes ``KL(q || p)`` where ``q`` and ``p`` are distributions defined
    by the logits. Uses log-softmax for numerical stability.

    Equivalent to MLX's ``_make_kl_forward_kernel`` but using pure PyTorch ops
    (works on CPU, CUDA, and MPS).

    Args:
        logits_q: ``[B, T, V]`` or ``[B, V]`` — logits for distribution q.
        logits_p: ``[B, T, V]`` or ``[B, V]`` — logits for distribution p.
        reduction: Reduction mode. One of ``"batchmean"``, ``"mean"``,
            ``"sum"``, ``"none"``.

    Returns:
        KL divergence loss.
    """
    log_q = F.log_softmax(logits_q, dim=-1)
    log_p = F.log_softmax(logits_p, dim=-1)
    return F.kl_div(log_q, log_p, log_target=True, reduction=reduction)


def js_div_loss(
    logits_q: torch.Tensor,
    logits_p: torch.Tensor,
    beta: float = 0.5,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """Jensen-Shannon divergence from logits.

    Computes ``JS(q, p) = beta * KL(q || m) + (1-beta) * KL(p || m)``
    where ``m = beta * q + (1-beta) * p`` is the mixture distribution.

    Equivalent to MLX's ``_make_js_forward_kernel`` and
    ``_make_js_backward_kernel`` but using pure PyTorch ops.

    Args:
        logits_q: ``[B, T, V]`` or ``[B, V]`` — logits for distribution q.
        logits_p: ``[B, T, V]`` or ``[B, V]`` — logits for distribution p.
        beta: Mixing coefficient. Default 0.5 (symmetric JS divergence).
        reduction: Reduction mode. One of ``"batchmean"``, ``"mean"``,
            ``"sum"``, ``"none"``.

    Returns:
        JS divergence loss.
    """
    # Convert to probabilities for the mixture
    q = F.softmax(logits_q, dim=-1)
    p = F.softmax(logits_p, dim=-1)

    # Mixture distribution
    m = beta * q + (1 - beta) * p

    # KL divergences using log-probs for stability
    log_q = F.log_softmax(logits_q, dim=-1)
    log_p = F.log_softmax(logits_p, dim=-1)
    log_m = torch.log(m + 1e-8)  # small epsilon for numerical stability

    kl_q_m = F.kl_div(log_m, log_q, log_target=True, reduction=reduction)
    kl_p_m = F.kl_div(log_m, log_p, log_target=True, reduction=reduction)

    return beta * kl_q_m + (1 - beta) * kl_p_m
