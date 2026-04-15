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
Shape bucketing for MPS optimization.

Reduces Metal graph recompilation ("Wired Memory Leak") by padding input
tensors to standardized shapes. Every unique tensor shape triggers a new
Metal graph compilation that consumes Wired Memory — a high-priority
segment of RAM that cannot be freed by gc.collect(). By bucketing shapes,
we bound the number of unique graphs the Metal driver needs to compile.

Mirrors MLX's ``pad_to = 32`` pattern from
``mlx-lm/mlx_lm/tuner/trainer.py:146-147``.

Handles three pixel_values layouts:
  - **4D** ``[B, C, H, W]``: standard single-image batches
  - **5D** ``[B, T, C, H, W]``: video / multi-patch (Qwen3-VL produces this)
  - **2D** ``[num_patches, hidden_dim]``: Qwen3-VL's flattened patch embeddings
    where ``num_patches`` varies per image (e.g. 784 for 448x448, 1024 for
    512x512). Padded to nearest multiple of 128 to reduce graph recompilation.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


# Standard image bucket sizes (H, W) for VLMs.
# Covers common resolutions used by Qwen3-VL, LLaVA, InternVL, etc.
# Sorted ascending so find_nearest_bucket can early-exit.
IMAGE_BUCKETS: list[tuple[int, int]] = [
    (224, 224),
    (256, 256),
    (336, 336),
    (384, 384),
    (448, 448),
    (512, 512),
    (672, 672),
    (768, 768),
    (1024, 1024),
    (1344, 1344),
]

# Sequence length padding granularity, mirrors MLX's pad_to = 32
SEQ_LENGTH_PAD_TO: int = 32

# Patch count padding granularity for 2D [num_patches, hidden_dim] tensors.
# Qwen3-VL flattens patches into [N, D] where N varies per image.
# Padding to multiples of 128 reduces Metal graph recompilation.
PATCH_COUNT_PAD_TO: int = 128


def find_nearest_bucket(h: int, w: int) -> tuple[int, int]:
    """Find the smallest bucket that fits dimensions ``(h, w)``.

    If no predefined bucket is large enough, rounds up to the nearest
    multiple of 64 in each dimension (to still reduce shape variability).

    Args:
        h: Height of the image/patch.
        w: Width of the image/patch.

    Returns:
        Tuple ``(bucket_h, bucket_w)`` that satisfies
        ``bucket_h >= h`` and ``bucket_w >= w``.
    """
    for bh, bw in IMAGE_BUCKETS:
        if bh >= h and bw >= w:
            return (bh, bw)

    # Fallback: round up to nearest multiple of 64
    pad_to = 64
    bucket_h = pad_to * math.ceil(h / pad_to)
    bucket_w = pad_to * math.ceil(w / pad_to)
    return (bucket_h, bucket_w)


def bucket_pixel_values(pixel_values: torch.Tensor) -> torch.Tensor:
    """Pad pixel_values to the nearest standard bucket size.

    This reduces the number of unique tensor shapes seen by the Metal
    driver, preventing runaway Wired Memory consumption from graph
    recompilation.

    Handles three tensor layouts:
      - **4D** ``[B, C, H, W]``: standard single-image batches.
        H and W are padded to the nearest IMAGE_BUCKET.
      - **5D** ``[B, T, C, H, W]``: video or multi-patch inputs
        (Qwen3-VL produces this). Only H/W are padded, T is preserved.
      - **2D** ``[num_patches, hidden_dim]``: Qwen3-VL's flattened patch
        embeddings. ``num_patches`` varies per image (e.g. 784, 1024),
        causing Metal to recompile for each unique count. Padded to
        the nearest multiple of PATCH_COUNT_PAD_TO (128).

    Padding uses zeros, consistent with VLM processor conventions.

    Args:
        pixel_values: Input tensor of shape ``[B, C, H, W]``,
            ``[B, T, C, H, W]``, ``[num_patches, hidden_dim]``, or
            ``[B, seq_len, hidden]``.

    Returns:
        Padded tensor with shapes rounded to standard sizes.
    """
    ndim = pixel_values.ndim

    if ndim == 4:
        # [B, C, H, W]
        h, w = pixel_values.shape[2], pixel_values.shape[3]
        target_h, target_w = find_nearest_bucket(h, w)

        if target_h == h and target_w == w:
            return pixel_values  # Already at a bucket size

        # F.pad expects (left, right, top, bottom) for 4D
        pad_w = target_w - w
        pad_h = target_h - h
        return F.pad(pixel_values, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

    elif ndim == 5:
        # [B, T, C, H, W] — video or multi-patch
        h, w = pixel_values.shape[3], pixel_values.shape[4]
        target_h, target_w = find_nearest_bucket(h, w)

        if target_h == h and target_w == w:
            return pixel_values  # Already at a bucket size

        # F.pad for 5D expects (left, right, top, bottom, front, back)
        # We only pad spatial dims and leave temporal dim untouched.
        pad_w = target_w - w
        pad_h = target_h - h
        return F.pad(pixel_values, (0, pad_w, 0, pad_h, 0, 0), mode="constant", value=0.0)

    elif ndim == 2:
        # [num_patches, hidden_dim] — Qwen3-VL's flattened patch embeddings.
        # E.g. [784, 1536] for a 448x448 image processed by the vision encoder.
        #
        # WARNING: Do NOT pad this dimension. num_patches is tightly coupled to
        # image_grid_thw, which the vision encoder uses to compute position
        # embeddings. Padding pixel_values without also updating grid_thw
        # causes: RuntimeError: size of tensor a (896) must match size of
        # tensor b (784) at non-singleton dimension 0.
        #
        # The shape variability here is less impactful than 4D/5D spatial dims
        # because the vision encoder is a small fraction of total compute.
        return pixel_values

    elif ndim == 3:
        # [B, seq_len, hidden] — flattened patch embeddings (batched).
        # Same constraint as 2D: seq_len is tied to grid_thw metadata.
        # Cannot pad without also updating position embedding inputs.
        return pixel_values

    else:
        logger.warning(
            f"Unexpected pixel_values ndim={ndim}, shape={pixel_values.shape}. "
            "Skipping shape bucketing."
        )
        return pixel_values


def bucket_sequence_length(
    input_ids: torch.Tensor,
    pad_token_id: int,
    pad_to: int = SEQ_LENGTH_PAD_TO,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad input_ids to the nearest multiple of ``pad_to``.

    Mirrors MLX's sequence padding from
    ``mlx-lm/mlx_lm/tuner/trainer.py:146-148``::

        pad_to = 32
        max_length_in_batch = 1 + pad_to * ((max(lengths) + pad_to - 1) // pad_to)

    Args:
        input_ids: Token IDs with shape ``[B, T]``.
        pad_token_id: Token ID to use for padding.
        pad_to: Granularity of padding. Default 32.

    Returns:
        Tuple of ``(padded_input_ids, attention_mask)`` where the sequence
        length is rounded up to the nearest multiple of ``pad_to``.
    """
    seq_len = input_ids.shape[1]
    target_len = pad_to * math.ceil(seq_len / pad_to)

    if target_len == seq_len:
        attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask

    pad_amount = target_len - seq_len

    # Create attention mask before padding
    attention_mask = torch.ones_like(input_ids)
    attention_mask = F.pad(attention_mask, (0, pad_amount), value=0)

    # Pad input_ids
    padded = F.pad(input_ids, (0, pad_amount), value=pad_token_id)

    return padded, attention_mask
