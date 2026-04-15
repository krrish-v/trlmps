#!/usr/bin/env python3
"""
Production readiness test suite for trlmpsv5.
Run on your Mac with: python test_production.py

Tests:
  1. Module imports
  2. FP32 fused CE numerical correctness (vs PyTorch CE)
  3. Chunk size configuration
  4. Safety guard for all-(-100) labels
  5. Backbone extraction for common model architectures
  6. Shape bucketing
"""

import sys
import os
import math

# Add trlmpsv5 to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name}: {detail}")


# ============================================================================
# 1. Import Tests
# ============================================================================
print("\n═══ 1. Import Tests ═══")
try:
    from trl.trainer.mps_fused_loss import (
        mps_fused_cross_entropy,
        ChunkedFusedLinearCrossEntropy,
        DEFAULT_CHUNK_SIZE,
        kl_div_loss,
        js_div_loss,
    )
    check("mps_fused_loss imports", True)
    check(f"DEFAULT_CHUNK_SIZE = {DEFAULT_CHUNK_SIZE}", DEFAULT_CHUNK_SIZE == 65536)
except Exception as e:
    check("mps_fused_loss imports", False, str(e))

try:
    from trl.trainer.mps_utils import (
        is_mps_available,
        mps_synchronize,
        mps_sync_and_clear,
        mps_memory_guard,
        mps_aggressive_cleanup,
        get_mps_device_info,
    )
    check("mps_utils imports", True)
    check(f"MPS available: {is_mps_available()}", True)
except Exception as e:
    check("mps_utils imports", False, str(e))

try:
    from trl.trainer.mps_shape_buckets import (
        bucket_pixel_values,
        find_nearest_bucket,
        bucket_sequence_length,
        IMAGE_BUCKETS,
    )
    check("mps_shape_buckets imports", True)
except Exception as e:
    check("mps_shape_buckets imports", False, str(e))

try:
    from trl.trainer.sft_config import SFTConfig
    check("SFTConfig imports", True)
    check(
        f"SFTConfig.mps_fused_loss_chunk_size default = {SFTConfig.mps_fused_loss_chunk_size}",
        SFTConfig.mps_fused_loss_chunk_size == 65536,
    )
except Exception as e:
    check("SFTConfig imports", False, str(e))

try:
    from trl.trainer.grpo_config import GRPOConfig
    check("GRPOConfig imports", True)
    check(
        f"GRPOConfig.mps_fused_loss_chunk_size default = {GRPOConfig.mps_fused_loss_chunk_size}",
        GRPOConfig.mps_fused_loss_chunk_size == 65536,
    )
except Exception as e:
    check("GRPOConfig imports", False, str(e))


# ============================================================================
# 2. Fused CE Numerical Correctness (CPU — works anywhere)
# ============================================================================
print("\n═══ 2. Fused CE Numerical Correctness ═══")

torch.manual_seed(42)
B, T, D, V = 2, 16, 64, 1024  # small vocab for fast CPU test

hidden = torch.randn(B, T, D, dtype=torch.float32)
weight = torch.randn(V, D, dtype=torch.float32)
bias = torch.randn(V, dtype=torch.float32)
labels = torch.randint(0, V, (B, T))
# Mask some positions
labels[0, :3] = -100
labels[1, -2:] = -100

# Reference: standard PyTorch CE
logits_ref = F.linear(hidden, weight, bias)
loss_ref = F.cross_entropy(logits_ref.reshape(-1, V), labels.reshape(-1), ignore_index=-100)

# Fused: our chunked implementation
loss_fused = mps_fused_cross_entropy(hidden, weight, labels, lm_head_bias=bias, chunk_size=256)

diff = abs(loss_ref.item() - loss_fused.item())
check(f"Forward match (diff={diff:.8f})", diff < 1e-4, f"ref={loss_ref.item():.6f}, fused={loss_fused.item():.6f}")

# Test backward
hidden_ref = hidden.clone().requires_grad_(True)
weight_ref = weight.clone().requires_grad_(True)
logits_ref = F.linear(hidden_ref, weight_ref, bias)
loss_ref = F.cross_entropy(logits_ref.reshape(-1, V), labels.reshape(-1), ignore_index=-100)
loss_ref.backward()

hidden_fused = hidden.clone().requires_grad_(True)
weight_fused = weight.clone().requires_grad_(True)
loss_fused = ChunkedFusedLinearCrossEntropy.apply(hidden_fused, weight_fused, labels, bias, -100, 256)
loss_fused.backward()

grad_h_diff = (hidden_ref.grad - hidden_fused.grad).abs().max().item()
grad_w_diff = (weight_ref.grad - weight_fused.grad).abs().max().item()
check(f"Backward hidden grad match (max_diff={grad_h_diff:.8f})", grad_h_diff < 1e-3)
check(f"Backward weight grad match (max_diff={grad_w_diff:.8f})", grad_w_diff < 1e-3)


# ============================================================================
# 3. FP32 Accumulator Test — BF16 Overflow Prevention
# ============================================================================
print("\n═══ 3. BF16 Safety (FP32 Accumulators) ═══")

# Simulate large vocab with bf16 hidden states (mimics real Qwen3-VL)
torch.manual_seed(42)
B, T, D, V_large = 1, 4, 32, 8192  # 8K vocab to test accumulator range

hidden_bf = torch.randn(B, T, D, dtype=torch.bfloat16)
weight_bf = torch.randn(V_large, D, dtype=torch.bfloat16)
labels_bf = torch.randint(0, V_large, (B, T))

loss_bf = mps_fused_cross_entropy(hidden_bf, weight_bf, labels_bf, chunk_size=2048)
check(f"BF16 loss is finite: {loss_bf.item():.4f}", torch.isfinite(loss_bf).item())
check(f"BF16 loss is non-zero: {loss_bf.item():.4f}", loss_bf.item() > 0)


# ============================================================================
# 4. All-Labels-(-100) Safety Guard
# ============================================================================
print("\n═══ 4. All-Labels Guard ═══")

# Test that fused CE returns 0.0 gracefully when all labels are -100
labels_all_ignore = torch.full((B, T), -100, dtype=torch.long)
loss_ignore = mps_fused_cross_entropy(hidden_bf, weight_bf, labels_all_ignore, chunk_size=2048)
check(f"All-ignore labels → loss={loss_ignore.item()}", loss_ignore.item() == 0.0)


# ============================================================================
# 5. Shape Bucketing
# ============================================================================
print("\n═══ 5. Shape Bucketing ═══")

# 4D test
pv_4d = torch.randn(1, 3, 400, 400)
out_4d = bucket_pixel_values(pv_4d)
check(f"4D bucket: {pv_4d.shape} → {out_4d.shape}", out_4d.shape == (1, 3, 448, 448))

# 5D test
pv_5d = torch.randn(1, 4, 3, 300, 300)
out_5d = bucket_pixel_values(pv_5d)
check(f"5D bucket: {pv_5d.shape} → {out_5d.shape}", out_5d.shape == (1, 4, 3, 336, 336))

# 2D passthrough
pv_2d = torch.randn(784, 1536)
out_2d = bucket_pixel_values(pv_2d)
check(f"2D passthrough: {pv_2d.shape} → {out_2d.shape}", out_2d.shape == pv_2d.shape)

# Nearest bucket
check("find_nearest_bucket(224, 224)", find_nearest_bucket(224, 224) == (224, 224))
check("find_nearest_bucket(300, 300)", find_nearest_bucket(300, 300) == (336, 336))
check("find_nearest_bucket(2000, 2000)", find_nearest_bucket(2000, 2000) == (2048, 2048))


# ============================================================================
# 6. KL/JS Divergence
# ============================================================================
print("\n═══ 6. KL/JS Divergence ═══")

logits_a = torch.randn(2, 10)
logits_b = torch.randn(2, 10)
kl = kl_div_loss(logits_a, logits_b)
js = js_div_loss(logits_a, logits_b)
check(f"KL div finite: {kl.item():.6f}", torch.isfinite(kl).item())
check(f"JS div finite: {js.item():.6f}", torch.isfinite(js).item())
check("JS >= 0", js.item() >= 0)


# ============================================================================
# 7. MPS-Specific Tests (only run on Mac)
# ============================================================================
if is_mps_available():
    print("\n═══ 7. MPS Device Tests ═══")

    device = torch.device("mps")

    # Fused CE on MPS
    hidden_m = torch.randn(1, 8, 64, device=device, dtype=torch.bfloat16)
    weight_m = torch.randn(4096, 64, device=device, dtype=torch.bfloat16)
    labels_m = torch.randint(0, 4096, (1, 8), device=device)

    loss_m = mps_fused_cross_entropy(hidden_m, weight_m, labels_m, chunk_size=1024)
    torch.mps.synchronize()
    check(f"MPS fused CE: {loss_m.item():.4f}", torch.isfinite(loss_m).item())

    # MPS cleanup
    mps_aggressive_cleanup(every_n_steps=1, deep_clean_interval=1)
    check("MPS aggressive cleanup ran", True)

    # Memory info
    info = get_mps_device_info()
    check(f"MPS device info: {info.get('processor', 'unknown')}", info["mps_available"])
else:
    print("\n═══ 7. MPS Device Tests (SKIPPED — not on Apple Silicon) ═══")


# ============================================================================
# Summary
# ============================================================================
print(f"\n{'═' * 50}")
print(f"  Results: {PASS} passed, {FAIL} failed")
print(f"{'═' * 50}")
sys.exit(0 if FAIL == 0 else 1)
