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
Apple Silicon MPS memory management utilities for TRL trainers.

Provides synchronization barriers and memory guards that mirror MLX's
implicit memory management (mx.eval, mx.set_wired_limit) in PyTorch's
MPS backend. These utilities solve the "Asynchronous Pile-up" problem
where the CPU queues work faster than the GPU can process it.

Key design principle: torch.mps.synchronize() MUST be called BEFORE
torch.mps.empty_cache(). If empty_cache() is called while the GPU is
still working, the command will be ignored or queued, defeating the
purpose of the sync point.
"""

from __future__ import annotations

import gc
import logging
import threading
from contextlib import contextmanager

import torch


logger = logging.getLogger(__name__)


def is_mps_available() -> bool:
    """Check if MPS (Metal Performance Shaders) backend is available.

    Returns `True` only when both `torch.backends.mps.is_available()` and
    `torch.backends.mps.is_built()` are `True`. This ensures the runtime
    environment has a Metal-capable GPU and PyTorch was compiled with MPS
    support.
    """
    return (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def get_mps_device_info() -> dict:
    """Return MPS device information.

    Returns a dictionary with device metadata. On non-MPS systems, returns
    a stub indicating MPS is unavailable.
    """
    if not is_mps_available():
        return {"device": "none", "mps_available": False}

    import platform
    import os

    return {
        "device": "mps",
        "mps_available": True,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_version": torch.__version__,
    }


def mps_synchronize():
    """Explicit GPU synchronization barrier.

    Forces the MPS GPU to complete all queued operations before the CPU
    continues. This is the PyTorch equivalent of MLX's ``mx.eval()`` —
    it prevents the CPU from racing ahead and queuing more work while
    the GPU is still processing, which would cause intermediate tensors
    to pile up in RAM.

    **IMPORTANT**: This must be called BEFORE ``mps_empty_cache()`` to
    ensure the GPU has finished using all buffers before they are released.

    No-op on non-MPS systems.
    """
    if is_mps_available():
        torch.mps.synchronize()


def mps_empty_cache():
    """Release cached MPS memory allocations back to the system.

    Mirrors MLX's implicit buffer clearing after graph evaluation.
    Only effective AFTER ``mps_synchronize()`` has been called — calling
    this while the GPU is still working will be silently ignored.

    No-op on non-MPS systems.
    """
    if is_mps_available():
        torch.mps.empty_cache()


def mps_sync_and_clear():
    """Combined synchronization barrier + cache clear.

    THE primary memory management call. Insert at every point where MLX
    would call ``mx.eval()``. The ordering is critical:

    1. ``synchronize()`` — wait for GPU to finish all queued work
    2. ``empty_cache()`` — release buffers that are now safe to free

    Calling them in reverse order would be ineffective because
    ``empty_cache()`` cannot free buffers the GPU is still using.

    No-op on non-MPS systems.
    """
    if is_mps_available():
        torch.mps.synchronize()  # Step 1: wait for GPU to finish
        torch.mps.empty_cache()  # Step 2: now safe to release buffers


# Tracks how often cleanup runs
_cleanup_step_counter: int = 0


def _background_gc(triple: bool = False):
    """Run gc.collect in background daemon thread — never blocks main thread."""
    if triple:
        gc.collect()
        gc.collect()
        gc.collect()
        gc.garbage.clear()
    else:
        gc.collect()


# Track peak allocated for memory pressure detection
_peak_allocated_bytes: int = 0


def mps_aggressive_cleanup(every_n_steps: int = 1, deep_clean_interval: int = 100):
    """Per-step MPS memory cleanup — synchronize + empty_cache every N steps.

    On MPS, torch.mps.synchronize() every step is necessary. Without it, the
    CPU races ahead queuing new tensor allocations while the GPU is still
    processing previous steps. This causes memory pile-up that forces the MPS
    allocator into slow recovery mode — far worse than the ~5ms sync cost.

    gc.collect() runs in a background thread to overlap with GPU execution.

    Tier 1 (every N steps):         synchronize + empty_cache + background gc
    Tier 2 (every deep_interval):   triple gc + log allocated memory

    No-op on non-MPS systems.
    """
    if not is_mps_available():
        return

    global _cleanup_step_counter, _peak_allocated_bytes
    _cleanup_step_counter += 1

    is_deep_clean = (_cleanup_step_counter % deep_clean_interval == 0)

    if _cleanup_step_counter % every_n_steps == 0:
        # Always sync + clear: prevents CPU from racing ahead of GPU.
        # ~5ms overhead per step is negligible vs memory pile-up cost.
        torch.mps.synchronize()
        torch.mps.empty_cache()

        if is_deep_clean:
            threading.Thread(target=_background_gc, args=(True,), daemon=True).start()
            allocated = torch.mps.current_allocated_memory()
            _peak_allocated_bytes = max(_peak_allocated_bytes, allocated)
            logger.info(
                f"MPS deep clean at step {_cleanup_step_counter}: "
                f"allocated={allocated / (1024**3):.2f} GiB"
            )
        else:
            threading.Thread(target=_background_gc, daemon=True).start()



@contextmanager
def mps_memory_guard(fraction: float = 0.9):
    """Context manager that sets MPS memory limits and ensures cleanup.

    On entry, optionally sets the per-process memory fraction via
    ``torch.mps.set_per_process_memory_fraction()``. On exit, performs
    a full sync + cache clear to return memory to the system.

    **IMPORTANT**: If the user has set ``PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0``
    to disable the memory ceiling (required on 16GB Macs), this function
    will NOT call ``set_per_process_memory_fraction()`` because doing so
    would re-impose a hard cap that conflicts with the user's intent.

    Args:
        fraction: Fraction of total system memory available to MPS.
            Default 0.9 (90%). Set to 1.0 to skip the fraction call.
            Ignored when ``PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0``.

    No-op on non-MPS systems (yields immediately).
    """
    if not is_mps_available():
        logger.debug("MPS not available, memory guard is a no-op")
        yield
        return

    import os
    watermark = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "")

    # Don't impose a memory cap if:
    # 1. User set HIGH_WATERMARK_RATIO=0.0 (they want unlimited memory)
    # 2. fraction >= 1.0 (no point capping at 100%+)
    if watermark == "0.0" or watermark == "0":
        logger.info(
            "MPS memory guard: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 detected, "
            "skipping set_per_process_memory_fraction to allow unlimited memory"
        )
    elif fraction >= 1.0:
        logger.info("MPS memory guard: fraction >= 1.0, no memory cap applied")
    else:
        logger.info(
            f"MPS memory guard: setting per-process memory fraction to {fraction:.1%}"
        )
        torch.mps.set_per_process_memory_fraction(fraction)

    try:
        yield
    finally:
        logger.info("MPS memory guard: cleaning up — sync + cache clear")
        mps_sync_and_clear()

