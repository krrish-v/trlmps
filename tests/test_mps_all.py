# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
# Licensed under the Apache License, Version 2.0
#
# Consolidated tests for all MPS optimization modules:
#   - mps_utils.py        (sync barriers, memory guard)
#   - mps_shape_buckets.py (4D/5D pixel_values bucketing, seq padding)
#   - mps_fused_loss.py    (chunked fused linear+CE, KL/JS divergence)
#
# Run:  python3 tests/test_mps_all.py
# Or:   python3 -m pytest tests/test_mps_all.py -v

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from trl.trainer.mps_utils import (
    is_mps_available,
    get_mps_device_info,
    mps_synchronize,
    mps_empty_cache,
    mps_sync_and_clear,
    mps_memory_guard,
    mps_aggressive_cleanup,
)
from trl.trainer.mps_shape_buckets import (
    find_nearest_bucket,
    bucket_pixel_values,
    bucket_sequence_length,
    IMAGE_BUCKETS,
    SEQ_LENGTH_PAD_TO,
    PATCH_COUNT_PAD_TO,
)
from trl.trainer.mps_fused_loss import (
    mps_fused_cross_entropy,
    kl_div_loss,
    js_div_loss,
    DEFAULT_CHUNK_SIZE,
)


# ============================================================================
# MPS Utils Tests
# ============================================================================

class TestIsMpsAvailable(unittest.TestCase):
    def test_returns_bool(self):
        self.assertIsInstance(is_mps_available(), bool)

    @patch("torch.backends.mps.is_available", return_value=True)
    @patch("torch.backends.mps.is_built", return_value=True)
    def test_true_when_both_true(self, *_):
        self.assertTrue(is_mps_available())

    @patch("torch.backends.mps.is_available", return_value=False)
    @patch("torch.backends.mps.is_built", return_value=True)
    def test_false_when_not_available(self, *_):
        self.assertFalse(is_mps_available())


class TestGetMpsDeviceInfo(unittest.TestCase):
    def test_returns_dict(self):
        self.assertIsInstance(get_mps_device_info(), dict)

    @patch("trl.trainer.mps_utils.is_mps_available", return_value=False)
    def test_unavailable_stub(self, _):
        info = get_mps_device_info()
        self.assertEqual(info["device"], "none")
        self.assertFalse(info["mps_available"])

    @patch("trl.trainer.mps_utils.is_mps_available", return_value=True)
    def test_available_info(self, _):
        info = get_mps_device_info()
        self.assertEqual(info["device"], "mps")
        self.assertIn("torch_version", info)


class TestMpsSyncFunctions(unittest.TestCase):
    @patch("trl.trainer.mps_utils.is_mps_available", return_value=False)
    def test_sync_and_clear_noop(self, _):
        """No-op on non-MPS systems."""
        mps_synchronize()
        mps_empty_cache()
        mps_sync_and_clear()

    @patch("trl.trainer.mps_utils.is_mps_available", return_value=True)
    @patch("torch.mps.synchronize")
    @patch("torch.mps.empty_cache")
    def test_sync_before_clear_order(self, mock_clear, mock_sync, _):
        """CRITICAL: synchronize() MUST be called BEFORE empty_cache()."""
        order = []
        mock_sync.side_effect = lambda: order.append("sync")
        mock_clear.side_effect = lambda: order.append("clear")
        mps_sync_and_clear()
        self.assertEqual(order, ["sync", "clear"])


class TestMpsMemoryGuard(unittest.TestCase):
    @patch("trl.trainer.mps_utils.is_mps_available", return_value=False)
    def test_noop_on_non_mps(self, _):
        with mps_memory_guard(0.9):
            pass

    @patch("trl.trainer.mps_utils.is_mps_available", return_value=True)
    @patch("torch.mps.set_per_process_memory_fraction")
    @patch("trl.trainer.mps_utils.mps_sync_and_clear")
    def test_sets_fraction_and_cleans_up(self, mock_clear, mock_set, _):
        with mps_memory_guard(0.85):
            mock_set.assert_called_once_with(0.85)
            mock_clear.assert_not_called()
        mock_clear.assert_called_once()

    @patch("trl.trainer.mps_utils.is_mps_available", return_value=True)
    @patch("torch.mps.set_per_process_memory_fraction")
    @patch("trl.trainer.mps_utils.mps_sync_and_clear")
    def test_cleans_up_on_exception(self, mock_clear, mock_set, _):
        with self.assertRaises(RuntimeError):
            with mps_memory_guard(0.9):
                raise RuntimeError("boom")
        mock_clear.assert_called_once()


class TestMpsAggressiveCleanup(unittest.TestCase):
    @patch("trl.trainer.mps_utils.is_mps_available", return_value=False)
    def test_noop_on_non_mps(self, _):
        mps_aggressive_cleanup()

    @patch("trl.trainer.mps_utils.is_mps_available", return_value=True)
    @patch("torch.mps.synchronize")
    @patch("torch.mps.empty_cache")
    @patch("trl.trainer.mps_utils.gc.collect")
    def test_gc_collect_called(self, mock_gc, mock_clear, mock_sync, _):
        """gc.collect must be called to break ref cycles holding dead tensors."""
        import trl.trainer.mps_utils as _mod
        _mod._aggressive_cleanup_counter = 0  # reset counter
        mps_aggressive_cleanup(every_n_steps=1)
        mock_gc.assert_called_once()
        mock_sync.assert_called_once()
        mock_clear.assert_called_once()

    @patch("trl.trainer.mps_utils.is_mps_available", return_value=True)
    @patch("torch.mps.synchronize")
    @patch("torch.mps.empty_cache")
    @patch("trl.trainer.mps_utils.gc.collect")
    def test_throttled_gc(self, mock_gc, mock_clear, mock_sync, _):
        """gc.collect runs only every N steps when throttled (two-tier system)."""
        import trl.trainer.mps_utils as _mod
        _mod._cleanup_step_counter = 0  # reset (updated counter name for two-tier)
        mps_aggressive_cleanup(every_n_steps=3)  # step 1 → 1%3≠0, no gc
        mock_gc.assert_not_called()
        mock_sync.assert_called()  # sync always happens
        mps_aggressive_cleanup(every_n_steps=3)  # step 2 → 2%3≠0, no gc
        mock_gc.assert_not_called()
        mps_aggressive_cleanup(every_n_steps=3)  # step 3 → 3%3=0, gc!
        mock_gc.assert_called_once()


# ============================================================================
# Shape Bucketing Tests
# ============================================================================

class TestFindNearestBucket(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(find_nearest_bucket(224, 224), (224, 224))
        self.assertEqual(find_nearest_bucket(512, 512), (512, 512))

    def test_rounds_up(self):
        self.assertEqual(find_nearest_bucket(225, 225), (256, 256))
        self.assertEqual(find_nearest_bucket(300, 300), (336, 336))

    def test_very_large_falls_back_to_64_multiple(self):
        h, w = find_nearest_bucket(2000, 2000)
        self.assertGreaterEqual(h, 2000)
        self.assertEqual(h % 64, 0)

    def test_small_input(self):
        self.assertEqual(find_nearest_bucket(1, 1), (224, 224))


class TestBucketPixelValues4D(unittest.TestCase):
    """4D [B, C, H, W] — single-image batches."""

    def test_no_padding_needed(self):
        x = torch.randn(2, 3, 224, 224)
        self.assertTrue(torch.equal(bucket_pixel_values(x), x))

    def test_pads_to_bucket(self):
        x = torch.randn(2, 3, 200, 200)
        r = bucket_pixel_values(x)
        self.assertEqual(r.shape, (2, 3, 224, 224))

    def test_preserves_content(self):
        x = torch.randn(2, 3, 200, 200)
        r = bucket_pixel_values(x)
        self.assertTrue(torch.equal(r[:, :, :200, :200], x))
        self.assertTrue(torch.all(r[:, :, 200:, :] == 0))


class TestBucketPixelValues5D(unittest.TestCase):
    """5D [B, T, C, H, W] — video / multi-patch (Qwen3-VL)."""

    def test_preserves_temporal_dim(self):
        x = torch.randn(2, 4, 3, 200, 200)
        r = bucket_pixel_values(x)
        self.assertEqual(r.shape[1], 4, "Temporal dim T must not change")
        self.assertGreaterEqual(r.shape[3], 200)

    def test_5d_preserves_content(self):
        x = torch.randn(1, 2, 3, 300, 300)
        r = bucket_pixel_values(x)
        self.assertTrue(torch.equal(r[:, :, :, :300, :300], x))


class TestBucketPixelValues2D(unittest.TestCase):
    """2D [num_patches, hidden_dim] — Qwen3-VL flattened patch embeddings."""

    def test_passthrough(self):
        """Must NOT pad — num_patches is tied to image_grid_thw."""
        x = torch.randn(784, 1536)
        r = bucket_pixel_values(x)
        self.assertTrue(torch.equal(r, x))

    def test_passthrough_various_sizes(self):
        """Different num_patches should all pass through unchanged."""
        for n in [784, 1024, 1225, 256]:
            x = torch.randn(n, 1536)
            r = bucket_pixel_values(x)
            self.assertTrue(torch.equal(r, x))


class TestBucketEdgeCases(unittest.TestCase):
    def test_3d_passthrough(self):
        """3D also tied to grid_thw — must not pad."""
        x = torch.randn(2, 256, 768)
        self.assertTrue(torch.equal(bucket_pixel_values(x), x))


class TestBucketSequenceLength(unittest.TestCase):
    def test_pads_to_multiple_of_32(self):
        x = torch.tensor([[1, 2, 3, 4, 5]])
        padded, mask = bucket_sequence_length(x, pad_token_id=0)
        self.assertEqual(padded.shape[1] % SEQ_LENGTH_PAD_TO, 0)

    def test_preserves_content(self):
        x = torch.tensor([[10, 20, 30]])
        padded, mask = bucket_sequence_length(x, pad_token_id=99)
        self.assertTrue(torch.equal(padded[0, :3], x[0]))
        self.assertTrue(torch.all(padded[0, 3:] == 99))

    def test_attention_mask(self):
        x = torch.tensor([[1, 2, 3, 4, 5]])
        _, mask = bucket_sequence_length(x, pad_token_id=0)
        self.assertTrue(torch.all(mask[0, :5] == 1))
        self.assertTrue(torch.all(mask[0, 5:] == 0))


# ============================================================================
# Fused Loss Tests
# ============================================================================

class TestFusedCrossEntropy(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.B, self.T, self.D, self.V = 2, 8, 32, 128
        self.hidden = torch.randn(self.B, self.T, self.D)
        self.weight = torch.randn(self.V, self.D)
        self.labels = torch.randint(0, self.V, (self.B, self.T))
        self.labels[0, 0] = -100  # some ignored

    def _standard_ce(self):
        logits = F.linear(self.hidden, self.weight)
        return F.cross_entropy(logits.reshape(-1, self.V), self.labels.reshape(-1), ignore_index=-100)

    def test_matches_standard_ce(self):
        fused = mps_fused_cross_entropy(self.hidden, self.weight, self.labels, chunk_size=16)
        standard = self._standard_ce()
        self.assertTrue(torch.allclose(fused, standard, atol=1e-4),
                        f"fused={fused.item():.6f} vs standard={standard.item():.6f}")

    def test_fallback_for_small_vocab(self):
        fused = mps_fused_cross_entropy(self.hidden, self.weight, self.labels, chunk_size=256)
        standard = self._standard_ce()
        self.assertTrue(torch.allclose(fused, standard, atol=1e-5))

    def test_all_ignored_labels(self):
        labels = torch.full((self.B, self.T), -100)
        loss = mps_fused_cross_entropy(self.hidden, self.weight, labels, chunk_size=16)
        self.assertTrue(torch.isfinite(loss))

    def test_consistent_across_chunk_sizes(self):
        losses = [mps_fused_cross_entropy(self.hidden, self.weight, self.labels, chunk_size=cs).item()
                  for cs in [8, 16, 32, 64, 128]]
        for l in losses[1:]:
            self.assertAlmostEqual(losses[0], l, places=3)

    def test_default_chunk_size(self):
        self.assertEqual(DEFAULT_CHUNK_SIZE, 8192)

    def test_with_bias(self):
        bias = torch.randn(self.V)
        loss = mps_fused_cross_entropy(self.hidden, self.weight, self.labels, lm_head_bias=bias, chunk_size=16)
        self.assertTrue(torch.isfinite(loss))


class TestKLDivLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.q = torch.randn(4, 10)
        self.p = torch.randn(4, 10)

    def test_nonnegative(self):
        self.assertGreaterEqual(kl_div_loss(self.q, self.p).item(), -1e-6)

    def test_zero_for_same(self):
        self.assertAlmostEqual(kl_div_loss(self.q, self.q).item(), 0.0, places=5)

    def test_matches_pytorch(self):
        expected = F.kl_div(F.log_softmax(self.q, -1), F.log_softmax(self.p, -1), log_target=True, reduction="batchmean")
        actual = kl_div_loss(self.q, self.p)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_3d(self):
        self.assertTrue(torch.isfinite(kl_div_loss(torch.randn(2, 5, 10), torch.randn(2, 5, 10))))


class TestJSDivLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.q = torch.randn(4, 10)
        self.p = torch.randn(4, 10)

    def test_nonnegative(self):
        self.assertGreaterEqual(js_div_loss(self.q, self.p).item(), -1e-6)

    def test_zero_for_same(self):
        self.assertAlmostEqual(js_div_loss(self.q, self.q).item(), 0.0, places=4)

    def test_symmetric(self):
        js_qp = js_div_loss(self.q, self.p, beta=0.5).item()
        js_pq = js_div_loss(self.p, self.q, beta=0.5).item()
        self.assertAlmostEqual(js_qp, js_pq, places=5)

    def test_bounded_by_kl(self):
        js = js_div_loss(self.q, self.p).item()
        kl = kl_div_loss(self.q, self.p).item()
        self.assertLessEqual(js, kl + 1e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
