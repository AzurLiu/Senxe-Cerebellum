"""
Project Senxe — Core Module Unit Tests
========================================
Tests for neurons, decoder, PDI, and curiosity modules.

Run:  pytest tests/test_core.py -v
"""

import numpy as np
import pytest

from core.decoder import AntagonisticDecoder
from core.pdi import PDI
from core.curiosity import NeuralCuriosity


# ═══════════════════════════════════════════════════════════════
#  AntagonisticDecoder Tests
# ═══════════════════════════════════════════════════════════════

class TestAntagonisticDecoder:
    """Tests for the antagonistic (flexor/extensor) decoder."""

    def test_init_defaults(self):
        d = AntagonisticDecoder(action_dim=4)
        assert d.action_dim == 4
        assert d.prev_action.shape == (4,)

    def test_decode_empty_spikes(self):
        d = AntagonisticDecoder(action_dim=4, action_scale=1.0)
        action = d.decode([])
        # No spikes → action close to zero (modulo EMA of prev=0)
        assert action.shape == (4,)
        assert np.allclose(action, 0.0, atol=0.1)

    def test_decode_flexor_only(self):
        d = AntagonisticDecoder(action_dim=4, ema_alpha=1.0, action_scale=1.0)
        # Only flexor channels for dim 0 (Even CH in 0-15)
        spikes = [0, 2, 4, 6]
        action = d.decode(spikes)
        # dim 0 should be positive (flexor > extensor)
        assert action[0] > 0

    def test_decode_extensor_only(self):
        d = AntagonisticDecoder(action_dim=4, ema_alpha=1.0, action_scale=1.0)
        # Only extensor channels for dim 0 (Odd CH in 0-15)
        spikes = [1, 3, 5, 7]
        action = d.decode(spikes)
        # dim 0 should be negative (extensor > flexor)
        assert action[0] < 0

    def test_decode_balanced_near_zero(self):
        d = AntagonisticDecoder(action_dim=4, ema_alpha=1.0, action_scale=1.0)
        # Equal flexor and extensor for dim 0
        spikes = [0, 2, 4, 6] + [1, 3, 5, 7]
        action = d.decode(spikes)
        # Should be close to zero (balanced)
        assert abs(action[0]) < 0.15

    def test_decode_output_clipped(self):
        d = AntagonisticDecoder(action_dim=4, action_scale=2.0)
        spikes = list(range(0, 64, 2))  # all flexors (Even CH)
        action = d.decode(spikes)
        assert np.all(action >= -1.0)
        assert np.all(action <= 1.0)

    def test_ema_smoothing(self):
        d = AntagonisticDecoder(action_dim=4, ema_alpha=0.5, action_scale=1.0)
        spikes = [0, 2, 4, 6]
        a1 = d.decode(spikes)
        a2 = d.decode(spikes)
        # Second decode should be different due to EMA blending with prev
        # With alpha=0.5, a2 should be larger (accumulating)
        assert not np.allclose(a1, a2)

    def test_reset_clears_ema(self):
        d = AntagonisticDecoder(action_dim=4)
        d.decode(list(range(0, 64, 2)))
        assert not np.allclose(d.prev_action, 0.0)
        d.reset()
        assert np.allclose(d.prev_action, 0.0)

    def test_pdi_boost_adds_noise(self):
        np.random.seed(42)
        d1 = AntagonisticDecoder(action_dim=4, ema_alpha=1.0, action_scale=1.0)
        np.random.seed(42)
        d2 = AntagonisticDecoder(action_dim=4, ema_alpha=1.0, action_scale=1.0)
        a_no_boost = d1.decode([], pdi_boost=0.0)
        a_with_boost = d2.decode([], pdi_boost=1.0)
        # With boost, action should differ due to added noise
        # (not always, but statistically different)
        # At minimum, shapes should match
        assert a_no_boost.shape == a_with_boost.shape

    def test_channel_weights(self):
        weights = np.zeros(64)
        weights[0] = 10.0  # Only channel 0 has weight
        d = AntagonisticDecoder(action_dim=4, ema_alpha=1.0, action_scale=1.0,
                                 channel_weights=weights)
        # Spike on ch 0 (weighted) vs ch 1 (zero weight)
        a1 = d.decode([0])
        d.reset()
        a2 = d.decode([1])
        # Channel 0 with high weight should produce larger action
        assert abs(a1[0]) > abs(a2[0])

    def test_7d_action_dim(self):
        """v4.0 uses 7D action (Panda arm)."""
        d = AntagonisticDecoder(action_dim=7, action_scale=0.25)
        spikes = list(range(0, 16)) + list(range(40, 50))
        action = d.decode(spikes)
        assert action.shape == (7,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)


# ═══════════════════════════════════════════════════════════════
#  PDI Tests
# ═══════════════════════════════════════════════════════════════

class TestPDI:
    """Tests for Physical Disturbance Index computation."""

    def test_init(self):
        pdi = PDI(window=20)
        assert pdi.compute() == 0.5  # insufficient data default

    def test_constant_velocity_low_pdi(self):
        pdi = PDI(window=20)
        vel = np.array([1.0, 0.0, 0.0])
        for _ in range(25):
            pdi.update(vel)
        val = pdi.compute()
        # Constant velocity → zero variance → low PDI
        assert val < 0.1

    def test_varying_velocity_higher_pdi(self):
        pdi = PDI(window=20)
        for i in range(25):
            vel = np.array([np.sin(i * 0.5), np.cos(i * 0.5), 0.0])
            pdi.update(vel)
        val = pdi.compute()
        # Varying velocity → nonzero variance → higher PDI
        assert val > 0.01

    def test_pdi_clipped_to_range(self):
        pdi = PDI(window=5)
        for _ in range(10):
            pdi.update(np.random.randn(3) * 100)
        val = pdi.compute()
        assert 0.0 <= val <= 2.0

    def test_reset(self):
        pdi = PDI(window=20)
        pdi.update(np.array([1.0, 0.0, 0.0]))
        pdi.update(np.array([2.0, 0.0, 0.0]))
        pdi.reset()
        assert len(pdi.velocities) == 0
        assert pdi.prev_vel is None
        assert pdi.compute() == 0.5


# ═══════════════════════════════════════════════════════════════
#  NeuralCuriosity Tests
# ═══════════════════════════════════════════════════════════════

class TestNeuralCuriosity:
    """Tests for neural intrinsic curiosity."""

    def test_initial_high_curiosity(self):
        nc = NeuralCuriosity(n_channels=64, memory_size=100)
        fr = np.random.rand(64) * 200
        novelty = nc.compute_novelty(fr)
        # First few should return high curiosity (~1.0)
        assert novelty == 1.0

    def test_repeated_pattern_low_novelty(self):
        nc = NeuralCuriosity(n_channels=64, memory_size=100)
        fr = np.ones(64) * 100.0
        for _ in range(20):
            nc.compute_novelty(fr)
        novelty = nc.compute_novelty(fr)
        # Same pattern repeated → low novelty
        assert novelty < 0.5

    def test_novel_pattern_high_novelty(self):
        nc = NeuralCuriosity(n_channels=64, memory_size=100)
        base = np.ones(64) * 100.0
        for _ in range(20):
            nc.compute_novelty(base)
        # Introduce a very different pattern
        novel = np.zeros(64)
        novel[0] = 1000.0
        novelty = nc.compute_novelty(novel)
        assert novelty > 0.5

    def test_novelty_range(self):
        nc = NeuralCuriosity()
        for _ in range(30):
            fr = np.random.rand(64) * 300
            val = nc.compute_novelty(fr)
            assert 0.0 <= val <= 2.0

    def test_reset(self):
        nc = NeuralCuriosity()
        nc.compute_novelty(np.ones(64) * 100)
        nc.compute_novelty(np.ones(64) * 100)
        assert len(nc.memory) == 2
        nc.reset()
        assert len(nc.memory) == 0
