"""
Senxe Cerebellum — Antagonistic Decoding Module
=============================================
Biological flexor/extensor antagonistic motor control decoder.

This module implements the core motor output pathway of the Senxe Cerebellum framework.
Inspired by the antagonistic muscle pairs in vertebrate motor systems
(e.g., biceps/triceps), it converts raw neural spike channels into smooth,
continuous action vectors suitable for robotic control.

The confirmatory contact-skill path supplies explicit, disjoint positive and
negative channel groups. Legacy modes retain the historical 64-channel
even/odd mapping:

    Flexor   (Even CH: 0, 2, 4...) → positive force per action dimension
    Extensor (Odd CH: 1, 3, 5...)  → negative force per action dimension

    Action[i] = (flexor_count − extensor_count) / (total + ε)

The differential signal naturally produces:
    - Balanced activity → near-zero output (co-contraction / stability)
    - Flexor dominance  → positive action (extension movement)
    - Extensor dominance → negative action (flexion movement)

An EMA (Exponential Moving Average) low-pass filter smooths the output,
mimicking the mechanical inertia of biological muscle tissue and producing
jerk-free trajectories suitable for force-sensitive industrial tasks.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


class AntagonisticDecoder:
    """Antagonistic Decoder — converts spike channels to smoothed action vectors.

    Maps spiking activity from a 64-channel MEA onto an N-dimensional action
    space using positive/negative antagonistic populations. Confirmatory paths
    pass fixed channel groups; omitted groups select the historical even/odd
    whole-array mapping for legacy comparisons.

    The decoder supports optional per-channel weighting from calibration data,
    allowing more responsive channels to contribute proportionally more to
    the population vector — analogous to how motor cortex neurons with
    stronger corticospinal projections dominate movement commands.

    Args:
        action_dim: Number of action dimensions (4 for Fetch, 7 for Panda).
        ema_alpha: EMA smoothing coefficient in [0, 1].
                   Higher values → more responsive (less smoothing).
                   Lower values → smoother trajectories (more inertia).
        action_scale: Multiplicative scaling factor applied after decoding.
        channel_weights: Optional (64,) array of per-channel weights from
                         warm-up calibration. If provided, each spike
                         contributes its channel's weight instead of 1.0.
        channel_groups: Optional explicit positive/negative channel groups,
                        one pair of populations per action dimension.
    """

    def __init__(
        self,
        action_dim: int = 4,
        ema_alpha: float = 0.35,
        action_scale: float = 0.25,
        channel_weights: Optional[np.ndarray] = None,
        channel_groups: Optional[
            Sequence[tuple[Sequence[int], Sequence[int]]]
        ] = None,
    ) -> None:
        if action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if not 0.0 <= ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in [0, 1]")
        self.action_dim: int = action_dim
        self.group_size: int = max(1, 32 // action_dim)
        self.ema_alpha: float = ema_alpha
        self.action_scale: float = action_scale
        self.prev_action: np.ndarray = np.zeros(action_dim)
        self.channel_groups = _normalize_channel_groups(
            action_dim,
            channel_groups,
        )

        # Population vector weights from calibration responsiveness
        if channel_weights is not None:
            self.ch_weights = np.array(channel_weights, dtype=np.float64)
            if self.ch_weights.shape != (64,):
                raise ValueError("channel_weights must have shape (64,)")
            self.ch_weights = np.where(
                np.isfinite(self.ch_weights),
                self.ch_weights,
                0.0,
            )
            self.ch_weights = np.maximum(
                self.ch_weights,
                0.0,
            )
            self.ch_weights = self.ch_weights / (self.ch_weights.max() + 1e-6)
        else:
            self.ch_weights = np.ones(64)

    def decode(
        self,
        spike_channels: List[int],
        pdi_boost: float = 0.0,
    ) -> np.ndarray:
        """Decode active spike channels into a smoothed action vector.

        For each action dimension i, the decoder computes:

            flexor_sum  = Σ weight[ch] for ch in [i*G, (i+1)*G)      (CH 0–31)
            extensor_sum = Σ weight[ch] for ch in [32+i*G, 32+(i+1)*G) (CH 32–63)
            raw_action[i] = (flexor_sum − extensor_sum) / (flexor_sum + extensor_sum + ε)

        where G = 32 // action_dim is the group size per dimension.

        The raw action is then:
            1. Perturbed by FEP exploration noise (if PDI is high).
            2. Smoothed via EMA: action = α·raw + (1−α)·prev_action.
            3. Scaled and clipped to [-1, 1].

        Args:
            spike_channels: List of active (spiking) channel indices [0–63].
            pdi_boost: FEP exploration noise magnitude (typically PDI × 0.4).
                       When PDI is high (unstable state), additional Gaussian
                       noise is injected to promote exploration.

        Returns:
            np.ndarray: Action vector of shape (action_dim,), clipped to [-1, 1].
        """
        channel_counts = np.bincount(
            np.asarray(spike_channels, dtype=np.int64),
            minlength=64,
        )[:64]
        return self.decode_counts(channel_counts, pdi_boost=pdi_boost)

    def decode_counts(
        self,
        channel_counts: np.ndarray,
        pdi_boost: float = 0.0,
    ) -> np.ndarray:
        """Decode per-channel spike counts without discarding temporal density.

        ``decode`` remains the compatibility entry point.  The timestamped CL1
        pipeline calls this method so repeated spikes on one electrode retain
        their contribution rather than being collapsed to channel presence.
        """

        counts = np.asarray(channel_counts, dtype=np.float64)
        if counts.shape != (64,):
            raise ValueError("channel_counts must have shape (64,)")
        counts = np.maximum(counts, 0.0)
        action = np.zeros(self.action_dim)

        if self.channel_groups is not None:
            for index, (positive_channels, negative_channels) in enumerate(
                self.channel_groups
            ):
                positive = float(np.sum(
                    counts[list(positive_channels)]
                    * self.ch_weights[list(positive_channels)]
                ))
                negative = float(np.sum(
                    counts[list(negative_channels)]
                    * self.ch_weights[list(negative_channels)]
                ))
                action[index] = (
                    (positive - negative)
                    / (positive + negative + 1e-6)
                )
            return self._smooth_and_scale(action, pdi_boost)

        for i in range(self.action_dim):
            # Legacy path: distribute all 32 channel pairs across action_dim.
            base = 32 // self.action_dim
            rem = 32 % self.action_dim
            p_lo = i * base + min(i, rem)
            p_hi = p_lo + base + (1 if i < rem else 0)

            flex = 0.0
            ext = 0.0
            for p in range(p_lo, p_hi):
                ch_f = 2 * p      # Even channel -> Flexor
                ch_e = 2 * p + 1  # Odd channel -> Extensor
                flex += counts[ch_f] * self.ch_weights[ch_f]
                ext += counts[ch_e] * self.ch_weights[ch_e]

            action[i] = (flex - ext) / (flex + ext + 1e-6)

        return self._smooth_and_scale(action, pdi_boost)

    def _smooth_and_scale(
        self,
        action: np.ndarray,
        pdi_boost: float,
    ) -> np.ndarray:
        # FEP: high PDI (unstable state) → inject exploration noise
        if pdi_boost > 0.1:
            action += np.random.randn(self.action_dim) * pdi_boost * 0.25

        # EMA smoothing → bio-muscle-like smooth movement trajectory
        action = self.ema_alpha * action + (1 - self.ema_alpha) * self.prev_action
        self.prev_action = action.copy()

        return np.clip(action * self.action_scale, -1.0, 1.0)

    def reset(self) -> None:
        """Reset EMA state for a new episode."""
        self.prev_action = np.zeros(self.action_dim)


def _normalize_channel_groups(
    action_dim: int,
    channel_groups: Optional[
        Sequence[tuple[Sequence[int], Sequence[int]]]
    ],
) -> Optional[
    tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
]:
    if channel_groups is None:
        return None
    if len(channel_groups) != action_dim:
        raise ValueError("channel_groups length must match action_dim")

    normalized = []
    used: set[int] = set()
    for index, (positive, negative) in enumerate(channel_groups):
        positive_channels = tuple(int(channel) for channel in positive)
        negative_channels = tuple(int(channel) for channel in negative)
        if not positive_channels or not negative_channels:
            raise ValueError(
                f"channel group {index} must have both signs"
            )
        group_channels = positive_channels + negative_channels
        if any(channel < 0 or channel >= 64 for channel in group_channels):
            raise ValueError(
                f"channel group {index} contains a channel outside 0..63"
            )
        if len(group_channels) != len(set(group_channels)):
            raise ValueError(
                f"channel group {index} contains duplicate channels"
            )
        overlap = used & set(group_channels)
        if overlap:
            raise ValueError(
                "channel_groups must be disjoint; overlapping channels: "
                f"{sorted(overlap)}"
            )
        used.update(group_channels)
        normalized.append((positive_channels, negative_channels))
    return tuple(normalized)
