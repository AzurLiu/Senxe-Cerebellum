"""Stimulus-responsive CL simulator data source.

The official simulator's default spontaneous activity is useful for plumbing
tests but cannot validate a closed-loop stimulation/response path.  This source
adds deterministic, seeded evoked responses, explicit stimulation artifact,
fatigue and a small optional feedback-dependent gain change.  It is a systems
test model, not a claim about biological learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class StimResponsiveConfig:
    channel_count: int = 64
    frames_per_second: int = 25_000
    seed: int = 42
    baseline_rate_hz: float = 0.5
    evoked_gain_hz: float = 28.0
    response_delay_ms: float = 8.0
    response_tau_ms: float = 24.0
    artifact_duration_ms: float = 3.0
    artifact_rate_hz: float = 400.0
    fatigue_per_stim: float = 0.08
    fatigue_tau_ms: float = 300.0
    plasticity_rate: float = 0.03
    positive_feedback_channels: tuple[int, ...] = ()
    negative_feedback_channels: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.channel_count <= 0:
            raise ValueError("channel_count must be positive")
        if self.frames_per_second != 25_000:
            raise ValueError("CL simulator data sources must use 25,000 Hz")
        non_negative = (
            self.baseline_rate_hz,
            self.evoked_gain_hz,
            self.response_delay_ms,
            self.response_tau_ms,
            self.artifact_duration_ms,
            self.artifact_rate_hz,
            self.fatigue_per_stim,
            self.fatigue_tau_ms,
            self.plasticity_rate,
        )
        if any(value < 0 for value in non_negative):
            raise ValueError("response model values must be non-negative")
        if self.response_tau_ms == 0 or self.fatigue_tau_ms == 0:
            raise ValueError("time constants must be positive")


@dataclass(frozen=True)
class ModelSpike:
    timestamp: int
    channel: int


@dataclass(frozen=True)
class ModelBatch:
    frames: np.ndarray
    spikes: tuple[ModelSpike, ...]


@dataclass(frozen=True)
class _StimEvent:
    timestamp: int
    channel: int
    strength: float


class StimResponsiveModel:
    """Seeded sequential generator with stimulus-dependent spike probability."""

    def __init__(self, config: StimResponsiveConfig | None = None) -> None:
        self.config = config or StimResponsiveConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self._cursor: int | None = None
        self._stim_events: list[_StimEvent] = []
        self._last_sensory_stim: _StimEvent | None = None
        self._feedback_gain = 1.0
        self._coupling = self._make_coupling()

    @property
    def feedback_gain(self) -> float:
        return float(self._feedback_gain)

    def _make_coupling(self) -> np.ndarray:
        coupling = self.rng.uniform(
            0.05,
            0.35,
            size=(self.config.channel_count, self.config.channel_count),
        )
        for channel in range(self.config.channel_count):
            coupling[channel, channel] = 1.0
            coupling[channel, (channel + 1) % self.config.channel_count] = 0.7
        return coupling

    def on_stim(
        self,
        *,
        timestamp: int,
        channel: int,
        strength: float = 1.0,
    ) -> None:
        if not 0 <= channel < self.config.channel_count:
            return
        if channel in self.config.positive_feedback_channels:
            self.apply_feedback(+1.0)
            return
        if channel in self.config.negative_feedback_channels:
            self.apply_feedback(-1.0)
            return
        fatigue_tau_frames = (
            self.config.fatigue_tau_ms
            * self.config.frames_per_second
            / 1000.0
        )
        fatigue_load = sum(
            np.exp(
                -max(0, int(timestamp) - prior.timestamp)
                / fatigue_tau_frames
            )
            for prior in self._stim_events
            if prior.timestamp <= int(timestamp)
        )
        effective_strength = (
            max(0.05, abs(float(strength)))
            * np.exp(-self.config.fatigue_per_stim * fatigue_load)
        )
        event = _StimEvent(
            timestamp=int(timestamp),
            channel=int(channel),
            strength=float(effective_strength),
        )
        self._stim_events.append(event)
        self._last_sensory_stim = event

    def apply_feedback(self, value: float) -> None:
        """Adjust evoked gain slightly for protocol plumbing tests."""

        delta = np.clip(float(value), -1.0, 1.0) * self.config.plasticity_rate
        self._feedback_gain = float(np.clip(
            self._feedback_gain + delta,
            0.5,
            1.5,
        ))

    def read(self, from_timestamp: int, frame_count: int) -> ModelBatch:
        start = int(from_timestamp)
        frame_count = int(frame_count)
        if frame_count < 0:
            raise ValueError("frame_count must be non-negative")
        if self._cursor is not None and start != self._cursor:
            raise ValueError(
                "StimResponsiveModel is sequential: from_timestamp must "
                f"equal {self._cursor}, got {start}"
            )
        self._cursor = start + frame_count
        if frame_count == 0:
            return ModelBatch(
                frames=np.zeros(
                    (0, self.config.channel_count),
                    dtype=np.int16,
                ),
                spikes=(),
            )

        timestamps = start + np.arange(frame_count, dtype=np.int64)
        rates = np.full(
            (frame_count, self.config.channel_count),
            self.config.baseline_rate_hz,
            dtype=np.float64,
        )
        fps_per_ms = self.config.frames_per_second / 1000.0

        for event in self._stim_events:
            dt_ms = (timestamps - event.timestamp) / fps_per_ms
            artifact_mask = (
                (dt_ms >= 0.0)
                & (dt_ms < self.config.artifact_duration_ms)
            )
            if np.any(artifact_mask):
                rates[artifact_mask, event.channel] += (
                    self.config.artifact_rate_hz * event.strength
                )

            response_age = dt_ms - self.config.response_delay_ms
            response_mask = response_age >= 0.0
            if np.any(response_mask):
                temporal = np.exp(
                    -response_age[response_mask]
                    / self.config.response_tau_ms
                )
                spatial = self._coupling[event.channel]
                rates[response_mask] += (
                    temporal[:, None]
                    * spatial[None, :]
                    * self.config.evoked_gain_hz
                    * event.strength
                    * self._feedback_gain
                )

        probabilities = 1.0 - np.exp(
            -rates / self.config.frames_per_second
        )
        hits = self.rng.random(probabilities.shape) < probabilities
        frame_indices, channels = np.nonzero(hits)
        spikes = tuple(
            ModelSpike(
                timestamp=int(start + frame_index),
                channel=int(channel),
            )
            for frame_index, channel in zip(frame_indices, channels)
        )

        retention_frames = int(
            self.config.frames_per_second
            * max(
                self.config.response_tau_ms * 8.0,
                self.config.fatigue_tau_ms,
            )
            / 1000.0
        )
        cutoff = self._cursor - retention_frames
        self._stim_events = [
            event for event in self._stim_events
            if event.timestamp >= cutoff
        ]
        return ModelBatch(
            frames=np.zeros(
                (frame_count, self.config.channel_count),
                dtype=np.int16,
            ),
            spikes=spikes,
        )


def _stim_strength(stim: Any) -> float:
    currents = tuple(getattr(stim, "phase_currents_uA", ()) or ())
    if not currents:
        return 1.0
    return max(0.05, max(abs(float(current)) for current in currents))


try:
    from cl.sim import (  # type: ignore[attr-defined]
        DataSourceBatch,
        DataSourceSpike,
        SimulatorDataSource,
        SimulatorDataSourceMetadata,
    )
except (ImportError, AttributeError):
    DataSourceBatch = None
    DataSourceSpike = None
    SimulatorDataSource = object
    SimulatorDataSourceMetadata = None


class StimResponsiveSimulatorDataSource(SimulatorDataSource):
    """Adapter for the official CL SDK custom simulator source API."""

    def __init__(self, **config: Any) -> None:
        if SimulatorDataSourceMetadata is None:
            raise RuntimeError(
                "This installed cl-sdk does not expose cl.sim custom data "
                "sources. Upgrade to a release that provides cl.sim."
            )
        self.config = StimResponsiveConfig(**config)
        self.model = StimResponsiveModel(self.config)

    @property
    def metadata(self) -> Any:
        return SimulatorDataSourceMetadata(
            channel_count=self.config.channel_count,
            frames_per_second=self.config.frames_per_second,
            uV_per_sample_unit=0.195,
            start_timestamp=0,
            duration_frames=None,
            seekable=False,
            realtime_only=False,
            supports_accelerated=True,
        )

    def open(self) -> None:
        return None

    def close(self) -> None:
        return None

    def on_stim(self, stim: Any) -> None:
        self.model.on_stim(
            timestamp=int(stim.timestamp),
            channel=int(stim.channel),
            strength=_stim_strength(stim),
        )

    def on_stims(self, stims: Sequence[Any]) -> None:
        for stim in stims:
            self.on_stim(stim)

    def read(self, from_timestamp: int, frame_count: int) -> Any:
        batch = self.model.read(from_timestamp, frame_count)
        return DataSourceBatch(
            frames=batch.frames,
            spikes=tuple(
                DataSourceSpike(
                    timestamp=spike.timestamp,
                    channel=spike.channel,
                )
                for spike in batch.spikes
            ),
        )


def create_stim_responsive_source(**config: Any) -> Any:
    """Importable factory for CL simulator configuration."""

    return StimResponsiveSimulatorDataSource(**config)
