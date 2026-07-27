"""Timestamp-preserving spike acquisition for CL1 closed-loop experiments.

The default path consumes spike detections produced by the CL SDK.  It never
reconstructs timestamps from an aggregate firing-rate vector and never treats
large raw sample values as biological spikes.  A configurable post-stimulation
artifact interval is observed but excluded from decoder features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import ceil
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class SpikeWindowConfig:
    """Timing and feature settings for one closed-loop response window."""

    channel_count: int = 64
    artifact_wait_ms: float = 50.0
    collect_window_ms: float = 50.0
    bin_width_ms: float = 10.0
    tick_ms: float = 10.0

    def __post_init__(self) -> None:
        if self.channel_count <= 0:
            raise ValueError("channel_count must be positive")
        if self.artifact_wait_ms < 0:
            raise ValueError("artifact_wait_ms must be non-negative")
        if self.collect_window_ms <= 0:
            raise ValueError("collect_window_ms must be positive")
        if self.bin_width_ms <= 0 or self.tick_ms <= 0:
            raise ValueError("bin_width_ms and tick_ms must be positive")


@dataclass(frozen=True)
class SpikeEvent:
    """One detected spike with its original CL frame timestamp."""

    timestamp: int
    channel: int
    latency_ms: float


@dataclass(frozen=True)
class SpikeFeatures:
    """Decoder-ready temporal features derived from a response window."""

    channel_counts: tuple[int, ...]
    time_bins: tuple[tuple[int, ...], ...]
    first_latency_ms: tuple[float | None, ...]
    total_spikes: int
    active_channels: tuple[int, ...]

    def firing_rates_hz(self, collect_window_ms: float) -> np.ndarray:
        seconds = max(float(collect_window_ms) / 1000.0, 1e-9)
        return np.asarray(self.channel_counts, dtype=np.float64) / seconds


@dataclass(frozen=True)
class SpikeWindow:
    """One artifact-separated neural response window."""

    frame_rate_hz: int
    artifact_start_timestamp: int | None
    collect_start_timestamp: int | None
    collect_end_timestamp: int | None
    artifact_spike_count: int
    stim_timestamps: tuple[int, ...]
    events: tuple[SpikeEvent, ...]
    features: SpikeFeatures

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SpikeWindowReader:
    """Read SDK-detected spikes using real CL timestamps.

    The implementation uses ``Neurons.loop`` because it is available in both
    the currently installed SDK and the hardware-facing API.  Each tick already
    contains the SDK's spike and stimulation detections.
    """

    def __init__(
        self,
        neurons: Any,
        config: SpikeWindowConfig | None = None,
    ) -> None:
        self.neurons = neurons
        self.config = config or SpikeWindowConfig()
        self.frame_rate_hz = int(neurons.get_frames_per_second())
        if self.frame_rate_hz <= 0:
            raise ValueError("neurons.get_frames_per_second() must be positive")

        ticks_per_second = int(round(1000.0 / self.config.tick_ms))
        if ticks_per_second <= 0:
            raise ValueError("tick_ms produces an invalid loop rate")
        self.ticks_per_second = ticks_per_second
        self._artifact_ticks = int(ceil(
            self.config.artifact_wait_ms / self.config.tick_ms
        ))
        self._collect_ticks = int(ceil(
            self.config.collect_window_ms / self.config.tick_ms
        ))

    def read(self) -> SpikeWindow:
        total_ticks = self._artifact_ticks + self._collect_ticks
        loop = self.neurons.loop(
            ticks_per_second=self.ticks_per_second,
            stop_after_ticks=total_ticks,
            ignore_jitter=True,
        )

        artifact_start: int | None = None
        collect_start: int | None = None
        collect_end: int | None = None
        artifact_spike_count = 0
        events_raw: list[tuple[int, int]] = []
        stim_timestamps: list[int] = []

        for tick_index, tick in enumerate(loop):
            tick_timestamp = int(tick.timestamp)
            frames = np.asarray(tick.frames)
            frame_count = int(frames.shape[0]) if frames.ndim >= 1 else 0
            analysis = getattr(tick, "analysis", None)
            spikes = tuple(getattr(analysis, "spikes", ()) or ())
            stims = tuple(getattr(analysis, "stims", ()) or ())

            if artifact_start is None:
                artifact_start = tick_timestamp
            stim_timestamps.extend(
                int(getattr(stim, "timestamp")) for stim in stims
            )

            if tick_index < self._artifact_ticks:
                artifact_spike_count += len(spikes)
                continue

            if collect_start is None:
                collect_start = tick_timestamp
            collect_end = tick_timestamp + frame_count
            for spike in spikes:
                timestamp = int(getattr(spike, "timestamp"))
                channel = int(getattr(spike, "channel"))
                if 0 <= channel < self.config.channel_count:
                    events_raw.append((timestamp, channel))

        features, events = _build_features(
            events_raw,
            collect_start=collect_start,
            frame_rate_hz=self.frame_rate_hz,
            config=self.config,
        )
        return SpikeWindow(
            frame_rate_hz=self.frame_rate_hz,
            artifact_start_timestamp=artifact_start,
            collect_start_timestamp=collect_start,
            collect_end_timestamp=collect_end,
            artifact_spike_count=artifact_spike_count,
            stim_timestamps=tuple(stim_timestamps),
            events=events,
            features=features,
        )


def _build_features(
    events_raw: Iterable[tuple[int, int]],
    *,
    collect_start: int | None,
    frame_rate_hz: int,
    config: SpikeWindowConfig,
) -> tuple[SpikeFeatures, tuple[SpikeEvent, ...]]:
    bin_count = max(1, int(ceil(
        config.collect_window_ms / config.bin_width_ms
    )))
    channel_counts = np.zeros(config.channel_count, dtype=np.int64)
    time_bins = np.zeros((bin_count, config.channel_count), dtype=np.int64)
    first_latency: list[float | None] = [None] * config.channel_count
    events: list[SpikeEvent] = []
    start = int(collect_start or 0)
    frames_per_ms = frame_rate_hz / 1000.0

    for timestamp, channel in sorted(events_raw):
        latency_ms = max(0.0, (int(timestamp) - start) / frames_per_ms)
        if latency_ms >= config.collect_window_ms:
            continue
        bin_index = min(
            bin_count - 1,
            int(latency_ms // config.bin_width_ms),
        )
        channel_counts[channel] += 1
        time_bins[bin_index, channel] += 1
        if first_latency[channel] is None:
            first_latency[channel] = latency_ms
        events.append(SpikeEvent(
            timestamp=int(timestamp),
            channel=int(channel),
            latency_ms=float(latency_ms),
        ))

    active = tuple(np.flatnonzero(channel_counts).astype(int).tolist())
    features = SpikeFeatures(
        channel_counts=tuple(channel_counts.astype(int).tolist()),
        time_bins=tuple(tuple(row.astype(int).tolist()) for row in time_bins),
        first_latency_ms=tuple(first_latency),
        total_spikes=int(channel_counts.sum()),
        active_channels=active,
    )
    return features, tuple(events)


def ablate_channel_counts(
    channel_counts: Sequence[int],
    mode: str,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a count-preserving spike ablation for causal controls."""

    counts = np.asarray(channel_counts, dtype=np.int64)
    normalized_mode = str(mode).strip().lower()
    if normalized_mode in {"none", "full"}:
        return counts.copy()
    if normalized_mode == "zero":
        return np.zeros_like(counts)
    if normalized_mode in {"random", "shuffled"}:
        return counts[rng.permutation(len(counts))]
    raise ValueError(
        "spike ablation mode must be none, zero, random, or shuffled"
    )
