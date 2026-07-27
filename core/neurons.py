"""
Senxe Cerebellum — Neural Interface Module
========================================
CL1 biological neural interface: Pure hardware mode.

This module provides the hardware abstraction layer for interfacing with
Cortical Labs CL1 biological neural organoids on a 64-channel MEA.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import os
import numpy as np
from contextlib import contextmanager
from typing import Any, Tuple

import warnings

from core.channel_map import (
    NON_STIMULATABLE_CHANNELS,
    STIMULATABLE_CHANNELS,
)

# Compatibility shim: cl-sdk uses np.bool which was removed in numpy 2.0
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    if not hasattr(np, 'bool'):
        np.bool = np.bool_
if not hasattr(np, 'concat'):
    np.concat = np.concatenate

try:
    import cl as _cl_sdk
    from cl import ChannelSet, StimDesign, BurstDesign
    from cl.neurons import Neurons
    CL_AVAILABLE = True
except ImportError:
    raise RuntimeError(
        "FATAL ERROR: cl-sdk is not installed. \n"
        "This project requires the official Cortical Labs SDK (cl-sdk) to run. \n"
        "Please run: pip install cl-sdk"
    )

# Monkeypatch cl.neurons.Neurons.close to be idempotent
if CL_AVAILABLE:
    _original_close = Neurons.close
    def _idempotent_close(self):
        if getattr(self, "_closed", False):
            return
        _original_close(self)
        self._closed = True
    Neurons.close = _idempotent_close

def is_cl_simulator() -> bool:
    """Returns True if cl-sdk is running in simulation mode (no real CL1 hardware).
    
    Detection: The official cl-sdk ships as a mock/simulator that replays
    Poisson-sampled recordings. We detect this by checking whether the
    Neurons class has the mock-only '_replay_file' attribute in its
    annotations, which only exists in the simulator implementation.
    """
    try:
        from cl.neurons import Neurons
        return '_replay_file' in getattr(Neurons, '__annotations__', {})
    except Exception:
        return False


def require_stimulation_approval() -> None:
    """Refuse real-culture stimulation without an explicit lab approval gate."""

    if is_cl_simulator():
        return
    approved = os.getenv("SENXE_LAB_APPROVED_STIM", "").strip() == "1"
    if not approved:
        raise RuntimeError(
            "Real CL1 stimulation is blocked until the supervising laboratory "
            "sets SENXE_LAB_APPROVED_STIM=1 after reviewing pulse amplitude, "
            "phase width, frequency, channel allocation, and cumulative dose."
        )
    required_limits = (
        "SENXE_MAX_STIM_AMPLITUDE_UA",
        "SENXE_MAX_STIM_PHASE_WIDTH_US",
        "SENXE_MAX_STIM_BURST_HZ",
        "SENXE_MAX_STIM_BURST_COUNT",
        "SENXE_MAX_STIM_CALLS",
        "SENXE_MAX_STIM_CHANNEL_PULSES",
        "SENXE_MAX_STIM_ABS_CHARGE_NC",
    )
    missing = [
        name for name in required_limits if not os.getenv(name, "").strip()
    ]
    if missing:
        raise RuntimeError(
            "Real CL1 stimulation requires explicit laboratory-approved "
            "limits for: " + ", ".join(missing)
        )


@dataclass(frozen=True)
class StimSafetyLimits:
    """Laboratory-reviewed electrical and cumulative stimulation envelope."""

    max_amplitude_ua: float = 1.5
    max_phase_width_us: int = 200
    max_burst_hz: int = 200
    max_burst_count: int = 15
    max_stim_calls: int = 10_000_000
    max_channel_pulses: int = 100_000_000
    max_abs_charge_nc: float = 100_000_000.0
    max_charge_imbalance_fraction: float = 0.05

    def __post_init__(self) -> None:
        positive = (
            self.max_amplitude_ua,
            self.max_phase_width_us,
            self.max_burst_hz,
            self.max_burst_count,
            self.max_stim_calls,
            self.max_channel_pulses,
            self.max_abs_charge_nc,
        )
        if any(float(value) <= 0 for value in positive):
            raise ValueError("stimulation safety limits must be positive")
        if not 0.0 <= self.max_charge_imbalance_fraction <= 1.0:
            raise ValueError(
                "max_charge_imbalance_fraction must be in [0, 1]"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def stimulation_limits_from_env() -> StimSafetyLimits:
    """Load the explicit lab envelope, with simulator-safe defaults."""

    return StimSafetyLimits(
        max_amplitude_ua=float(os.getenv(
            "SENXE_MAX_STIM_AMPLITUDE_UA",
            "1.5",
        )),
        max_phase_width_us=int(os.getenv(
            "SENXE_MAX_STIM_PHASE_WIDTH_US",
            "200",
        )),
        max_burst_hz=int(os.getenv(
            "SENXE_MAX_STIM_BURST_HZ",
            "200",
        )),
        max_burst_count=int(os.getenv(
            "SENXE_MAX_STIM_BURST_COUNT",
            "15",
        )),
        max_stim_calls=int(os.getenv(
            "SENXE_MAX_STIM_CALLS",
            "10000000",
        )),
        max_channel_pulses=int(os.getenv(
            "SENXE_MAX_STIM_CHANNEL_PULSES",
            "100000000",
        )),
        max_abs_charge_nc=float(os.getenv(
            "SENXE_MAX_STIM_ABS_CHARGE_NC",
            "100000000",
        )),
    )


class BudgetedNeurons:
    """Fail-closed proxy for the approved electrical stimulation envelope."""

    def __init__(
        self,
        neurons,
        *,
        max_stim_calls: int | None = None,
        max_channel_pulses: int | None = None,
        safety_limits: StimSafetyLimits | None = None,
    ) -> None:
        if safety_limits is not None and (
            max_stim_calls is not None or max_channel_pulses is not None
        ):
            raise ValueError(
                "pass safety_limits or legacy count limits, not both"
            )
        if safety_limits is None:
            defaults = StimSafetyLimits()
            safety_limits = StimSafetyLimits(
                max_stim_calls=(
                    defaults.max_stim_calls
                    if max_stim_calls is None
                    else int(max_stim_calls)
                ),
                max_channel_pulses=(
                    defaults.max_channel_pulses
                    if max_channel_pulses is None
                    else int(max_channel_pulses)
                ),
            )
        self._neurons = neurons
        self.safety_limits = safety_limits
        self.max_stim_calls = int(safety_limits.max_stim_calls)
        self.max_channel_pulses = int(safety_limits.max_channel_pulses)
        self.stim_calls = 0
        self.channel_pulses = 0
        self.abs_charge_nc = 0.0
        self.last_stim_timestamp: int | None = None
        self.last_stim_event: dict[str, Any] | None = None
        self._stim_events: deque[dict[str, Any]] = deque(maxlen=4096)

    def stim(self, channels, design, burst):
        raw_channel_mask = getattr(channels, "_channels", None)
        raw_burst_count = getattr(burst, "_burst_count", None)
        raw_burst_hz = getattr(burst, "_burst_hz", None)
        if raw_burst_hz is None:
            raw_burst_hz = getattr(burst, "_burst_requested_hz", None)
        raw_design_args = getattr(design, "_args", None)
        if (
            raw_channel_mask is None
            or raw_burst_count is None
            or raw_burst_hz is None
            or raw_design_args is None
        ):
            raise RuntimeError(
                "Cannot audit stimulation dose for this CL SDK design"
            )
        channel_mask = np.asarray(
            raw_channel_mask,
            dtype=bool,
        )
        selected_channels = set(
            np.flatnonzero(channel_mask).astype(int).tolist()
        )
        invalid_channels = (
            selected_channels - set(STIMULATABLE_CHANNELS)
        )
        if invalid_channels:
            raise RuntimeError(
                "Refusing stimulation on non-stimulatable CL1 channels: "
                f"{sorted(invalid_channels)}"
            )
        channel_count = int(np.count_nonzero(channel_mask))
        burst_count = int(raw_burst_count)
        burst_hz = int(raw_burst_hz)
        if channel_count <= 0 or burst_count <= 0:
            raise RuntimeError("Invalid zero-dose stimulation design")
        if burst_count > self.safety_limits.max_burst_count:
            raise RuntimeError(
                "Laboratory-approved stimulation burst-count limit exceeded"
            )
        if burst_hz <= 0 or burst_hz > self.safety_limits.max_burst_hz:
            raise RuntimeError(
                "Laboratory-approved stimulation frequency limit exceeded"
            )

        phases = _stim_design_phases(raw_design_args)
        phase_widths = [duration_us for duration_us, _ in phases]
        amplitudes = [abs(current_ua) for _, current_ua in phases]
        if max(phase_widths) > self.safety_limits.max_phase_width_us:
            raise RuntimeError(
                "Laboratory-approved stimulation phase-width limit exceeded"
            )
        if max(amplitudes) > self.safety_limits.max_amplitude_ua:
            raise RuntimeError(
                "Laboratory-approved stimulation amplitude limit exceeded"
            )
        absolute_charge_per_pulse_nc = sum(
            duration_us * abs(current_ua)
            for duration_us, current_ua in phases
        ) / 1000.0
        signed_charge_per_pulse_nc = sum(
            duration_us * current_ua
            for duration_us, current_ua in phases
        ) / 1000.0
        imbalance_fraction = (
            abs(signed_charge_per_pulse_nc)
            / max(absolute_charge_per_pulse_nc, 1e-12)
        )
        if (
            imbalance_fraction
            > self.safety_limits.max_charge_imbalance_fraction
        ):
            raise RuntimeError(
                "Stimulation design exceeds the approved charge-balance "
                "tolerance"
            )
        proposed_calls = self.stim_calls + 1
        proposed_channel_pulses = (
            self.channel_pulses + channel_count * burst_count
        )
        event_abs_charge_nc = (
            absolute_charge_per_pulse_nc
            * channel_count
            * burst_count
        )
        proposed_abs_charge_nc = self.abs_charge_nc + event_abs_charge_nc
        if proposed_calls > self.max_stim_calls:
            raise RuntimeError(
                "Laboratory-approved stimulation call budget exhausted"
            )
        if proposed_channel_pulses > self.max_channel_pulses:
            raise RuntimeError(
                "Laboratory-approved channel-pulse budget exhausted"
            )
        if proposed_abs_charge_nc > self.safety_limits.max_abs_charge_nc:
            raise RuntimeError(
                "Laboratory-approved absolute-charge budget exhausted"
            )
        result = self._neurons.stim(channels, design, burst)
        self.stim_calls = proposed_calls
        self.channel_pulses = proposed_channel_pulses
        self.abs_charge_nc = proposed_abs_charge_nc
        self.last_stim_timestamp = _safe_neuron_timestamp(self._neurons)
        self.last_stim_event = {
            "sequence": self.stim_calls,
            "timestamp": self.last_stim_timestamp,
            "channels": sorted(selected_channels),
            "burst_count": burst_count,
            "burst_hz": burst_hz,
            "phase_widths_us": phase_widths,
            "amplitudes_ua": amplitudes,
            "absolute_charge_nc": event_abs_charge_nc,
        }
        self._stim_events.append(dict(self.last_stim_event))
        return result

    def budget_snapshot(self) -> dict[str, Any]:
        return {
            "stim_calls": self.stim_calls,
            "max_stim_calls": self.max_stim_calls,
            "channel_pulses": self.channel_pulses,
            "max_channel_pulses": self.max_channel_pulses,
            "abs_charge_nc": self.abs_charge_nc,
            "max_abs_charge_nc": (
                self.safety_limits.max_abs_charge_nc
            ),
            "last_stim_timestamp": self.last_stim_timestamp,
        }

    def stim_events_since(self, sequence: int) -> tuple[dict[str, Any], ...]:
        """Return retained audited stimulation events after ``sequence``."""

        return tuple(
            dict(event)
            for event in self._stim_events
            if int(event["sequence"]) > int(sequence)
        )

    def __getattr__(self, name):
        return getattr(self._neurons, name)


def stimulation_budget_delta(
    start: dict[str, Any],
    end: dict[str, Any],
) -> dict[str, float | int]:
    """Return per-interval dose from two cumulative budget snapshots."""

    return {
        "stim_calls": int(end["stim_calls"]) - int(start["stim_calls"]),
        "channel_pulses": (
            int(end["channel_pulses"]) - int(start["channel_pulses"])
        ),
        "abs_charge_nc": (
            float(end["abs_charge_nc"]) - float(start["abs_charge_nc"])
        ),
    }


def _stim_design_phases(
    raw_design_args: Any,
) -> tuple[tuple[int, float], ...]:
    args = tuple(raw_design_args)
    if len(args) < 2 or len(args) % 2:
        raise RuntimeError("StimDesign phases are not auditable")
    phases: list[tuple[int, float]] = []
    for index in range(0, len(args), 2):
        duration_us = int(args[index])
        current_ua = float(args[index + 1])
        if duration_us <= 0 or not np.isfinite(current_ua):
            raise RuntimeError("StimDesign contains an invalid phase")
        phases.append((duration_us, current_ua))
    if not any(abs(current) > 0 for _, current in phases):
        raise RuntimeError("StimDesign has zero current on every phase")
    return tuple(phases)


def _safe_neuron_timestamp(neurons: Any) -> int | None:
    timestamp = getattr(neurons, "timestamp", None)
    if not callable(timestamp):
        return None
    try:
        return int(timestamp())
    except (RuntimeError, TypeError, ValueError):
        return None

@contextmanager
def cl_open():
    """Unified CL1 entry point: connects to real Cortical Labs biological hardware."""
    with _cl_sdk.open() as neurons:
        yield neurons

def warmup_calibration(
    neurons,
    duration_sec: float = 10.0,
    *,
    max_stim_amplitude: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Channel warm-up calibration over the 59 stimulatable channels.

    Phase 1 (Baseline): Read spontaneous activity.
    Phase 2 (Probing): Deliver a standard biphasic pulse to every valid channel,
    measure evoked response.

    Returns:
        channel_ranking: Indices sorted by responsiveness.
        responsiveness: Response delta (evoked - baseline).
    """
    print("  [Calibration] Channel warm-up calibrating on real biology...")
    n_rounds = int(duration_sec * 250)  # 10s * 250 rounds of 100 frames @ 25kHz = 250,000 frames
    baseline_rounds = n_rounds // 2
    stim_rounds = n_rounds - baseline_rounds

    baseline_responses = np.zeros(64)
    for _ in range(baseline_rounds):
        frames = neurons.read(100, None)
        baseline_responses += np.mean(np.abs(frames.astype(float)), axis=0)
    baseline_responses /= max(1, baseline_rounds)

    if max_stim_amplitude <= 0:
        raise ValueError("max_stim_amplitude must be positive")
    probe_amplitude = min(1.0, float(max_stim_amplitude))
    stim = StimDesign(
        160,
        -probe_amplitude,
        160,
        probe_amplitude,
    )
    burst = BurstDesign(1, 50)  # Single weak pulse to avoid global seizure
    stim_responses = np.zeros(64)
    rounds_per_ch = max(1, stim_rounds // len(STIMULATABLE_CHANNELS))

    # Sequentially stimulate only hardware-valid channels.
    for ch in STIMULATABLE_CHANNELS:
        neurons.stim(ChannelSet(ch), stim, burst)
        ch_resp = 0.0
        for _ in range(rounds_per_ch):
            frames = neurons.read(100, None)
            ch_resp += np.mean(np.abs(frames[:, ch].astype(float)))
        stim_responses[ch] = ch_resp / max(1, rounds_per_ch)

    responsiveness = stim_responses - baseline_responses
    # Non-stimulatable electrodes may still be recorded, but they cannot be
    # selected by legacy stimulation code using this ranking.
    channel_ranking = np.asarray(
        sorted(
            STIMULATABLE_CHANNELS,
            key=lambda channel: responsiveness[channel],
            reverse=True,
        )
        + sorted(NON_STIMULATABLE_CHANNELS),
        dtype=np.int64,
    )

    top8 = channel_ranking[:8]
    print(f"  [Calibration] Done! Top-8: {top8.tolist()} "
          f"range: {responsiveness[top8[0]]:.1f}~{responsiveness[top8[-1]]:.1f}")
    return channel_ranking, responsiveness


@dataclass(frozen=True)
class TimestampedCalibrationResult:
    """Input-evoked and output-recorded health from one blinded calibration."""

    channel_ranking: np.ndarray
    output_responsiveness: np.ndarray
    input_responsiveness: np.ndarray
    baseline_counts: np.ndarray
    evoked_counts_by_input: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel_ranking": self.channel_ranking.tolist(),
            "output_responsiveness": self.output_responsiveness.tolist(),
            "input_responsiveness": self.input_responsiveness.tolist(),
            "baseline_counts": self.baseline_counts.tolist(),
            "evoked_counts_by_input": self.evoked_counts_by_input.tolist(),
        }


def timestamped_contact_calibration(
    neurons,
    duration_sec: float = 10.0,
    *,
    artifact_wait_ms: float = 50.0,
    collect_window_ms: float = 50.0,
    max_stim_amplitude: float = 0.75,
) -> TimestampedCalibrationResult:
    """Measure per-input efficacy and per-output response using CL timestamps."""

    from core.spike_pipeline import SpikeWindowConfig, SpikeWindowReader

    print("  [Calibration] Timestamped contact-channel calibration...")
    reader = SpikeWindowReader(
        neurons,
        SpikeWindowConfig(
            artifact_wait_ms=artifact_wait_ms,
            collect_window_ms=collect_window_ms,
            bin_width_ms=10.0,
            tick_ms=10.0,
        ),
    )
    baseline_window_count = max(3, int(max(duration_sec, 0.1) * 5))
    baseline_counts = np.zeros(64, dtype=np.float64)
    for _ in range(baseline_window_count):
        window = reader.read()
        baseline_counts += np.asarray(
            window.features.channel_counts,
            dtype=np.float64,
        )
    baseline_counts /= baseline_window_count

    if max_stim_amplitude <= 0:
        raise ValueError("max_stim_amplitude must be positive")
    probe_amplitude = min(0.75, float(max_stim_amplitude))
    probe = StimDesign(
        160,
        -probe_amplitude,
        160,
        probe_amplitude,
    )
    evoked_counts_by_input = np.zeros((64, 64), dtype=np.float64)
    input_responsiveness = np.zeros(64, dtype=np.float64)
    for input_channel in STIMULATABLE_CHANNELS:
        neurons.stim(
            ChannelSet(input_channel),
            probe,
            BurstDesign(1, 50),
        )
        trigger_timestamp = getattr(neurons, "last_stim_timestamp", None)
        window = reader.read(
            trigger_stim_timestamps=(
                ()
                if trigger_timestamp is None
                else (int(trigger_timestamp),)
            ),
        )
        counts = np.asarray(
            window.features.channel_counts,
            dtype=np.float64,
        )
        evoked_counts_by_input[input_channel] = counts
        input_responsiveness[input_channel] = float(np.sum(
            np.maximum(counts - baseline_counts, 0.0)
        ))

    valid_rows = evoked_counts_by_input[list(STIMULATABLE_CHANNELS)]
    mean_evoked_counts = np.mean(valid_rows, axis=0)
    output_responsiveness = mean_evoked_counts - baseline_counts
    channel_ranking = np.argsort(output_responsiveness)[::-1]
    top8 = channel_ranking[:8]
    print(
        f"  [Calibration] Done! Top-8: {top8.tolist()} "
        f"spike delta: {output_responsiveness[top8[0]]:.3f}"
        f"~{output_responsiveness[top8[-1]]:.3f}"
    )
    return TimestampedCalibrationResult(
        channel_ranking=channel_ranking,
        output_responsiveness=output_responsiveness,
        input_responsiveness=input_responsiveness,
        baseline_counts=baseline_counts,
        evoked_counts_by_input=evoked_counts_by_input,
    )


def timestamped_warmup_calibration(
    neurons,
    duration_sec: float = 10.0,
    *,
    artifact_wait_ms: float = 50.0,
    collect_window_ms: float = 50.0,
    max_stim_amplitude: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rank output electrodes using SDK-detected, timestamped spikes.

    Baseline and evoked responses use the same artifact-separated collection
    window as the control experiment.  This replaces raw-voltage amplitude as
    the default hardware calibration metric.
    """

    result = timestamped_contact_calibration(
        neurons,
        duration_sec=duration_sec,
        artifact_wait_ms=artifact_wait_ms,
        collect_window_ms=collect_window_ms,
        max_stim_amplitude=max_stim_amplitude,
    )
    return result.channel_ranking, result.output_responsiveness
