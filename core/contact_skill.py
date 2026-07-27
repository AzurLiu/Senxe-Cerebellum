"""CL1 contact-skill boundary for safe, multi-axis manipulation.

The biological path receives a compact contact state and emits five skill
coordinates:

    [delta_x, delta_y, delta_z, soften, retract]

Long-horizon task state, rotation, gripper intent, joint control and safety
remain outside the biological substrate.  ``soften`` can only reduce motion
authority, never increase it.  ``retract`` selects a deterministic retreat
primitive whose direction is computed by the safety controller.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from core.channel_map import (
    DEFAULT_CONTACT_CHANNEL_MAP,
    STIMULATABLE_CHANNELS,
)
from core.hybrid_control import TaskPhase
from core.neurons import BurstDesign, ChannelSet, StimDesign


CONTACT_SKILL_DIM = 5
TRANSLATION_SKILL_INDICES = (0, 1, 2)
SOFTEN_SKILL_INDEX = 3
RETRACT_SKILL_INDEX = 4


@dataclass(frozen=True)
class ContactEncoderConfig:
    """Sparse, contact-focused MEA encoding configuration."""

    position_deadband_m: float = 0.001
    position_large_m: float = 0.020
    force_deadband_n: float = 0.25
    force_large_n: float = 5.0
    contact_soft_n: float = 1.0
    contact_hard_n: float = 12.0
    max_stim_amplitude: float = 1.5
    position_channels: tuple[int, ...] = (
        DEFAULT_CONTACT_CHANNEL_MAP.position_channels
    )
    force_channels: tuple[int, ...] = (
        DEFAULT_CONTACT_CHANNEL_MAP.force_channels
    )
    phase_channels: tuple[int, ...] = (
        DEFAULT_CONTACT_CHANNEL_MAP.phase_channels
    )
    contact_channels: tuple[int, ...] = (
        DEFAULT_CONTACT_CHANNEL_MAP.contact_channels
    )

    def __post_init__(self) -> None:
        if self.max_stim_amplitude <= 0:
            raise ValueError("max_stim_amplitude must be positive")
        if len(self.position_channels) != 6:
            raise ValueError("position_channels must contain six channels")
        if len(self.force_channels) != 6:
            raise ValueError("force_channels must contain six channels")
        if len(self.phase_channels) != 3:
            raise ValueError("phase_channels must contain three code channels")
        if len(self.contact_channels) != 3:
            raise ValueError("contact_channels must contain three channels")
        sensory_channels = (
            self.position_channels
            + self.force_channels
            + self.phase_channels
            + self.contact_channels
        )
        if len(sensory_channels) != len(set(sensory_channels)):
            raise ValueError("sensory channel groups must be disjoint")
        invalid = set(sensory_channels) - set(STIMULATABLE_CHANNELS)
        if invalid:
            raise ValueError(
                "sensory channel groups contain non-stimulatable channels: "
                f"{sorted(invalid)}"
            )


@dataclass(frozen=True)
class ContactEncodingReport:
    """Auditable record of one compact sensory stimulation."""

    phase: str
    position_error_m: list[float]
    force_n: list[float]
    normalized_features: list[float]
    stimulated_channels: list[int]
    contact_state: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ContactStateEncoder:
    """Encode only alignment, contact wrench and external task phase."""

    def __init__(
        self,
        neurons: Any,
        config: ContactEncoderConfig | None = None,
    ) -> None:
        self.neurons = neurons
        self.config = config or ContactEncoderConfig()
        self.last_report: ContactEncodingReport | None = None

    def reset(self) -> None:
        self.last_report = None

    def encode(
        self,
        observation: Mapping[str, Any],
        phase: TaskPhase | str,
        *,
        stimulation_enabled: bool = True,
    ) -> ContactEncodingReport:
        # During the CL-enabled transport / insertion phases, the physically
        # meaningful error is the nut pose relative to the peg. The historical
        # ``peg_to_hole`` name is retained only as a compatibility fallback.
        position_error = _vector3(
            observation.get("nut_to_peg", observation.get("peg_to_hole"))
        )
        force = _vector3(observation.get("force"))
        phase_value = phase.value if isinstance(phase, TaskPhase) else str(phase)
        stimulated: list[int] = []

        for axis in range(3):
            position_channel = _signed_axis_channel(
                position_error[axis],
                deadband=self.config.position_deadband_m,
                channels=self.config.position_channels[
                    axis * 2:(axis + 1) * 2
                ],
            )
            if position_channel is not None:
                magnitude = abs(float(position_error[axis]))
                amplitude = np.clip(
                    0.25 + magnitude / max(self.config.position_large_m, 1e-9),
                    0.25,
                    self.config.max_stim_amplitude,
                )
                if stimulation_enabled:
                    self._stim(
                        position_channel,
                        float(amplitude),
                        burst_hz=100,
                    )
                    stimulated.append(position_channel)

            force_channel = _signed_axis_channel(
                force[axis],
                deadband=self.config.force_deadband_n,
                channels=self.config.force_channels[
                    axis * 2:(axis + 1) * 2
                ],
            )
            if force_channel is not None:
                magnitude = abs(float(force[axis]))
                amplitude = np.clip(
                    0.25 + magnitude / max(self.config.force_large_n, 1e-9),
                    0.25,
                    self.config.max_stim_amplitude,
                )
                if stimulation_enabled:
                    self._stim(
                        force_channel,
                        float(amplitude),
                        burst_hz=140,
                    )
                    stimulated.append(force_channel)

        phase_members = list(TaskPhase)
        try:
            phase_index = [member.value for member in phase_members].index(
                phase_value
            )
        except ValueError:
            phase_index = 0
        # Seven phases fit into the seven non-zero patterns of three channels.
        phase_code = phase_index + 1
        phase_pattern = tuple(
            channel
            for bit, channel in enumerate(self.config.phase_channels)
            if phase_code & (1 << bit)
        )
        if stimulation_enabled:
            for phase_channel in phase_pattern:
                self._stim(phase_channel, 0.4, burst_hz=80)
                stimulated.append(phase_channel)

        force_magnitude = float(np.linalg.norm(force))
        if force_magnitude < self.config.contact_soft_n:
            contact_index = 0
            contact_state = "free"
        elif force_magnitude < self.config.contact_hard_n:
            contact_index = 1
            contact_state = "contact"
        else:
            contact_index = 2
            contact_state = "high_force"
        contact_channel = self.config.contact_channels[contact_index]
        if stimulation_enabled:
            self._stim(contact_channel, 0.5, burst_hz=100)
            stimulated.append(contact_channel)

        normalized = np.concatenate([
            np.clip(
                position_error / max(self.config.position_large_m, 1e-9),
                -1.0,
                1.0,
            ),
            np.clip(
                force / max(self.config.force_large_n, 1e-9),
                -1.0,
                1.0,
            ),
        ])
        report = ContactEncodingReport(
            phase=phase_value,
            position_error_m=position_error.tolist(),
            force_n=force.tolist(),
            normalized_features=normalized.tolist(),
            stimulated_channels=stimulated,
            contact_state=contact_state,
        )
        self.last_report = report
        return report

    def _stim(self, channel: int, amplitude: float, *, burst_hz: int) -> None:
        amplitude = float(min(
            abs(amplitude),
            self.config.max_stim_amplitude,
        ))
        design = StimDesign(160, -amplitude, 160, amplitude)
        self.neurons.stim(
            ChannelSet(int(channel)),
            design,
            BurstDesign(1, int(burst_hz)),
        )


@dataclass(frozen=True)
class ContactSkillControlConfig:
    """Authority budget for the five-dimensional CL1 contact skill."""

    residual_scales: tuple[float, float, float] = (0.08, 0.08, 0.06)
    max_residual_abs: tuple[float, float, float] = (0.04, 0.04, 0.03)
    min_confidence: float = 0.15
    min_compliance_scale: float = 0.35
    retract_threshold: float = 0.55
    retract_speed: float = 0.08
    retract_min_force_n: float = 1.0
    max_translation_norm: float = 0.60
    force_soft_limit_n: float = 20.0
    force_hard_limit_n: float = 25.0
    enabled_phases: tuple[str, ...] = (
        TaskPhase.TRANSPORT.value,
        TaskPhase.INSERT.value,
    )

    def __post_init__(self) -> None:
        if len(self.residual_scales) != 3 or len(self.max_residual_abs) != 3:
            raise ValueError("contact residual settings must have xyz values")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if not 0.0 < self.min_compliance_scale <= 1.0:
            raise ValueError("min_compliance_scale must be in (0, 1]")
        if not 0.0 <= self.retract_threshold <= 1.0:
            raise ValueError("retract_threshold must be in [0, 1]")


@dataclass(frozen=True)
class ContactSkillReport:
    """One safe composition of nominal control and a CL1 contact skill."""

    phase: str
    mode: str
    baseline_action: list[float]
    skill_vector: list[float]
    proposed_residual_vector: list[float]
    applied_residual_vector: list[float]
    compliance_signal: float
    compliance_scale: float
    retract_signal: float
    retract_applied: bool
    final_action: list[float]
    residual_confidence: float
    force_vector_n: list[float]
    force_n: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        # Compatibility fields consumed by the existing ablation CSV/HUD.
        value["proposed_residual"] = float(np.linalg.norm(
            self.proposed_residual_vector
        ))
        value["applied_residual"] = float(np.linalg.norm(
            self.applied_residual_vector
        ))
        value["residual_axis"] = "xyz"
        return value


class ContactSkillController:
    """Grant CL1 multi-axis contact authority without task-level control."""

    def __init__(
        self,
        config: ContactSkillControlConfig | None = None,
    ) -> None:
        self.config = config or ContactSkillControlConfig()

    def compose(
        self,
        baseline_action: Sequence[float] | np.ndarray,
        skill_vector: Sequence[float] | np.ndarray,
        *,
        phase: TaskPhase | str,
        residual_confidence: float,
        force_vector_n: Sequence[float] | np.ndarray,
        residual_enabled: bool = True,
    ) -> tuple[np.ndarray, ContactSkillReport]:
        baseline = np.asarray(baseline_action, dtype=float).reshape(-1)
        skill = np.asarray(skill_vector, dtype=float).reshape(-1)
        force = _vector3(force_vector_n)
        if baseline.size < 3:
            raise ValueError("baseline_action must contain xyz translation")
        if skill.shape != (CONTACT_SKILL_DIM,):
            raise ValueError(
                f"skill_vector must have shape ({CONTACT_SKILL_DIM},)"
            )
        if not np.all(np.isfinite(baseline)):
            raise ValueError("baseline_action must be finite")

        cfg = self.config
        phase_value = phase.value if isinstance(phase, TaskPhase) else str(phase)
        confidence = float(np.clip(residual_confidence, 0.0, 1.0))
        force_n = float(np.linalg.norm(force))
        proposed = (
            skill[:3] * np.asarray(cfg.residual_scales, dtype=float)
        )
        proposed = np.where(np.isfinite(proposed), proposed, 0.0)
        applied = np.zeros(3, dtype=float)
        compliance_signal = float(
            skill[SOFTEN_SKILL_INDEX]
            if np.isfinite(skill[SOFTEN_SKILL_INDEX])
            else 0.0
        )
        retract_signal = float(
            skill[RETRACT_SKILL_INDEX]
            if np.isfinite(skill[RETRACT_SKILL_INDEX])
            else 0.0
        )
        compliance_scale = 1.0
        retract_applied = False
        final = baseline.copy()
        mode = "baseline_only"
        reason = "contact_skill_disabled"

        if force_n >= cfg.force_hard_limit_n:
            final[:] = 0.0
            mode = "hard_stop"
            reason = "force_hard_limit"
        elif not residual_enabled:
            reason = "protocol_phase_gate"
        elif phase_value not in cfg.enabled_phases:
            reason = "outside_contact_phase"
        elif confidence < cfg.min_confidence:
            reason = "low_residual_confidence"
        elif force_n >= cfg.force_soft_limit_n:
            reason = "force_soft_limit"
        elif (
            retract_signal >= cfg.retract_threshold
            and force_n >= cfg.retract_min_force_n
        ):
            direction = -force / max(force_n, 1e-9)
            final[:3] = direction * cfg.retract_speed
            retract_applied = True
            mode = "cl_selected_safe_retract"
            reason = "retract_selected"
        else:
            soften = float(np.clip(compliance_signal, 0.0, 1.0))
            compliance_scale = 1.0 - (
                soften * (1.0 - cfg.min_compliance_scale)
            )
            final[:3] *= compliance_scale
            applied = np.clip(
                proposed,
                -np.asarray(cfg.max_residual_abs),
                np.asarray(cfg.max_residual_abs),
            )
            final[:3] += applied
            mode = "baseline_plus_cl_contact_skill"
            reason = "contact_skill_applied"

        final[:3] = _clip_norm(final[:3], cfg.max_translation_norm)
        final = np.clip(final, -1.0, 1.0)
        report = ContactSkillReport(
            phase=phase_value,
            mode=mode,
            baseline_action=baseline.tolist(),
            skill_vector=skill.tolist(),
            proposed_residual_vector=proposed.tolist(),
            applied_residual_vector=applied.tolist(),
            compliance_signal=compliance_signal,
            compliance_scale=compliance_scale,
            retract_signal=retract_signal,
            retract_applied=retract_applied,
            final_action=final.tolist(),
            residual_confidence=confidence,
            force_vector_n=force.tolist(),
            force_n=force_n,
            reason=reason,
        )
        return final, report


@dataclass(frozen=True)
class ContactFeedbackConfig:
    """Task-event thresholds for structured biological feedback."""

    progress_epsilon_m: float = 0.0005
    regress_epsilon_m: float = 0.001
    high_force_n: float = 16.0
    stalled_force_n: float = 6.0
    success_strength: float = 1.0
    progress_strength: float = 0.45
    regress_strength: float = 0.55
    collision_strength: float = 1.0
    cooldown_steps: int = 3

    def __post_init__(self) -> None:
        if self.cooldown_steps < 0:
            raise ValueError("cooldown_steps must be non-negative")


@dataclass(frozen=True)
class ContactFeedbackEvent:
    kind: str
    valence: int
    strength: float
    distance_delta_m: float
    force_n: float

    @property
    def active(self) -> bool:
        return self.valence != 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ContactFeedbackEvent":
        return cls(
            kind=str(value["kind"]),
            valence=int(value["valence"]),
            strength=float(value["strength"]),
            distance_delta_m=float(value["distance_delta_m"]),
            force_n=float(value["force_n"]),
        )


class ContactFeedbackEvaluator:
    """Convert physical contact outcomes into fixed, auditable events."""

    def __init__(
        self,
        config: ContactFeedbackConfig | None = None,
    ) -> None:
        self.config = config or ContactFeedbackConfig()
        self.previous_distance: float | None = None
        self.step_index = -1
        self.last_active_step: int | None = None

    def reset(self, initial_distance_m: float | None = None) -> None:
        self.previous_distance = (
            None if initial_distance_m is None else float(initial_distance_m)
        )
        self.step_index = -1
        self.last_active_step = None

    def evaluate(
        self,
        *,
        distance_m: float,
        force_n: float,
        success: bool,
        enabled: bool = True,
    ) -> ContactFeedbackEvent:
        self.step_index += 1
        current = float(distance_m)
        previous = self.previous_distance
        delta = 0.0 if previous is None else previous - current
        self.previous_distance = current
        cfg = self.config

        if not enabled:
            candidate = ContactFeedbackEvent(
                "outside_contact_phase",
                0,
                0.0,
                delta,
                float(force_n),
            )
        elif success:
            candidate = ContactFeedbackEvent(
                "success", 1, cfg.success_strength, delta, float(force_n)
            )
        elif force_n >= cfg.high_force_n:
            candidate = ContactFeedbackEvent(
                "collision", -1, cfg.collision_strength, delta, float(force_n)
            )
        elif delta >= cfg.progress_epsilon_m:
            candidate = ContactFeedbackEvent(
                "contact_progress", 1, cfg.progress_strength, delta, float(force_n)
            )
        elif delta <= -cfg.regress_epsilon_m:
            candidate = ContactFeedbackEvent(
                "regress", -1, cfg.regress_strength, delta, float(force_n)
            )
        elif force_n >= cfg.stalled_force_n:
            candidate = ContactFeedbackEvent(
                "stalled_under_force",
                -1,
                cfg.regress_strength,
                delta,
                float(force_n),
            )
        else:
            candidate = ContactFeedbackEvent(
                "neutral", 0, 0.0, delta, float(force_n)
            )

        if not candidate.active:
            return candidate
        if (
            candidate.kind != "success"
            and self.last_active_step is not None
            and self.step_index - self.last_active_step < cfg.cooldown_steps
        ):
            return ContactFeedbackEvent(
                f"cooldown_suppressed_{candidate.kind}",
                0,
                0.0,
                delta,
                float(force_n),
            )
        self.last_active_step = self.step_index
        return candidate


class ContactFeedbackStimulator:
    """Emit distinct, fixed positive and negative feedback patterns."""

    def __init__(
        self,
        neurons: Any,
        *,
        positive_channels: Sequence[int] = (
            DEFAULT_CONTACT_CHANNEL_MAP.positive_feedback_channels
        ),
        negative_channels: Sequence[int] = (
            DEFAULT_CONTACT_CHANNEL_MAP.negative_feedback_channels
        ),
        max_amplitude: float = 1.5,
    ) -> None:
        self.neurons = neurons
        self.positive_channels = tuple(int(ch) for ch in positive_channels)
        self.negative_channels = tuple(int(ch) for ch in negative_channels)
        self.max_amplitude = float(max_amplitude)
        if self.max_amplitude <= 0:
            raise ValueError("max_amplitude must be positive")
        all_channels = self.positive_channels + self.negative_channels
        if not self.positive_channels or not self.negative_channels:
            raise ValueError("feedback requires positive and negative channels")
        if len(all_channels) != len(set(all_channels)):
            raise ValueError("feedback channel groups must be disjoint")
        invalid = set(all_channels) - set(STIMULATABLE_CHANNELS)
        if invalid:
            raise ValueError(
                "feedback contains non-stimulatable channels: "
                f"{sorted(invalid)}"
            )

    def emit(self, event: ContactFeedbackEvent) -> bool:
        if not event.active:
            return False
        amplitude = float(np.clip(
            0.35 + event.strength,
            min(0.35, self.max_amplitude),
            self.max_amplitude,
        ))
        if event.valence > 0:
            channels = self.positive_channels
            burst = BurstDesign(3, 100)
        else:
            channels = self.negative_channels
            burst = BurstDesign(5, 180)
        design = StimDesign(160, -amplitude, 160, amplitude)
        self.neurons.stim(ChannelSet(*channels), design, burst)
        return True


def summarize_contact_reports(
    reports: Sequence[ContactSkillReport | Mapping[str, Any]],
) -> dict[str, Any]:
    if not reports:
        return {
            "step_count": 0,
            "residual_applied_rate": 0.0,
            "mean_abs_applied_residual": 0.0,
            "mean_residual_norm": 0.0,
            "mean_compliance_scale": 1.0,
            "retract_count": 0,
            "hard_stop_count": 0,
            "mode_counts": {},
            "phase_counts": {},
        }
    normalized = [
        report.to_dict() if isinstance(report, ContactSkillReport) else dict(report)
        for report in reports
    ]
    mode_counts: dict[str, int] = {}
    phase_counts: dict[str, int] = {}
    residual_norms: list[float] = []
    compliance_scales: list[float] = []
    retract_count = 0
    for report in normalized:
        mode = str(report.get("mode", "unknown"))
        phase = str(report.get("phase", "unknown"))
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        vector = np.asarray(
            report.get("applied_residual_vector", [0.0, 0.0, 0.0]),
            dtype=float,
        )
        residual_norms.append(float(np.linalg.norm(vector)))
        compliance_scales.append(float(report.get("compliance_scale", 1.0)))
        retract_count += int(bool(report.get("retract_applied", False)))
    step_count = len(normalized)
    applied_count = mode_counts.get("baseline_plus_cl_contact_skill", 0)
    mean_norm = float(np.mean(residual_norms))
    return {
        "step_count": step_count,
        "residual_applied_rate": applied_count / step_count,
        "mean_abs_applied_residual": mean_norm,
        "mean_residual_norm": mean_norm,
        "mean_compliance_scale": float(np.mean(compliance_scales)),
        "retract_count": retract_count,
        "hard_stop_count": mode_counts.get("hard_stop", 0),
        "mode_counts": mode_counts,
        "phase_counts": phase_counts,
    }


def _signed_axis_channel(
    value: float,
    *,
    deadband: float,
    channels: Sequence[int],
) -> int | None:
    if len(channels) != 2:
        raise ValueError("signed axis encoding requires two channels")
    if abs(float(value)) <= deadband:
        return None
    index = 0 if value > 0 else 1
    return int(channels[index])


def _vector3(value: Any) -> np.ndarray:
    vector = np.asarray(
        np.zeros(3) if value is None else value,
        dtype=float,
    ).reshape(-1)
    output = np.zeros(3, dtype=float)
    output[: min(3, vector.size)] = vector[:3]
    return output


def _clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm <= 1e-12:
        return vector
    return vector * (max_norm / norm)
