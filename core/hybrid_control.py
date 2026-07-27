"""Hybrid nominal-plus-CL1 residual control primitives.

This module deliberately keeps long-horizon task state in a deterministic
white-box controller.  The biological path is granted authority over one
bounded residual axis only.  The split makes the first CL1 experiment narrow,
auditable, and safe to ablate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

import numpy as np


class TaskPhase(str, Enum):
    """Externally maintained task phase for the nominal controller."""

    APPROACH_NUT = "approach_nut"
    GRASP = "grasp"
    TRANSPORT = "transport"
    INSERT = "insert"
    RELEASE = "release"
    RETREAT = "retreat"
    COMPLETE = "complete"


@dataclass(frozen=True)
class NominalControlConfig:
    """Configuration for the deterministic task controller."""

    action_dim: int = 7
    position_gain: float = 6.0
    insertion_gain: float = 4.0
    max_translation_norm: float = 0.60
    max_insertion_norm: float = 0.15
    hover_height_m: float = 0.08
    hover_tolerance_m: float = 0.025
    grasp_z_offset_m: float = -0.03
    grasp_distance_m: float = 0.04
    grasp_yaw_tolerance_rad: float = 0.12
    grasp_confirm_steps: int = 10
    grasp_timeout_steps: int = 60
    grasp_loss_tolerance_steps: int = 10
    lift_height_above_table_m: float = 0.15
    lift_tolerance_m: float = 0.02
    transport_clearance_m: float = 0.10
    transport_xy_tolerance_m: float = 0.008
    transport_z_tolerance_m: float = 0.02
    yaw_tolerance_rad: float = 0.06
    yaw_gain: float = 1.5
    max_yaw_action: float = 0.30
    yaw_action_index: int = 5
    release_hold_steps: int = 12
    retreat_speed: float = 0.25
    retreat_min_steps: int = 10
    gripper_index: int = -1


@dataclass(frozen=True)
class ResidualControlConfig:
    """Authority and safety limits for the biological residual."""

    residual_axis: int = 2
    residual_scale: float = 0.08
    max_residual_abs: float = 0.05
    min_residual_confidence: float = 0.15
    max_translation_norm: float = 0.30
    force_soft_limit_n: float = 20.0
    force_hard_limit_n: float = 25.0


@dataclass(frozen=True)
class HybridControlReport:
    """Auditable record of one nominal-plus-residual composition."""

    phase: str
    mode: str
    baseline_action: list[float]
    proposed_residual: float
    applied_residual: float
    final_action: list[float]
    residual_axis: int
    residual_confidence: float
    force_n: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return asdict(self)


class NominalTaskController:
    """Small deterministic state machine for NutAssembly approach and insertion.

    It is intentionally not a learned policy.  Task phase and gripper intent
    remain outside the biological substrate so the CL1 path can be tested as a
    local contact-control residual rather than a seven-dimensional policy.
    """

    def __init__(self, config: NominalControlConfig | None = None) -> None:
        self.config = config or NominalControlConfig()
        if self.config.action_dim < 3:
            raise ValueError("action_dim must provide at least xyz translation")
        self.phase = TaskPhase.APPROACH_NUT
        self._approach_stage = "hover"
        self._transport_stage = "lift"
        self._grasp_steps = 0
        self._grasp_confirmed_steps = 0
        self._lost_grasp_steps = 0
        self._release_steps = 0
        self._retreat_steps = 0

    def reset(self) -> None:
        """Reset externally held task state for a new episode."""

        self.phase = TaskPhase.APPROACH_NUT
        self._approach_stage = "hover"
        self._transport_stage = "lift"
        self._grasp_steps = 0
        self._grasp_confirmed_steps = 0
        self._lost_grasp_steps = 0
        self._release_steps = 0
        self._retreat_steps = 0

    def propose(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Return a bounded white-box action for the current task phase."""

        eef_to_nut = _vector3(observation.get("eef_to_nut"))
        nut_to_peg = _vector3(
            observation.get("nut_to_peg", observation.get("peg_to_hole"))
        )
        grasp_confirmed = bool(observation.get("grasp_confirmed", False))
        nut_on_peg = bool(observation.get("nut_on_peg", False))
        placement_success = bool(observation.get("placement_success", False))
        yaw_error_rad = float(observation.get(
            "nut_peg_yaw_error_rad",
            0.0,
        ))
        grasp_yaw_error_rad = float(observation.get(
            "grasp_yaw_error_rad",
            0.0,
        ))
        nut_height = float(observation.get(
            "nut_height_above_table_m",
            self.config.lift_height_above_table_m,
        ))

        target, gripper, gain, max_norm, yaw_action = self._phase_command(
            eef_to_nut=eef_to_nut,
            nut_to_peg=nut_to_peg,
            nut_height_above_table_m=nut_height,
            yaw_error_rad=yaw_error_rad,
            grasp_yaw_error_rad=grasp_yaw_error_rad,
            grasp_confirmed=grasp_confirmed,
            nut_on_peg=nut_on_peg,
            placement_success=placement_success,
        )

        action = np.zeros(self.config.action_dim, dtype=float)
        action[:3] = _clip_norm(
            target * gain,
            max_norm,
        )
        if self.config.action_dim > self.config.yaw_action_index:
            action[self.config.yaw_action_index] = yaw_action
        gripper_index = self.config.gripper_index % self.config.action_dim
        action[gripper_index] = gripper
        return np.clip(action, -1.0, 1.0)

    def _phase_command(
        self,
        *,
        eef_to_nut: np.ndarray,
        nut_to_peg: np.ndarray,
        nut_height_above_table_m: float,
        yaw_error_rad: float,
        grasp_yaw_error_rad: float,
        grasp_confirmed: bool,
        nut_on_peg: bool,
        placement_success: bool,
    ) -> tuple[np.ndarray, float, float, float, float]:
        cfg = self.config
        zero = np.zeros(3, dtype=float)

        # A nut can settle onto the peg after a contact transient or a lost
        # grasp. Do not re-grasp an already placed object; release and retreat
        # so the official environment check can evaluate the placement.
        if (
            nut_on_peg
            and self.phase
            not in {
                TaskPhase.RELEASE,
                TaskPhase.RETREAT,
                TaskPhase.COMPLETE,
            }
        ):
            self.phase = TaskPhase.RELEASE
            self._release_steps = 0

        if self.phase is TaskPhase.APPROACH_NUT:
            grasp_yaw_action = float(np.clip(
                grasp_yaw_error_rad * cfg.yaw_gain,
                -cfg.max_yaw_action,
                cfg.max_yaw_action,
            ))
            if self._approach_stage == "hover":
                target = eef_to_nut + np.array(
                    [0.0, 0.0, cfg.hover_height_m],
                    dtype=float,
                )
                if (
                    np.linalg.norm(target) <= cfg.hover_tolerance_m
                    and abs(grasp_yaw_error_rad)
                    <= cfg.grasp_yaw_tolerance_rad
                ):
                    self._approach_stage = "descend"
            if self._approach_stage == "descend":
                target = eef_to_nut + np.array(
                    [0.0, 0.0, cfg.grasp_z_offset_m],
                    dtype=float,
                )
            if (
                self._approach_stage == "descend"
                and np.linalg.norm(target) <= cfg.grasp_distance_m
            ):
                self.phase = TaskPhase.GRASP
                self._grasp_steps = 0
                self._grasp_confirmed_steps = 0
                return (
                    zero,
                    1.0,
                    cfg.position_gain,
                    cfg.max_translation_norm,
                    grasp_yaw_action,
                )
            return (
                target,
                -1.0,
                cfg.position_gain,
                cfg.max_translation_norm,
                grasp_yaw_action,
            )

        if self.phase is TaskPhase.GRASP:
            self._grasp_steps += 1
            self._grasp_confirmed_steps = (
                self._grasp_confirmed_steps + 1
                if grasp_confirmed
                else 0
            )
            if self._grasp_confirmed_steps >= cfg.grasp_confirm_steps:
                self.phase = TaskPhase.TRANSPORT
                self._transport_stage = "lift"
                self._lost_grasp_steps = 0
            elif self._grasp_steps >= cfg.grasp_timeout_steps:
                # Never infer a grasp from elapsed time or wrist force. Re-open
                # and retry the approach if object contacts were not confirmed.
                self._retry_grasp()
                return (
                    zero,
                    -1.0,
                    cfg.position_gain,
                    cfg.max_translation_norm,
                    0.0,
                )
            else:
                return (
                    zero,
                    1.0,
                    cfg.position_gain,
                    cfg.max_translation_norm,
                    0.0,
                )

        if self.phase is TaskPhase.TRANSPORT:
            if self._grasp_was_lost(grasp_confirmed, nut_on_peg):
                self._retry_grasp()
                return (
                    zero,
                    -1.0,
                    cfg.position_gain,
                    cfg.max_translation_norm,
                    0.0,
                )

            if self._transport_stage == "lift":
                height_error = (
                    cfg.lift_height_above_table_m
                    - nut_height_above_table_m
                )
                if height_error <= cfg.lift_tolerance_m:
                    self._transport_stage = "align"
                else:
                    return (
                        np.array([0.0, 0.0, height_error], dtype=float),
                        1.0,
                        cfg.position_gain,
                        cfg.max_translation_norm,
                        0.0,
                    )

            target = nut_to_peg + np.array(
                [0.0, 0.0, cfg.transport_clearance_m],
                dtype=float,
            )
            yaw_action = float(np.clip(
                -yaw_error_rad * cfg.yaw_gain,
                -cfg.max_yaw_action,
                cfg.max_yaw_action,
            ))
            if (
                np.linalg.norm(nut_to_peg[:2])
                <= cfg.transport_xy_tolerance_m
                and abs(target[2]) <= cfg.transport_z_tolerance_m
                and abs(yaw_error_rad) <= cfg.yaw_tolerance_rad
            ):
                self.phase = TaskPhase.INSERT
            else:
                return (
                    target,
                    1.0,
                    cfg.position_gain,
                    cfg.max_translation_norm,
                    yaw_action,
                )

        if self.phase is TaskPhase.INSERT:
            if nut_on_peg:
                self.phase = TaskPhase.RELEASE
                self._release_steps = 0
            elif self._grasp_was_lost(grasp_confirmed, nut_on_peg):
                self._retry_grasp()
                return (
                    zero,
                    -1.0,
                    cfg.position_gain,
                    cfg.max_translation_norm,
                    0.0,
                )
            else:
                yaw_action = float(np.clip(
                    -yaw_error_rad * cfg.yaw_gain,
                    -cfg.max_yaw_action,
                    cfg.max_yaw_action,
                ))
                return (
                    nut_to_peg,
                    1.0,
                    cfg.insertion_gain,
                    cfg.max_insertion_norm,
                    yaw_action,
                )

        if self.phase is TaskPhase.RELEASE:
            self._release_steps += 1
            if self._release_steps >= cfg.release_hold_steps:
                self.phase = TaskPhase.RETREAT
                self._retreat_steps = 0
            return (
                zero,
                -1.0,
                cfg.position_gain,
                cfg.max_translation_norm,
                0.0,
            )

        if self.phase is TaskPhase.RETREAT:
            self._retreat_steps += 1
            if (
                self._retreat_steps >= cfg.retreat_min_steps
                and placement_success
            ):
                self.phase = TaskPhase.COMPLETE
                return (
                    zero,
                    -1.0,
                    cfg.position_gain,
                    cfg.max_translation_norm,
                    0.0,
                )
            return (
                np.array([0.0, 0.0, cfg.retreat_speed], dtype=float),
                -1.0,
                1.0,
                cfg.retreat_speed,
                0.0,
            )

        return zero, -1.0, cfg.position_gain, cfg.max_translation_norm, 0.0

    def _grasp_was_lost(
        self,
        grasp_confirmed: bool,
        nut_on_peg: bool,
    ) -> bool:
        if grasp_confirmed or nut_on_peg:
            self._lost_grasp_steps = 0
        else:
            self._lost_grasp_steps += 1
        return self._lost_grasp_steps >= self.config.grasp_loss_tolerance_steps

    def _retry_grasp(self) -> None:
        self.phase = TaskPhase.APPROACH_NUT
        self._approach_stage = "hover"
        self._transport_stage = "lift"
        self._grasp_steps = 0
        self._grasp_confirmed_steps = 0
        self._lost_grasp_steps = 0
        self._release_steps = 0
        self._retreat_steps = 0


class BoundedResidualController:
    """Compose a baseline action with one bounded, confidence-gated residual."""

    def __init__(self, config: ResidualControlConfig | None = None) -> None:
        self.config = config or ResidualControlConfig()
        if self.config.residual_axis not in (0, 1, 2):
            raise ValueError("residual_axis must be a translation axis: 0, 1, or 2")
        if self.config.max_residual_abs < 0:
            raise ValueError("max_residual_abs must be non-negative")
        if not 0.0 <= self.config.min_residual_confidence <= 1.0:
            raise ValueError("min_residual_confidence must be in [0, 1]")

    def compose(
        self,
        baseline_action: Sequence[float] | np.ndarray,
        neural_action: Sequence[float] | np.ndarray | None,
        *,
        phase: TaskPhase | str,
        residual_confidence: float,
        force_n: float,
        residual_enabled: bool = True,
    ) -> tuple[np.ndarray, HybridControlReport]:
        """Return the safe final action and an auditable composition report."""

        baseline = np.asarray(baseline_action, dtype=float).reshape(-1)
        if baseline.size <= self.config.residual_axis:
            raise ValueError("baseline_action does not contain the residual axis")
        if not np.all(np.isfinite(baseline)):
            raise ValueError("baseline_action must be finite")

        confidence = float(np.clip(residual_confidence, 0.0, 1.0))
        measured_force = float(force_n)
        phase_value = phase.value if isinstance(phase, TaskPhase) else str(phase)

        proposed = 0.0
        applied = 0.0
        mode = "baseline_only"
        reason = "no_neural_action"

        if measured_force >= self.config.force_hard_limit_n:
            final = np.zeros_like(baseline)
            mode = "hard_stop"
            reason = "force_hard_limit"
        else:
            final = baseline.copy()
            neural = None if neural_action is None else np.asarray(neural_action, dtype=float).reshape(-1)
            if not residual_enabled:
                reason = "protocol_phase_gate"
            elif neural is not None and neural.size > self.config.residual_axis:
                proposed = float(neural[self.config.residual_axis]) * self.config.residual_scale
                if not np.isfinite(proposed):
                    reason = "non_finite_neural_action"
                elif measured_force >= self.config.force_soft_limit_n:
                    reason = "force_soft_limit"
                elif confidence < self.config.min_residual_confidence:
                    reason = "low_residual_confidence"
                else:
                    applied = float(np.clip(
                        proposed,
                        -self.config.max_residual_abs,
                        self.config.max_residual_abs,
                    ))
                    final[self.config.residual_axis] += applied
                    mode = "baseline_plus_cl_residual"
                    reason = "residual_applied"

            final[:3] = _clip_norm(final[:3], self.config.max_translation_norm)
            final = np.clip(final, -1.0, 1.0)

        report = HybridControlReport(
            phase=phase_value,
            mode=mode,
            baseline_action=baseline.tolist(),
            proposed_residual=proposed,
            applied_residual=applied,
            final_action=final.tolist(),
            residual_axis=self.config.residual_axis,
            residual_confidence=confidence,
            force_n=measured_force,
            reason=reason,
        )
        return final, report


def spike_confidence(spike_channels: Sequence[int], target_count: int = 4) -> float:
    """Convert the number of responsive channels into a conservative confidence."""

    if target_count <= 0:
        raise ValueError("target_count must be positive")
    unique_channels = len(set(int(channel) for channel in spike_channels))
    return float(np.clip(unique_channels / target_count, 0.0, 1.0))


def summarize_control_reports(
    reports: Sequence[HybridControlReport | Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize residual authority and fallback behavior for one episode."""

    if not reports:
        return {
            "step_count": 0,
            "residual_applied_rate": 0.0,
            "mean_abs_applied_residual": 0.0,
            "hard_stop_count": 0,
            "mode_counts": {},
            "phase_counts": {},
        }

    normalized = [
        report.to_dict() if isinstance(report, HybridControlReport) else dict(report)
        for report in reports
    ]
    mode_counts: dict[str, int] = {}
    phase_counts: dict[str, int] = {}
    applied_values: list[float] = []

    for report in normalized:
        mode = str(report.get("mode", "unknown"))
        phase = str(report.get("phase", "unknown"))
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        applied_values.append(abs(float(report.get("applied_residual", 0.0))))

    step_count = len(normalized)
    applied_count = mode_counts.get("baseline_plus_cl_residual", 0)
    return {
        "step_count": step_count,
        "residual_applied_rate": applied_count / step_count,
        "mean_abs_applied_residual": float(np.mean(applied_values)),
        "hard_stop_count": mode_counts.get("hard_stop", 0),
        "mode_counts": mode_counts,
        "phase_counts": phase_counts,
    }


def _vector3(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros(3, dtype=float)
    vector = np.asarray(value, dtype=float).reshape(-1)
    output = np.zeros(3, dtype=float)
    output[: min(3, vector.size)] = vector[:3]
    return output


def _clip_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= max_norm or norm <= 1e-12:
        return vector
    return vector * (max_norm / norm)
