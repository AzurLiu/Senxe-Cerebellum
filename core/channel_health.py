"""Outcome-blinded CL1 channel-health gates and reserve replacement."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np

from core.channel_map import (
    DEFAULT_CONTACT_CHANNEL_MAP,
    AntagonisticChannelGroup,
    ContactChannelMap,
    validate_contact_channel_map,
)


@dataclass(frozen=True)
class ChannelReplacement:
    role: str
    original_channel: int
    replacement_channel: int
    replacement_score: float


@dataclass(frozen=True)
class ChannelHealthReport:
    """Frozen result of a calibration-only channel-map decision."""

    passed: bool
    remap_allowed: bool
    input_threshold: float
    output_threshold: float
    unhealthy_input_channels: tuple[int, ...]
    unhealthy_motor_channels: tuple[int, ...]
    replacements: tuple[ChannelReplacement, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_contact_channel_map(
    input_responsiveness: np.ndarray,
    output_responsiveness: np.ndarray,
    *,
    channel_map: ContactChannelMap = DEFAULT_CONTACT_CHANNEL_MAP,
    input_threshold: float = 0.0,
    output_threshold: float = 0.0,
    allow_reserve_remap: bool = False,
) -> tuple[ContactChannelMap, ChannelHealthReport]:
    """Validate or deterministically repair a map using calibration only.

    Sensory and feedback electrodes are judged by how much network activity
    they evoke. Motor electrodes are judged by their recorded evoked response.
    Replacements are selected from the reserve in descending score order and
    never use task outcomes.
    """

    validate_contact_channel_map(channel_map)
    input_scores = _score_vector(
        input_responsiveness,
        "input_responsiveness",
    )
    output_scores = _score_vector(
        output_responsiveness,
        "output_responsiveness",
    )
    input_assigned = (
        channel_map.sensory_channels + channel_map.feedback_channels
    )
    unhealthy_inputs = tuple(
        channel
        for channel in input_assigned
        if input_scores[channel] <= input_threshold
    )
    unhealthy_motor = tuple(
        channel
        for channel in channel_map.motor_channels
        if output_scores[channel] <= output_threshold
    )
    if not unhealthy_inputs and not unhealthy_motor:
        return channel_map, ChannelHealthReport(
            passed=True,
            remap_allowed=allow_reserve_remap,
            input_threshold=float(input_threshold),
            output_threshold=float(output_threshold),
            unhealthy_input_channels=(),
            unhealthy_motor_channels=(),
            replacements=(),
            reason="base_map_passed_calibration",
        )
    if not allow_reserve_remap:
        return channel_map, ChannelHealthReport(
            passed=False,
            remap_allowed=False,
            input_threshold=float(input_threshold),
            output_threshold=float(output_threshold),
            unhealthy_input_channels=unhealthy_inputs,
            unhealthy_motor_channels=unhealthy_motor,
            replacements=(),
            reason="assigned_channels_failed_calibration",
        )

    reserve = list(channel_map.reserve_channels)
    replacements: list[ChannelReplacement] = []

    def replace_region(
        values: tuple[int, ...],
        *,
        role: str,
        scores: np.ndarray,
        threshold: float,
    ) -> tuple[int, ...]:
        resolved = list(values)
        for index, original in enumerate(values):
            if scores[original] > threshold:
                continue
            candidates = [
                channel
                for channel in reserve
                if scores[channel] > threshold
            ]
            if not candidates:
                raise RuntimeError(
                    f"no healthy reserve channel is available for {role}"
                )
            replacement = max(
                candidates,
                key=lambda channel: (scores[channel], -channel),
            )
            reserve.remove(replacement)
            reserve.append(original)
            resolved[index] = replacement
            replacements.append(ChannelReplacement(
                role=role,
                original_channel=int(original),
                replacement_channel=int(replacement),
                replacement_score=float(scores[replacement]),
            ))
        return tuple(resolved)

    position = replace_region(
        channel_map.position_channels,
        role="sensory.position",
        scores=input_scores,
        threshold=input_threshold,
    )
    force = replace_region(
        channel_map.force_channels,
        role="sensory.force",
        scores=input_scores,
        threshold=input_threshold,
    )
    phase = replace_region(
        channel_map.phase_channels,
        role="sensory.phase",
        scores=input_scores,
        threshold=input_threshold,
    )
    contact = replace_region(
        channel_map.contact_channels,
        role="sensory.contact",
        scores=input_scores,
        threshold=input_threshold,
    )
    positive_feedback = replace_region(
        channel_map.positive_feedback_channels,
        role="feedback.positive",
        scores=input_scores,
        threshold=input_threshold,
    )
    negative_feedback = replace_region(
        channel_map.negative_feedback_channels,
        role="feedback.negative",
        scores=input_scores,
        threshold=input_threshold,
    )
    motor_groups = []
    for group_index, group in enumerate(channel_map.motor_groups):
        motor_groups.append(AntagonisticChannelGroup(
            positive=replace_region(
                group.positive,
                role=f"motor.{group_index}.positive",
                scores=output_scores,
                threshold=output_threshold,
            ),
            negative=replace_region(
                group.negative,
                role=f"motor.{group_index}.negative",
                scores=output_scores,
                threshold=output_threshold,
            ),
        ))

    resolved_map = replace(
        channel_map,
        position_channels=position,
        force_channels=force,
        phase_channels=phase,
        contact_channels=contact,
        motor_groups=tuple(motor_groups),
        positive_feedback_channels=positive_feedback,
        negative_feedback_channels=negative_feedback,
        reserve_channels=tuple(sorted(reserve)),
        version=f"{channel_map.version}_calibrated",
    )
    validate_contact_channel_map(resolved_map)
    return resolved_map, ChannelHealthReport(
        passed=True,
        remap_allowed=True,
        input_threshold=float(input_threshold),
        output_threshold=float(output_threshold),
        unhealthy_input_channels=unhealthy_inputs,
        unhealthy_motor_channels=unhealthy_motor,
        replacements=tuple(replacements),
        reason="reserve_remap_passed_calibration",
    )


def require_channel_health(report: ChannelHealthReport) -> None:
    """Fail before task execution when the calibrated map is not viable."""

    if report.passed:
        return
    raise RuntimeError(
        "CL1 channel-health gate failed before task execution: "
        f"input={list(report.unhealthy_input_channels)}, "
        f"motor={list(report.unhealthy_motor_channels)}"
    )


def _score_vector(value: np.ndarray, name: str) -> np.ndarray:
    scores = np.asarray(value, dtype=np.float64)
    if scores.shape != (64,):
        raise ValueError(f"{name} must have shape (64,)")
    return np.where(np.isfinite(scores), scores, float("-inf"))
