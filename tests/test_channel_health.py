import numpy as np
import pytest

from core.channel_health import (
    require_channel_health,
    resolve_contact_channel_map,
)
from core.channel_map import (
    DEFAULT_CONTACT_CHANNEL_MAP,
    validate_contact_channel_map,
)


def test_channel_health_gate_passes_healthy_frozen_map():
    input_scores = np.ones(64)
    output_scores = np.ones(64)

    resolved, report = resolve_contact_channel_map(
        input_scores,
        output_scores,
    )

    assert resolved is DEFAULT_CONTACT_CHANNEL_MAP
    assert report.passed
    assert report.replacements == ()
    require_channel_health(report)


def test_channel_health_gate_fails_closed_without_remap_permission():
    input_scores = np.ones(64)
    output_scores = np.ones(64)
    failed = DEFAULT_CONTACT_CHANNEL_MAP.motor_groups[0].positive[0]
    output_scores[failed] = 0.0

    resolved, report = resolve_contact_channel_map(
        input_scores,
        output_scores,
    )

    assert resolved is DEFAULT_CONTACT_CHANNEL_MAP
    assert not report.passed
    assert report.unhealthy_motor_channels == (failed,)
    with pytest.raises(RuntimeError, match="channel-health"):
        require_channel_health(report)


def test_outcome_blinded_remap_swaps_failed_channel_with_best_reserve():
    input_scores = np.ones(64)
    output_scores = np.ones(64)
    failed = DEFAULT_CONTACT_CHANNEL_MAP.motor_groups[0].positive[0]
    reserve = DEFAULT_CONTACT_CHANNEL_MAP.reserve_channels
    output_scores[failed] = 0.0
    output_scores[list(reserve)] = np.linspace(2.0, 3.0, len(reserve))
    expected = reserve[-1]

    resolved, report = resolve_contact_channel_map(
        input_scores,
        output_scores,
        allow_reserve_remap=True,
    )

    validate_contact_channel_map(resolved)
    assert report.passed
    assert len(report.replacements) == 1
    assert report.replacements[0].original_channel == failed
    assert report.replacements[0].replacement_channel == expected
    assert expected in resolved.motor_channels
    assert failed in resolved.reserve_channels
    assert resolved.stable_hash() != DEFAULT_CONTACT_CHANNEL_MAP.stable_hash()


def test_remap_refuses_to_hide_insufficient_healthy_reserve():
    input_scores = np.ones(64)
    output_scores = np.zeros(64)

    with pytest.raises(RuntimeError, match="no healthy reserve"):
        resolve_contact_channel_map(
            input_scores,
            output_scores,
            allow_reserve_remap=True,
        )
