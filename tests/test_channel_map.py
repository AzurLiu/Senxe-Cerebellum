import numpy as np

from core.channel_map import (
    DEFAULT_CONTACT_CHANNEL_MAP,
    MOTOR_OUTPUT_NAMES,
    NON_STIMULATABLE_CHANNELS,
    STIMULATABLE_CHANNELS,
    channel_map_metadata,
    validate_contact_channel_map,
)
from core.decoder import AntagonisticDecoder


def test_default_contact_map_has_18_20_6_15_disjoint_allocation():
    channel_map = DEFAULT_CONTACT_CHANNEL_MAP

    validate_contact_channel_map(channel_map)
    assert len(channel_map.sensory_channels) == 18
    assert len(channel_map.motor_channels) == 20
    assert len(channel_map.feedback_channels) == 6
    assert len(channel_map.reserve_channels) == 15
    assert len(channel_map.motor_groups) == len(MOTOR_OUTPUT_NAMES) == 5

    regions = (
        set(channel_map.sensory_channels),
        set(channel_map.motor_channels),
        set(channel_map.feedback_channels),
        set(channel_map.reserve_channels),
    )
    assert set().union(*regions) == set(STIMULATABLE_CHANNELS)
    assert all(
        not (left & right)
        for index, left in enumerate(regions)
        for right in regions[index + 1:]
    )
    assert not set(NON_STIMULATABLE_CHANNELS) & set().union(*regions)


def test_channel_map_metadata_is_stable_and_auditable():
    first = channel_map_metadata()
    second = channel_map_metadata()

    assert first == second
    assert len(first["channel_map_hash"]) == 64
    assert first["channel_map"]["sensory_channel_count"] == 18
    assert first["channel_map"]["reserve_channel_count"] == 15


def test_explicit_contact_decoder_uses_only_its_intended_motor_group():
    channel_map = DEFAULT_CONTACT_CHANNEL_MAP
    decoder = AntagonisticDecoder(
        action_dim=5,
        ema_alpha=1.0,
        action_scale=1.0,
        channel_groups=channel_map.decoder_groups(),
    )

    for output_index, group in enumerate(channel_map.motor_groups):
        positive_counts = np.zeros(64)
        positive_counts[group.positive[0]] = 3
        positive = decoder.decode_counts(positive_counts)
        decoder.reset()

        negative_counts = np.zeros(64)
        negative_counts[group.negative[0]] = 3
        negative = decoder.decode_counts(negative_counts)
        decoder.reset()

        assert positive[output_index] > 0.99
        assert negative[output_index] < -0.99
        assert np.allclose(
            np.delete(positive, output_index),
            0.0,
        )
        assert np.allclose(
            np.delete(negative, output_index),
            0.0,
        )


def test_contact_decoder_ignores_sensory_feedback_and_reserve_spikes():
    channel_map = DEFAULT_CONTACT_CHANNEL_MAP
    decoder = AntagonisticDecoder(
        action_dim=5,
        ema_alpha=1.0,
        action_scale=1.0,
        channel_groups=channel_map.decoder_groups(),
    )
    counts = np.zeros(64)
    ignored_channels = (
        channel_map.sensory_channels
        + channel_map.feedback_channels
        + channel_map.reserve_channels
    )
    counts[list(ignored_channels)] = 10

    assert np.allclose(decoder.decode_counts(counts), 0.0)
