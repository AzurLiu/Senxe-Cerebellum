from types import SimpleNamespace

import numpy as np

from core.spike_pipeline import (
    SpikeWindowConfig,
    SpikeWindowReader,
    ablate_channel_counts,
)


class _FakeNeurons:
    def __init__(self, ticks):
        self.ticks = ticks
        self.loop_kwargs = None

    def get_frames_per_second(self):
        return 1000

    def loop(self, **kwargs):
        self.loop_kwargs = kwargs
        return iter(self.ticks)


def _tick(timestamp, spike_pairs=(), stim_timestamps=()):
    spikes = [
        SimpleNamespace(timestamp=ts, channel=channel)
        for ts, channel in spike_pairs
    ]
    stims = [
        SimpleNamespace(timestamp=ts, channel=0)
        for ts in stim_timestamps
    ]
    return SimpleNamespace(
        timestamp=timestamp,
        frames=np.zeros((10, 64), dtype=np.int16),
        analysis=SimpleNamespace(spikes=spikes, stims=stims),
    )


def test_timestamped_reader_excludes_artifact_ticks_and_builds_bins():
    neurons = _FakeNeurons([
        _tick(0, [(2, 1)], [1]),
        _tick(10, [(12, 2)]),
        _tick(20, [(21, 3), (29, 3)]),
        _tick(30, [(35, 4)]),
        _tick(40, [(49, 3)]),
    ])
    reader = SpikeWindowReader(
        neurons,
        SpikeWindowConfig(
            artifact_wait_ms=20,
            collect_window_ms=30,
            bin_width_ms=10,
            tick_ms=10,
        ),
    )

    window = reader.read(trigger_stim_timestamps=(0,))

    assert window.artifact_spike_count == 2
    assert window.collect_start_timestamp == 20
    assert window.collect_end_timestamp == 50
    assert [event.timestamp for event in window.events] == [21, 29, 35, 49]
    assert window.features.channel_counts[3] == 3
    assert window.features.channel_counts[4] == 1
    assert window.features.first_latency_ms[3] == 1.0
    assert window.features.time_bins[0][3] == 2
    assert window.features.time_bins[1][4] == 1
    assert window.features.time_bins[2][3] == 1
    assert neurons.loop_kwargs["stop_after_ticks"] == 5
    assert window.trigger_stim_timestamps == (0,)
    assert window.trigger_to_collect_ms == 20.0
    assert window.timing_valid
    assert window.observed_tick_count == window.expected_tick_count == 5


def test_zero_and_shuffled_ablations_preserve_expected_statistics():
    counts = np.arange(64, dtype=np.int64)
    zero = ablate_channel_counts(
        counts,
        "zero",
        rng=np.random.default_rng(1),
    )
    shuffled = ablate_channel_counts(
        counts,
        "shuffled",
        rng=np.random.default_rng(1),
    )

    assert np.count_nonzero(zero) == 0
    assert shuffled.sum() == counts.sum()
    assert sorted(shuffled.tolist()) == sorted(counts.tolist())
    assert not np.array_equal(shuffled, counts)


def test_subset_shuffle_preserves_motor_count_and_nonmotor_channels():
    counts = np.arange(64, dtype=np.int64)
    motor_channels = (13, 14, 41, 42)

    shuffled = ablate_channel_counts(
        counts,
        "shuffled",
        rng=np.random.default_rng(2),
        channel_subset=motor_channels,
    )

    assert shuffled[list(motor_channels)].sum() == counts[
        list(motor_channels)
    ].sum()
    nonmotor = sorted(set(range(64)) - set(motor_channels))
    assert np.array_equal(shuffled[nonmotor], counts[nonmotor])
    assert not np.array_equal(
        shuffled[list(motor_channels)],
        counts[list(motor_channels)],
    )
