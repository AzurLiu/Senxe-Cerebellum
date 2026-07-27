from types import SimpleNamespace

import numpy as np

from core import spike_pipeline
from core.channel_map import STIMULATABLE_CHANNELS
from core.neurons import timestamped_contact_calibration


class _FakeNeurons:
    def __init__(self):
        self.stimulated_channels = []

    def stim(self, channels, design, burst):
        selected = np.flatnonzero(
            np.asarray(channels._channels, dtype=bool)
        ).astype(int)
        self.stimulated_channels.extend(selected.tolist())


class _FakeSpikeReader:
    def __init__(self, neurons, config):
        self.neurons = neurons
        self.config = config

    def read(self, **kwargs):
        return SimpleNamespace(
            features=SimpleNamespace(
                channel_counts=tuple(np.zeros(64, dtype=int))
            )
        )


def test_timestamped_calibration_probes_only_stimulatable_channels(
    monkeypatch,
):
    neurons = _FakeNeurons()
    monkeypatch.setattr(
        spike_pipeline,
        "SpikeWindowReader",
        _FakeSpikeReader,
    )

    calibration = timestamped_contact_calibration(
        neurons,
        duration_sec=0.1,
    )

    assert neurons.stimulated_channels == list(STIMULATABLE_CHANNELS)
    assert calibration.channel_ranking.shape == (64,)
    assert calibration.output_responsiveness.shape == (64,)
    assert calibration.input_responsiveness.shape == (64,)
    assert calibration.evoked_counts_by_input.shape == (64, 64)
    assert np.isfinite(calibration.input_responsiveness).all()
