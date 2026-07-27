from types import SimpleNamespace

import numpy as np

from core.neurons import timestamped_warmup_calibration


class _FakeTimestampedNeurons:
    def __init__(self):
        self.timestamp_value = 0
        self.probe_index = -1
        self.active_output = None

    def get_frames_per_second(self):
        return 1000

    def stim(self, channels, design, burst):
        self.probe_index += 1
        self.active_output = self.probe_index % 64

    def loop(self, **kwargs):
        ticks = []
        for tick_index in range(kwargs["stop_after_ticks"]):
            timestamp = self.timestamp_value
            self.timestamp_value += 10
            spikes = []
            # Default 50/50 ms reader: tick 5 begins collection.
            if tick_index == 5 and self.active_output is not None:
                spikes.append(SimpleNamespace(
                    timestamp=timestamp + 1,
                    channel=self.active_output,
                ))
            ticks.append(SimpleNamespace(
                timestamp=timestamp,
                frames=np.zeros((10, 64), dtype=np.int16),
                analysis=SimpleNamespace(spikes=spikes, stims=[]),
            ))
        return iter(ticks)


def test_timestamped_calibration_ranks_sdk_detected_spike_responses():
    neurons = _FakeTimestampedNeurons()

    ranking, responsiveness = timestamped_warmup_calibration(
        neurons,
        duration_sec=0.1,
    )

    assert ranking.shape == (64,)
    assert responsiveness.shape == (64,)
    assert np.isfinite(responsiveness).all()
    assert np.all(responsiveness > 0)
