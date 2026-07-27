import numpy as np
import pytest

from core.stim_responsive_source import (
    StimResponsiveConfig,
    StimResponsiveModel,
)


def test_stimulation_creates_delayed_evoked_spikes():
    config = StimResponsiveConfig(
        seed=7,
        baseline_rate_hz=0.0,
        evoked_gain_hz=3000.0,
        response_delay_ms=5.0,
        response_tau_ms=50.0,
        artifact_rate_hz=0.0,
    )
    model = StimResponsiveModel(config)
    model.on_stim(timestamp=0, channel=8, strength=1.0)

    batch = model.read(0, 5000)

    assert batch.frames.shape == (5000, 64)
    assert batch.frames.dtype == np.int16
    assert batch.spikes
    delay_frames = int(config.response_delay_ms * 25)
    assert min(spike.timestamp for spike in batch.spikes) >= delay_frames
    assert any(spike.channel == 8 for spike in batch.spikes)


def test_feedback_gain_is_bounded_and_changes_response_model_state():
    model = StimResponsiveModel(StimResponsiveConfig(plasticity_rate=0.2))

    model.apply_feedback(+1)
    assert model.feedback_gain == pytest.approx(1.2)
    for _ in range(20):
        model.apply_feedback(+1)
    assert model.feedback_gain == pytest.approx(1.5)
    for _ in range(20):
        model.apply_feedback(-1)
    assert model.feedback_gain == pytest.approx(0.5)


def test_model_rejects_nonsequential_reads():
    model = StimResponsiveModel()
    model.read(100, 10)

    with pytest.raises(ValueError, match="sequential"):
        model.read(0, 10)

