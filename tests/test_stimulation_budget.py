import pytest

from core.neurons import (
    BudgetedNeurons,
    BurstDesign,
    ChannelSet,
    StimSafetyLimits,
    StimDesign,
    stimulation_budget_delta,
)


class _FakeNeurons:
    def __init__(self):
        self.calls = 0
        self.timestamp_value = 100

    def stim(self, channels, design, burst):
        self.calls += 1

    def timestamp(self):
        self.timestamp_value += 1
        return self.timestamp_value


class _LegacyBurst:
    """Burst shape used by CL SDK 0.1.x."""

    def __init__(self, burst_count, burst_hz):
        self._burst_count = burst_count
        self._burst_hz = burst_hz


class _CurrentBurst:
    """Burst shape used by CL SDK 1.x."""

    def __init__(self, burst_count, burst_hz):
        self._burst_count = burst_count
        self._burst_requested_hz = burst_hz


@pytest.mark.parametrize(
    "burst",
    [_LegacyBurst(1, 100), _CurrentBurst(1, 100)],
)
def test_stimulation_proxy_supports_cl_sdk_burst_field_layouts(burst):
    raw = _FakeNeurons()
    neurons = BudgetedNeurons(raw)

    neurons.stim(
        ChannelSet(1),
        StimDesign(160, -0.5, 160, 0.5),
        burst,
    )

    assert raw.calls == 1


def test_stimulation_budget_counts_channel_pulses_and_stops_before_limit():
    raw = _FakeNeurons()
    neurons = BudgetedNeurons(
        raw,
        max_stim_calls=2,
        max_channel_pulses=4,
    )
    channels = ChannelSet(1, 2)
    design = StimDesign(160, -0.5, 160, 0.5)

    neurons.stim(channels, design, BurstDesign(2, 100))
    snapshot = neurons.budget_snapshot()
    assert snapshot["channel_pulses"] == 4
    assert snapshot["abs_charge_nc"] == pytest.approx(0.64)
    assert snapshot["last_stim_timestamp"] == 101
    with pytest.raises(RuntimeError, match="channel-pulse"):
        neurons.stim(channels, design, BurstDesign(1, 100))
    assert raw.calls == 1


@pytest.mark.parametrize("channel", [0, 4, 7, 56, 63])
def test_stimulation_proxy_rejects_non_stimulatable_cl1_channels(channel):
    raw = _FakeNeurons()
    neurons = BudgetedNeurons(
        raw,
        max_stim_calls=2,
        max_channel_pulses=4,
    )
    design = StimDesign(160, -0.5, 160, 0.5)

    with pytest.raises(RuntimeError, match="non-stimulatable"):
        neurons.stim(
            ChannelSet(channel),
            design,
            BurstDesign(1, 100),
        )
    assert raw.calls == 0


@pytest.mark.parametrize(
    ("design", "burst", "match"),
    [
        (
            StimDesign(160, -1.6, 160, 1.6),
            BurstDesign(1, 100),
            "amplitude",
        ),
        (
            StimDesign(220, -0.5, 220, 0.5),
            BurstDesign(1, 100),
            "phase-width",
        ),
        (
            StimDesign(160, -0.5, 160, 0.5),
            _CurrentBurst(1, 201),
            "frequency",
        ),
        (
            StimDesign(160, -0.5, 160, 0.5),
            BurstDesign(16, 100),
            "burst-count",
        ),
        (
            StimDesign(160, -0.5, 160, 0.2),
            BurstDesign(1, 100),
            "charge-balance",
        ),
    ],
)
def test_stimulation_proxy_rejects_out_of_envelope_designs(
    design,
    burst,
    match,
):
    raw = _FakeNeurons()
    neurons = BudgetedNeurons(
        raw,
        safety_limits=StimSafetyLimits(
            max_amplitude_ua=1.5,
            max_phase_width_us=200,
            max_burst_hz=200,
            max_burst_count=15,
            max_stim_calls=100,
            max_channel_pulses=100,
            max_abs_charge_nc=10.0,
        ),
    )

    with pytest.raises(RuntimeError, match=match):
        neurons.stim(ChannelSet(1), design, burst)
    assert raw.calls == 0


def test_absolute_charge_budget_and_interval_delta_are_enforced():
    raw = _FakeNeurons()
    neurons = BudgetedNeurons(
        raw,
        safety_limits=StimSafetyLimits(
            max_stim_calls=10,
            max_channel_pulses=10,
            max_abs_charge_nc=0.33,
        ),
    )
    start = neurons.budget_snapshot()
    design = StimDesign(160, -0.5, 160, 0.5)

    neurons.stim(ChannelSet(1), design, BurstDesign(2, 100))
    end = neurons.budget_snapshot()
    assert stimulation_budget_delta(start, end) == {
        "stim_calls": 1,
        "channel_pulses": 2,
        "abs_charge_nc": pytest.approx(0.32),
    }
    with pytest.raises(RuntimeError, match="absolute-charge"):
        neurons.stim(ChannelSet(1), design, BurstDesign(1, 100))
    assert raw.calls == 1
