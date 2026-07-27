from types import SimpleNamespace

import numpy as np
import pytest

import senxe_demo_robosuite as demo
from core.channel_map import DEFAULT_CONTACT_CHANNEL_MAP


class _TimestampedFakeNeurons:
    """Small CL loop double that preserves frame timestamps end to end."""

    def __init__(self):
        self.cursor = 0
        self.stim_calls = 0

    def get_frames_per_second(self):
        return 25_000

    def timestamp(self):
        return self.cursor

    def stim(self, channels, design, burst):
        self.stim_calls += 1

    def loop(self, **kwargs):
        ticks = []
        frames_per_tick = int(
            round(25_000 / kwargs["ticks_per_second"])
        )
        motor_channel = (
            DEFAULT_CONTACT_CHANNEL_MAP.motor_groups[0].positive[0]
        )
        for tick_index in range(kwargs["stop_after_ticks"]):
            timestamp = self.cursor
            self.cursor += frames_per_tick
            spikes = []
            if tick_index >= 5:
                spikes.append(SimpleNamespace(
                    timestamp=timestamp + 1,
                    channel=motor_channel,
                ))
            ticks.append(SimpleNamespace(
                timestamp=timestamp,
                frames=np.zeros(
                    (frames_per_tick, 64),
                    dtype=np.int16,
                ),
                analysis=SimpleNamespace(spikes=spikes, stims=[]),
            ))
        return iter(ticks)

    def get_health(self):
        return np.ones(64)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_default_contact_agent_runs_audited_loop_without_legacy_components():
    pytest.importorskip("robosuite")
    env, raw_env = demo.make_robosuite_env(render=False)
    neurons = _TimestampedFakeNeurons()
    try:
        agent = demo.CL1Agent(
            env,
            raw_env,
            neurons,
            channel_ranking=np.arange(64),
            responsiveness=np.ones(64),
        )

        for legacy_attribute in (
            "vie",
            "legacy_decoder",
            "residual_controller",
            "pdi",
            "curiosity",
        ):
            assert not hasattr(agent, legacy_attribute)

        agent.run_episode(max_steps=2, ep_num=0)

        assert agent.last_episode_metrics["step_count"] == 2
        assert agent.last_episode_metrics["episode_stim_calls"] > 0
        assert agent.last_episode_metrics["episode_abs_charge_nc"] > 0
        assert agent.last_spike_window["timing_valid"] is True
        assert agent.last_spike_window["trigger_stim_timestamps"]
        assert neurons.stim_calls > 0
    finally:
        env.close()
