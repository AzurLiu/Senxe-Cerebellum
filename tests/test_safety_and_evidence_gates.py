"""Regression tests for the safety and evidence-chain gates.

Each test pins one failure that was possible before: a retract command that
pushed into the contact, simulator output recorded as hardware evidence,
unaudited stimulation entry points, a channel-health gate with no
statistical meaning, a timing gate that passed when timestamps were missing,
a weighted stateful decoder behind a "zero-bias fixed" label, and evidence
lost when a run crashed mid-condition.
"""

from types import SimpleNamespace

import csv
import sys

import numpy as np
import pytest

import analyze_ablations
import core.neurons as neurons_module
import run_ablation_benchmark as benchmark
import senxe_demo_robosuite as demo
from core import spike_pipeline
from core.channel_health import (
    health_thresholds_from_env,
    resolve_contact_channel_map,
)
from core.channel_map import DEFAULT_CONTACT_CHANNEL_MAP, STIMULATABLE_CHANNELS
from core.contact_skill import ContactSkillController
from core.neurons import (
    SIMULATOR_ENV_VAR,
    BudgetedNeurons,
    ChannelSet,
    BurstDesign,
    StimDesign,
    cl_open,
    evidence_backend,
    timestamped_contact_calibration,
)


# ── 1. force frame ───────────────────────────────────────────────────────

@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_wrist_force_is_world_frame_tared_and_retract_leaves_the_table():
    env, raw_env = demo.make_robosuite_env(render=False)
    try:
        obs, _ = env.reset(seed=42)
        idle = np.zeros(env.action_space.shape[0])
        for _ in range(30):
            obs, *_ = env.step(idle)
        free = demo.extract_obs(obs, raw_env=raw_env)["force"]
        # Free space must not look like contact: the gripper's own weight
        # (about 5 N) is removed.
        assert np.linalg.norm(free) < 0.5

        press = idle.copy()
        press[2] = -1.0
        press[-1] = 1.0
        force = free
        for _ in range(200):
            obs, *_ = env.step(press)
            force = demo.extract_obs(obs, raw_env=raw_env)["force"]
            if np.linalg.norm(force) > 8.0:
                break
        assert np.linalg.norm(force) > 8.0, "gripper never reached the table"

        controller = ContactSkillController()
        retract_min = controller.config.retract_min_force_n
        assert np.linalg.norm(force) >= retract_min
        # The deterministic retract direction is -force. Pressing down on
        # the table must produce an upward (+z, world frame) retract.
        retract_direction = -force / np.linalg.norm(force)
        assert retract_direction[2] > 0.5
    finally:
        env.close()


# ── 2. backend label ─────────────────────────────────────────────────────

def test_simulator_sdk_cannot_be_recorded_as_hardware(monkeypatch):
    assert evidence_backend() == "simulator"
    monkeypatch.delenv(SIMULATOR_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match="replay simulator"):
        evidence_backend()
    monkeypatch.setattr(neurons_module, "sdk_reports_simulator", lambda: False)
    assert evidence_backend() == "hardware"


def test_evidence_table_carries_backend_and_refuses_mixing(tmp_path):
    assert "Backend" in benchmark.CSV_FIELDS
    path = tmp_path / "results.csv"
    row = {field: "" for field in benchmark.CSV_FIELDS}
    row.update({
        "RunID": "c1:contact_skill",
        "Condition": "contact_skill",
        "Episode": 0,
        "Backend": "simulator",
    })
    benchmark._append_rows(path, [row])
    other = dict(row, Episode=1, Backend="hardware")
    with pytest.raises(RuntimeError, match="Backend"):
        benchmark._append_rows(path, [other])


def _evidence_rows(backend: str) -> list[dict]:
    rows = []
    for condition, success in (("contact_skill", 1), ("baseline_only", 0)):
        for culture in range(6):
            for episode in range(4):
                rows.append({
                    "Backend": backend,
                    "ChannelMapHash": "m",
                    "ChannelMapVersion": "v",
                    "ChannelHealthReport": "{}",
                    "CodeHash": "c",
                    "Condition": condition,
                    "CultureID": f"{condition}_{culture}",
                    "Episode": episode,
                    "EpisodeForceSafe": 1,
                    "EpisodeSuccess": success,
                    "ProtocolHash": "p",
                    "ProtocolPhase": "retention_15_min",
                    "RuntimeVersions": "r",
                    "StimSafetyLimits": "{}",
                    "EpisodeStimAbsChargeNc": 0,
                    "EpisodeStimCalls": 0,
                    "EpisodeStimChannelPulses": 0,
                })
    return rows


@pytest.mark.parametrize(
    "backend, expected",
    [("simulator", "0"), ("hardware", "1")],
)
def test_confirmatory_gates_only_accept_hardware_rows(
    tmp_path,
    monkeypatch,
    backend,
    expected,
):
    rows = _evidence_rows(backend)
    source = tmp_path / "in.csv"
    with source.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    out = tmp_path / "analysis"
    monkeypatch.setattr(
        sys,
        "argv",
        ["analyze_ablations.py", str(source), "--output-dir", str(out)],
    )
    analyze_ablations.main()

    with (out / "culture_level_summary.csv").open() as handle:
        summary = list(csv.DictReader(handle))
    assert {row["ConfirmatoryValid"] for row in summary} == {expected}
    with (out / "preregistered_comparisons.csv").open() as handle:
        comparisons = list(csv.DictReader(handle))
    baseline = [row for row in comparisons if row["Control"] == "baseline_only"]
    assert baseline
    assert {row["PassesFalsificationGate"] for row in baseline} == {expected}
    assert {row["Backend"] for row in comparisons} == {backend}


# ── 3. stimulation gate cannot be bypassed ───────────────────────────────

class _RawNeurons:
    def __init__(self, timestamp=100):
        self.calls = 0
        self._timestamp = timestamp

    def stim(self, channels, design, burst):
        self.calls += 1

    def timestamp(self):
        if self._timestamp is None:
            raise RuntimeError("device clock unavailable")
        return self._timestamp

    def interrupt_then_stim(self, *args):
        self.calls += 1

    def create_stim_plan(self):
        return object()

    def get_frames_per_second(self):
        return 25_000


@pytest.mark.parametrize(
    "name",
    ["interrupt_then_stim", "create_stim_plan", "stimulate"],
)
def test_proxy_refuses_unaudited_stimulation_entry_points(name):
    neurons = BudgetedNeurons(_RawNeurons())
    with pytest.raises(AttributeError, match="audited stim"):
        getattr(neurons, name)
    assert neurons.get_frames_per_second() == 25_000


def test_proxy_is_never_double_wrapped():
    neurons = BudgetedNeurons(_RawNeurons())
    with pytest.raises(TypeError, match="already"):
        BudgetedNeurons(neurons)


def test_cl_open_yields_the_audited_proxy_and_blocks_bad_channels(
    monkeypatch,
):
    from contextlib import contextmanager

    raw = _RawNeurons()

    @contextmanager
    def _fake_open():
        yield raw

    monkeypatch.setattr(neurons_module._cl_sdk, "open", _fake_open)
    with cl_open() as neurons:
        assert isinstance(neurons, BudgetedNeurons)
        with pytest.raises(RuntimeError, match="non-stimulatable"):
            neurons.stim(
                ChannelSet(0),
                StimDesign(160, -0.5, 160, 0.5),
                BurstDesign(1, 100),
            )
        with pytest.raises(RuntimeError, match="amplitude"):
            neurons.stim(
                ChannelSet(1),
                StimDesign(160, -3.0, 160, 3.0),
                BurstDesign(1, 100),
            )
    assert raw.calls == 0


def test_cl_open_requires_lab_approval_before_opening_hardware(monkeypatch):
    monkeypatch.delenv(SIMULATOR_ENV_VAR, raising=False)
    monkeypatch.delenv("SENXE_LAB_APPROVED_STIM", raising=False)
    opened = []
    monkeypatch.setattr(
        neurons_module._cl_sdk,
        "open",
        lambda: opened.append(1),
    )
    with pytest.raises(RuntimeError, match="blocked"):
        with cl_open():
            pass
    assert not opened


# ── 4. channel-health statistics ─────────────────────────────────────────

class _ScriptedReader:
    """Poisson background; probing a 'live' input adds network spikes."""

    live_inputs: set = set()
    rng = None

    def __init__(self, neurons, config):
        self.neurons = neurons

    def read(self, **kwargs):
        counts = self.rng.poisson(2.0, size=64)
        if kwargs.get("trigger_stim_timestamps") is not None:
            if self.neurons.last_channel in self.live_inputs:
                counts = counts + 3
            self.neurons.last_channel = None
        return SimpleNamespace(
            features=SimpleNamespace(channel_counts=tuple(counts)),
            timing_valid=True,
        )


class _ProbeNeurons:
    last_stim_timestamp = 1
    last_channel = None

    def stim(self, channels, design, burst):
        self.last_channel = int(
            np.flatnonzero(np.asarray(channels._channels, dtype=bool))[0]
        )


def test_health_gate_separates_dead_inputs_from_live_ones(monkeypatch):
    inputs = (
        DEFAULT_CONTACT_CHANNEL_MAP.sensory_channels
        + DEFAULT_CONTACT_CHANNEL_MAP.feedback_channels
    )
    dead = set(inputs[:3])
    _ScriptedReader.live_inputs = set(STIMULATABLE_CHANNELS) - dead
    _ScriptedReader.rng = np.random.default_rng(0)
    monkeypatch.setattr(spike_pipeline, "SpikeWindowReader", _ScriptedReader)

    calibration = timestamped_contact_calibration(
        _ProbeNeurons(),
        duration_sec=10.0,
    )
    live = sorted(set(inputs) - dead)
    assert calibration.input_response_z[live].min() > 3.0
    assert np.abs(calibration.input_response_z[sorted(dead)]).max() < 3.0

    _, report = resolve_contact_channel_map(
        calibration.input_response_z,
        calibration.output_response_z,
        input_threshold=3.0,
        output_threshold=-np.inf,
    )
    assert not report.passed
    assert set(report.unhealthy_input_channels) == dead


def test_hardware_health_thresholds_have_no_default(monkeypatch):
    for name in (
        "SENXE_MIN_INPUT_RESPONSE_Z",
        "SENXE_MIN_OUTPUT_RESPONSE_Z",
        "SENXE_MIN_INPUT_RESPONSE_DELTA",
    ):
        monkeypatch.delenv(name, raising=False)
    assert health_thresholds_from_env(simulator=True) == (0.0, 0.0)
    with pytest.raises(RuntimeError, match="pre-frozen"):
        health_thresholds_from_env(simulator=False)
    monkeypatch.setenv("SENXE_MIN_INPUT_RESPONSE_Z", "3")
    monkeypatch.setenv("SENXE_MIN_OUTPUT_RESPONSE_Z", "2.5")
    assert health_thresholds_from_env(simulator=False) == (3.0, 2.5)
    monkeypatch.setenv("SENXE_MIN_INPUT_RESPONSE_DELTA", "0")
    with pytest.raises(RuntimeError, match="retired"):
        health_thresholds_from_env(simulator=False)


# ── 5. timing gate with missing timestamps ───────────────────────────────

def test_missing_stim_timestamp_stops_hardware_but_not_simulator(monkeypatch):
    design = StimDesign(160, -0.5, 160, 0.5)
    simulated = BudgetedNeurons(_RawNeurons(timestamp=None))
    simulated.stim(ChannelSet(1), design, BurstDesign(1, 100))
    assert simulated.last_stim_timestamp is None

    monkeypatch.delenv(SIMULATOR_ENV_VAR, raising=False)
    hardware = BudgetedNeurons(_RawNeurons(timestamp=None))
    with pytest.raises(RuntimeError, match="timestamp"):
        hardware.stim(ChannelSet(1), design, BurstDesign(1, 100))
    # The delivered dose is still accounted for.
    assert hardware.stim_calls == 1


# ── 6. decoder matches its documented policy ─────────────────────────────

def test_contact_decoder_is_unweighted_and_stateless():
    source = open(demo.__file__, encoding="utf-8").read()
    assert "channel_weights=resp_weights" not in source

    from core.decoder import AntagonisticDecoder

    decoder = AntagonisticDecoder(
        5,
        ema_alpha=1.0,
        action_scale=1.0,
        channel_weights=None,
        channel_groups=DEFAULT_CONTACT_CHANNEL_MAP.decoder_groups(),
    )
    positive, negative = DEFAULT_CONTACT_CHANNEL_MAP.decoder_groups()[0]
    balanced = np.zeros(64)
    balanced[list(positive)] = 2
    balanced[list(negative)] = 2
    one_sided = np.zeros(64)
    one_sided[list(positive)] = 2

    assert decoder.decode_counts(one_sided)[0] == pytest.approx(1.0, abs=1e-5)
    # No carry-over from the previous step and no sign bias at balance.
    assert decoder.decode_counts(balanced)[0] == pytest.approx(0.0, abs=1e-9)


# ── 7. crash-safe evidence ───────────────────────────────────────────────

def test_started_run_blocks_culture_reuse_even_without_episode_rows(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(benchmark, "RECORD_CL_SESSION", True)
    output = tmp_path / "results.csv"
    benchmark._record_run_started(
        output,
        culture_id="culture_1",
        sequence_index=0,
        conditions=["baseline_only"],
        backend="hardware",
    )
    assert not output.exists()
    args = SimpleNamespace(
        culture_id="culture_1",
        condition="contact_skill",
        allow_hardware_crossover=False,
        yoked_feedback=None,
        output=output,
    )
    with pytest.raises(RuntimeError, match="already appears"):
        benchmark._validate_hardware_scope(args, simulator=False)

    args.condition = "baseline_only"
    benchmark._validate_hardware_scope(args, simulator=False)


def test_episode_rows_survive_a_crash_in_a_later_episode(
    tmp_path,
    monkeypatch,
):
    class _CrashingAgent:
        active_scenario = None
        recorded_feedback_schedule = {0: ()}

        def __init__(self, *args, **kwargs):
            self.session_recorder = SimpleNamespace(
                start=lambda: None,
                stop=lambda: None,
            )
            self.current_protocol_phase = SimpleNamespace(
                phase=SimpleNamespace(value="feedback_training"),
                rest_before_seconds=0,
            )
            self.last_episode_control_summary = {
                "residual_applied_rate": 0.0,
                "mean_abs_applied_residual": 0.0,
                "hard_stop_count": 0,
            }
            self.last_episode_metrics = {
                key: 0 for key in (
                    "episode_success", "episode_force_safe", "peak_force_n",
                    "p95_force_n", "peak_torque_nm", "p95_torque_nm",
                    "final_nut_peg_xy_error_m", "final_nut_peg_distance_m",
                    "final_placement_depth_margin_m",
                    "final_nut_peg_yaw_error_deg", "final_grasp_confirmed",
                    "max_stim_calls", "max_channel_pulses", "stim_calls",
                    "channel_pulses", "abs_charge_nc", "episode_stim_calls",
                    "episode_channel_pulses", "episode_abs_charge_nc",
                )
            }
            self.last_episode_metrics.update({
                "time_to_success_steps": None,
                "health_min": None,
                "health_mean": None,
                "health_active_channels": None,
            })

        def run_episode(self, *, ep_num, **kwargs):
            if ep_num == 2:
                raise RuntimeError("stimulation budget exhausted")
            return 0.0, 0.0, [], 0.0, 100.0

    monkeypatch.setattr(benchmark, "CL1Agent", _CrashingAgent)
    monkeypatch.setattr(
        benchmark,
        "make_robosuite_env",
        lambda render=False: (SimpleNamespace(close=lambda: None), None),
    )
    output = tmp_path / "results.csv"
    partial = []
    args = SimpleNamespace(
        culture_id="culture_1",
        sequence_index=0,
        episodes=5,
        output=output,
        yoked_feedback=None,
    )
    provenance = benchmark.project_provenance()
    with pytest.raises(RuntimeError, match="budget exhausted"):
        benchmark._run_condition(
            args=args,
            neurons=None,
            ranking=np.arange(64),
            responsiveness=np.zeros(64),
            condition_name="contact_skill",
            condition_order=["contact_skill"],
            yoked_schedule={},
            channel_map=DEFAULT_CONTACT_CHANNEL_MAP,
            channel_health_report=SimpleNamespace(to_dict=lambda: {}),
            provenance=provenance,
            backend="simulator",
            on_feedback_schedule=partial.append,
        )

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["Episode"] for row in rows] == ["0", "1"]
    assert {row["Backend"] for row in rows} == {"simulator"}
    assert partial == [{0: ()}]
