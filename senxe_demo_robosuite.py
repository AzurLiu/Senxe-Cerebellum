#!/usr/bin/env python3
"""
Senxe Cerebellum v5.0 — RoboSuite Contact-Skill NutAssembly
=========================================================================
CL1 Bio-Computer — Bounded Multi-Axis Contact Skill

Usage:  python senxe_demo_robosuite.py
Output: cl1_nutassembly.mp4
"""
import os
import time
import numpy as np
from tqdm import tqdm
from collections import deque

from core.channel_map import (
    DEFAULT_CONTACT_CHANNEL_MAP,
    ContactChannelMap,
    channel_map_metadata,
)
from core.channel_health import (
    ChannelHealthReport,
    require_channel_health,
    resolve_contact_channel_map,
)
from core.neurons import (
    BudgetedNeurons,
    cl_open,
    is_cl_simulator,
    require_stimulation_approval,
    stimulation_budget_delta,
    stimulation_limits_from_env,
    timestamped_contact_calibration,
)
from core.decoder import AntagonisticDecoder
from core.hybrid_control import (
    NominalControlConfig,
    NominalTaskController,
    TaskPhase,
    spike_confidence,
)
from core.contact_scenarios import (
    ContactScenarioSchedule,
    RoboSuiteContactPerturbation,
    default_contact_scenario_path,
    load_contact_scenarios,
)
from core.contact_skill import (
    CONTACT_SKILL_DIM,
    ContactEncoderConfig,
    ContactFeedbackEvent,
    ContactFeedbackEvaluator,
    ContactFeedbackStimulator,
    ContactSkillController,
    ContactStateEncoder,
    summarize_contact_reports,
)
from core.learning_protocol import (
    LearningPhase,
    default_protocol_path,
    load_protocols,
)
from core.session_recording import CLSessionRecorder, SessionRecordingConfig
from core.provenance import project_provenance
from core.yoked_feedback import event_for_step
from core.spike_pipeline import (
    SpikeWindowConfig,
    SpikeWindowReader,
    ablate_channel_counts,
)
from core.video import save_video
from core.hud import draw_overlay, hud

# ═══ Configuration ═══
SEED            = 42
# The controller and perturbation suite operate on SquareNut / peg1 only.
# NutAssemblySquare makes the official task definition match that scope.
ENV_NAME        = "NutAssemblySquare"
ROBOT           = "Panda"
PROTOCOL_ID     = os.getenv(
    "SENXE_PROTOCOL_ID",
    "senxe_contact_skill_application_v1",
).strip()
EPISODES        = int(os.getenv(
    "SENXE_EPISODES",
    str(load_protocols(default_protocol_path())[PROTOCOL_ID].total_episodes),
))
MAX_STEPS       = 600
VIDEO_FPS       = 30
VIDEO_CL1       = "cl1_nutassembly.mp4"
RECORD_LAST_N   = int(os.getenv("SENXE_RECORD_LAST_N", "10"))
WARMUP_SECONDS  = 10
RENDER_W        = 720
RENDER_H        = 720
PLACEMENT_HEIGHT_THRESHOLD_M = 0.05
FORCE_SAFETY_THRESHOLD    = 20.0
TORQUE_SAFETY_THRESHOLD   = 5.0
CONTROL_MODE              = os.getenv("SENXE_CONTROL_MODE", "contact_skill").strip().lower()
SPIKE_PIPELINE            = os.getenv("SENXE_SPIKE_PIPELINE", "timestamped").strip().lower()
ARTIFACT_WAIT_OVERRIDE    = os.getenv("SENXE_ARTIFACT_WAIT_MS")
COLLECT_WINDOW_OVERRIDE   = os.getenv("SENXE_COLLECT_WINDOW_MS")
SPIKE_BIN_MS              = float(os.getenv("SENXE_SPIKE_BIN_MS", "10"))
SPIKE_TICK_MS             = float(os.getenv("SENXE_SPIKE_TICK_MS", "10"))
RECORD_CL_SESSION         = os.getenv("SENXE_RECORD_SESSION", "0").strip() == "1"
RECORDING_LOCATION        = os.getenv("SENXE_RECORDING_LOCATION") or None
GENERALIZATION_ENABLED    = os.getenv("SENXE_GENERALIZATION", "1").strip() == "1"
STIM_SAFETY_LIMITS        = stimulation_limits_from_env()
MAX_STIM_AMPLITUDE_UA     = STIM_SAFETY_LIMITS.max_amplitude_ua
MAX_STIM_CALLS            = STIM_SAFETY_LIMITS.max_stim_calls
MAX_STIM_CHANNEL_PULSES   = STIM_SAFETY_LIMITS.max_channel_pulses
MAX_STIM_ABS_CHARGE_NC    = STIM_SAFETY_LIMITS.max_abs_charge_nc
MIN_INPUT_RESPONSE_DELTA   = float(os.getenv(
    "SENXE_MIN_INPUT_RESPONSE_DELTA",
    "0",
))
MIN_OUTPUT_RESPONSE_DELTA  = float(os.getenv(
    "SENXE_MIN_OUTPUT_RESPONSE_DELTA",
    "0",
))
ALLOW_CALIBRATION_REMAP    = os.getenv(
    "SENXE_ALLOW_CALIBRATION_REMAP",
    "0",
).strip() == "1"

# ═══ RoboSuite Environment ═══
def make_robosuite_env(render=False):
    import robosuite as suite
    from robosuite.wrappers import GymWrapper
    raw = suite.make(ENV_NAME, robots=ROBOT, has_renderer=False,
                     has_offscreen_renderer=render, use_camera_obs=False,
                     render_camera="frontview", horizon=MAX_STEPS, reward_shaping=True)
    return GymWrapper(raw), raw

def extract_obs(obs, raw_env=None):
    if raw_env is None:
        raise RuntimeError("extract_obs requires raw_env")
    try:
        sim_data = raw_env.sim.data
        
        # EEF position and velocity (try different version-robust naming schemes)
        eef_site_names = ["gripper0_right_grip_site", "gripper0_grip_site", "right_grip_site"]
        eef, vel, eef_xmat = None, None, None
        for name in eef_site_names:
            try:
                eef = sim_data.get_site_xpos(name)
                vel = sim_data.get_site_xvelp(name)
                eef_xmat = sim_data.get_site_xmat(name)
                break
            except Exception:
                continue
        if eef is None or vel is None:
            raise RuntimeError(f"Could not find end-effector grip site in Mujoco simulation (tried: {eef_site_names})")

        # Active square-peg position.
        peg_body_names = ["peg1", "peg"]
        peg = None
        peg_quat = None
        for name in peg_body_names:
            try:
                if hasattr(sim_data, "body"):
                    peg = sim_data.body(name).xpos
                    peg_quat = sim_data.body(name).xquat
                else:
                    peg = sim_data.get_body_xpos(name)
                    peg_quat = sim_data.get_body_xquat(name)
                break
            except Exception:
                continue
        if peg is None:
            raise RuntimeError(f"Could not find peg body in Mujoco simulation (tried: {peg_body_names})")

        eef_to_peg = peg - eef
        
        frc_raw = getattr(raw_env.robots[0], "ee_force", np.zeros(3))
        trq_raw = getattr(raw_env.robots[0], "ee_torque", np.zeros(3))
        frc = frc_raw["right"] if isinstance(frc_raw, dict) and "right" in frc_raw else frc_raw
        trq = trq_raw["right"] if isinstance(trq_raw, dict) and "right" in trq_raw else trq_raw
        frc = np.array(frc).flatten()[:3]
        trq = np.array(trq).flatten()[:3]
        eef = np.array(eef).flatten()[:3]
        vel = np.array(vel).flatten()[:3]
        jnt = raw_env.robots[0]._joint_positions

        # Nut position (try different names)
        nut_body_names = ["SquareNut_main", "nut"]
        nut = None
        nut_quat = None
        for name in nut_body_names:
            try:
                if hasattr(sim_data, "body"):
                    nut = sim_data.body(name).xpos
                    nut_quat = sim_data.body(name).xquat
                else:
                    nut = sim_data.get_body_xpos(name)
                    nut_quat = sim_data.get_body_xquat(name)
                break
            except Exception:
                continue
        if nut is None:
            raise RuntimeError(f"Could not find nut body in Mujoco simulation (tried: {nut_body_names})")

        nut = np.asarray(nut, dtype=float).flatten()[:3]
        peg = np.asarray(peg, dtype=float).flatten()[:3]
        grasp_target = nut.copy()
        try:
            active_nut = raw_env.nuts[int(getattr(raw_env, "nut_id", 0))]
            handle_site = active_nut.important_sites["handle"]
            grasp_target = (
                sim_data.site(handle_site).xpos
                if hasattr(sim_data, "site")
                else sim_data.get_site_xpos(handle_site)
            )
            grasp_target = np.asarray(
                grasp_target,
                dtype=float,
            ).flatten()[:3]
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            pass
        e2n = grasp_target - eef
        nut_to_peg = peg - nut
        table_height = float(np.asarray(raw_env.table_offset)[2])
        nut_height_above_table = float(nut[2] - table_height)
        placement_depth_margin = float(
            PLACEMENT_HEIGHT_THRESHOLD_M - nut_height_above_table
        )
        yaw_error_rad = _square_symmetric_yaw_error_rad(
            nut_quat,
            peg_quat,
        )
        grasp_yaw_error_rad = _grasp_alignment_yaw_error_rad(
            nut_quat,
            eef_xmat,
        )
        yaw_error_deg = float(abs(np.degrees(yaw_error_rad)))
        grasp_confirmed = _square_nut_grasp_confirmed(raw_env)
        nut_on_peg = bool(raw_env.on_peg(
            nut,
            int(getattr(raw_env, "nut_id", 0)),
        ))
        placement_success = bool(raw_env._check_success())

        return dict(
            eef_pos=eef,
            eef_vel=vel,
            force=frc,
            torque=trq,
            # Compatibility alias for legacy modules: vector from EEF to peg.
            peg_to_hole=np.asarray(eef_to_peg).flatten()[:3],
            eef_to_peg=np.asarray(eef_to_peg).flatten()[:3],
            nut_to_peg=np.asarray(nut_to_peg).flatten()[:3],
            joint_pos=jnt,
            eef_to_nut=e2n,
            grasp_target_pos=grasp_target,
            nut_pos=nut,
            peg_pos=peg,
            nut_peg_xy_error_m=float(np.linalg.norm(nut_to_peg[:2])),
            nut_peg_distance_m=float(np.linalg.norm(nut_to_peg)),
            nut_height_above_table_m=nut_height_above_table,
            placement_depth_margin_m=placement_depth_margin,
            nut_peg_yaw_error_deg=yaw_error_deg,
            nut_peg_yaw_error_rad=yaw_error_rad,
            grasp_yaw_error_rad=grasp_yaw_error_rad,
            grasp_confirmed=grasp_confirmed,
            nut_on_peg=nut_on_peg,
            placement_success=placement_success,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to extract observation from raw_env: {e}") from e

def _square_nut_grasp_confirmed(raw_env) -> bool:
    """Return RoboSuite's contact-based grasp result for the active square nut."""

    try:
        nut_index = int(getattr(raw_env, "nut_id", 0))
        nut = raw_env.nuts[nut_index]
        arm = raw_env.robots[0].arms[0]
        gripper = raw_env.robots[0].gripper[arm]
        return bool(raw_env._check_grasp(
            gripper=gripper,
            object_geoms=nut.contact_geoms,
        ))
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False


def _square_symmetric_yaw_error_rad(nut_quat, peg_quat) -> float:
    """Return signed nut-to-peg yaw error modulo square symmetry."""

    if nut_quat is None or peg_quat is None:
        return float("nan")

    def yaw(quat) -> float:
        w, x, y, z = np.asarray(quat, dtype=float).flatten()[:4]
        return float(np.arctan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        ))

    delta = yaw(nut_quat) - yaw(peg_quat)
    return float(
        (delta + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0
    )


def _grasp_alignment_yaw_error_rad(nut_quat, eef_xmat) -> float:
    """Return signed gripper yaw error for closing across the nut handle."""

    if nut_quat is None or eef_xmat is None:
        return 0.0
    w, x, y, z = np.asarray(nut_quat, dtype=float).flatten()[:4]
    nut_yaw = float(np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    ))
    matrix = np.asarray(eef_xmat, dtype=float).reshape(3, 3)
    eef_x_yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    desired = nut_yaw + np.pi / 2.0
    return float((desired - eef_x_yaw + np.pi / 2.0) % np.pi - np.pi / 2.0)


def compute_placement_metrics(info):
    """Return an auditable placement margin and nut-to-peg distance."""

    return (
        float(info["placement_depth_margin_m"]),
        float(info["nut_peg_distance_m"]),
    )


def culture_health_metrics(neurons) -> dict:
    """Return simple, non-interpretive culture health telemetry."""

    get_health = getattr(neurons, "get_health", None)
    if not callable(get_health):
        return {
            "health_min": None,
            "health_mean": None,
            "health_active_channels": None,
        }
    try:
        values = np.asarray(get_health(), dtype=float).flatten()
    except (RuntimeError, TypeError, ValueError):
        values = np.asarray([], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "health_min": None,
            "health_mean": None,
            "health_active_channels": 0,
        }
    return {
        "health_min": float(np.min(values)),
        "health_mean": float(np.mean(values)),
        "health_active_channels": int(np.count_nonzero(values > 0.0)),
    }


def require_hardware_evidence_recording() -> None:
    if not is_cl_simulator() and not RECORD_CL_SESSION:
        raise RuntimeError(
            "Real CL1 experiments require SENXE_RECORD_SESSION=1 so raw "
            "samples, detected spikes, stimulations, and synchronized control "
            "telemetry are retained."
        )

# ═══ CL1 Biological Agent ═══
class CL1Agent:
    def __init__(self, env, raw_env, neurons, channel_ranking=None, responsiveness=None,
                 ablation_spike_mode="none", ablation_stim_mode="full",
                 control_mode=CONTROL_MODE, yoked_feedback_schedule=None,
                 experiment_metadata=None,
                 sensory_stimulation_enabled=True,
                 channel_map: ContactChannelMap | None = None,
                 channel_health_report: ChannelHealthReport | None = None,
                 protocol_id: str = PROTOCOL_ID):
        require_stimulation_approval()
        require_hardware_evidence_recording()
        self.env = env
        self.raw_env = raw_env
        self.neurons = (
            neurons
            if isinstance(neurons, BudgetedNeurons)
            else BudgetedNeurons(
                neurons,
                safety_limits=STIM_SAFETY_LIMITS,
            )
        )
        self.ablation_spike_mode = ablation_spike_mode
        self.ablation_stim_mode = ablation_stim_mode
        if self.ablation_stim_mode not in {"full", "none", "yoked"}:
            raise ValueError(
                "ablation_stim_mode must be full, none, or yoked"
            )
        self.yoked_feedback_schedule = dict(yoked_feedback_schedule or {})
        if (
            self.ablation_stim_mode == "yoked"
            and not self.yoked_feedback_schedule
        ):
            raise ValueError(
                "yoked feedback mode requires a donor feedback schedule"
            )
        self.recorded_feedback_schedule = {}
        self.sensory_stimulation_enabled = bool(
            sensory_stimulation_enabled
        )
        self.channel_map = channel_map or DEFAULT_CONTACT_CHANNEL_MAP
        self.channel_map_metadata = channel_map_metadata(self.channel_map)
        self.channel_health_report = channel_health_report
        self.experiment_metadata = {
            **dict(experiment_metadata or {}),
            **self.channel_map_metadata,
            "channel_health_report": (
                None
                if channel_health_report is None
                else channel_health_report.to_dict()
            ),
        }
        self.control_mode = str(control_mode).strip().lower()
        if self.control_mode != "contact_skill":
            raise ValueError(
                "The audited RoboSuite agent only supports contact_skill. "
                "Use senxe_demo.py for archived legacy demonstrations."
            )
        self.spike_pipeline = SPIKE_PIPELINE
        if self.spike_pipeline not in {"timestamped", "legacy_voltage"}:
            raise ValueError(
                "SENXE_SPIKE_PIPELINE must be timestamped or legacy_voltage"
            )
        protocols = load_protocols(default_protocol_path())
        if protocol_id not in protocols:
            raise ValueError(f"unknown protocol_id: {protocol_id}")
        self.protocol = protocols[protocol_id]
        self.current_protocol_phase = self.protocol.phase_for_episode(0)
        self._entered_protocol_phase = None
        self.last_rest_observation = None
        self.artifact_wait_ms = float(
            ARTIFACT_WAIT_OVERRIDE
            if ARTIFACT_WAIT_OVERRIDE is not None
            else self.protocol.artifact_wait_ms
        )
        self.collect_window_ms = float(
            COLLECT_WINDOW_OVERRIDE
            if COLLECT_WINDOW_OVERRIDE is not None
            else self.protocol.collect_window_ms
        )
        self.spike_reader = (
            SpikeWindowReader(
                neurons,
                SpikeWindowConfig(
                    artifact_wait_ms=self.artifact_wait_ms,
                    collect_window_ms=self.collect_window_ms,
                    bin_width_ms=SPIKE_BIN_MS,
                    tick_ms=SPIKE_TICK_MS,
                ),
            )
            if self.spike_pipeline == "timestamped"
            else None
        )
        self.last_spike_window = None
        self.rng = np.random.default_rng(SEED)
        self.session_recorder = CLSessionRecorder(
            self.neurons,
            SessionRecordingConfig(
                enabled=RECORD_CL_SESSION,
                file_location=RECORDING_LOCATION,
                file_suffix=str(
                    self.experiment_metadata.get(
                        "file_suffix",
                        "senxe_contact_skill_v2",
                    )
                ),
            ),
            attributes={
                **self.experiment_metadata,
                "protocol_id": self.protocol.protocol_id,
                "control_mode": self.control_mode,
                "spike_pipeline": self.spike_pipeline,
                "sensory_stimulation_enabled": (
                    self.sensory_stimulation_enabled
                ),
                "max_stim_amplitude_ua": MAX_STIM_AMPLITUDE_UA,
                "max_stim_calls": MAX_STIM_CALLS,
                "max_stim_channel_pulses": MAX_STIM_CHANNEL_PULSES,
                "stimulation_safety_limits": (
                    STIM_SAFETY_LIMITS.to_dict()
                ),
            },
        )
        self.action_dim = env.action_space.shape[0]
        self.contact_encoder = ContactStateEncoder(
            self.neurons,
            ContactEncoderConfig(
                max_stim_amplitude=MAX_STIM_AMPLITUDE_UA,
                position_channels=self.channel_map.position_channels,
                force_channels=self.channel_map.force_channels,
                phase_channels=self.channel_map.phase_channels,
                contact_channels=self.channel_map.contact_channels,
            ),
        )
        resp_weights = responsiveness if channel_ranking is not None else None
        self.contact_decoder = AntagonisticDecoder(
            CONTACT_SKILL_DIM,
            action_scale=1.0,
            channel_weights=resp_weights,
            channel_groups=self.channel_map.decoder_groups(),
        )
        self.decoder = self.contact_decoder
        self.nominal_controller = NominalTaskController(NominalControlConfig(
            action_dim=self.action_dim,
        ))
        self.contact_controller = ContactSkillController()
        self.contact_feedback = ContactFeedbackEvaluator()
        self.contact_feedback_stimulator = ContactFeedbackStimulator(
            self.neurons,
            positive_channels=(
                self.channel_map.positive_feedback_channels
            ),
            negative_channels=(
                self.channel_map.negative_feedback_channels
            ),
            max_amplitude=MAX_STIM_AMPLITUDE_UA,
        )
        self.last_contact_encoding = None
        self.last_feedback_event = None
        self.last_outcome_feedback_event = None
        self.active_scenario = None
        self.scenario_schedule = ContactScenarioSchedule(
            load_contact_scenarios(default_contact_scenario_path())
        )
        self.contact_perturbation = RoboSuiteContactPerturbation(raw_env)
        self.episode_rewards = []
        self.best_reward = -np.inf
        self.last_control_report = None
        self.last_episode_control_summary = summarize_contact_reports([])
        self.last_episode_metrics = {}

    def _detect_spikes(self, trigger_stim_timestamps=()):
        if self.spike_reader is not None:
            window = self.spike_reader.read(
                trigger_stim_timestamps=trigger_stim_timestamps,
            )
            if (
                trigger_stim_timestamps
                and not is_cl_simulator()
                and window.timing_valid is not True
            ):
                raise RuntimeError(
                    "CL1 response window failed the stimulation-to-collection "
                    "timing gate"
                )
            self.last_spike_window = window.to_dict()
            real_counts = np.asarray(
                window.features.channel_counts,
                dtype=np.int64,
            )
            counts = ablate_channel_counts(
                real_counts,
                self.ablation_spike_mode,
                rng=self.rng,
                channel_subset=self.channel_map.motor_channels,
            )
            spikes = np.flatnonzero(counts).astype(int).tolist()
            firing_rates = (
                counts.astype(np.float64)
                / max(self.collect_window_ms / 1000.0, 1e-9)
            )
            return spikes, firing_rates, counts

        # Explicit compatibility mode only. This path cannot establish spike
        # timing and must not be used for CL1 learning claims.
        frames = self.neurons.read(250, None)
        abs_frames = np.abs(frames.astype(np.float32))
        # Enforce an absolute minimum threshold (e.g. 50uV) to prevent 
        # the percentile function from hallucinating spikes from Gaussian noise
        # when the culture is silent or dead.
        threshold = max(50.0, np.percentile(abs_frames, 99.5))
        real_spikes = list(set(np.where(abs_frames > threshold)[1]))
        firing_rates = np.mean(abs_frames, axis=0)
        
        if self.ablation_spike_mode == "zero":
            spikes = []
        elif self.ablation_spike_mode == "random":
            n = len(real_spikes)
            spikes = np.random.choice(64, size=n, replace=False).tolist() if n else []
        else:
            spikes = real_spikes
        counts = np.zeros(64, dtype=np.int64)
        counts[spikes] = 1
        self.last_spike_window = None
        return spikes, firing_rates, counts

    def _enter_phase_and_observe_rest(self) -> None:
        """Enforce each protocol phase's cumulative retention delay once."""

        phase = self.current_protocol_phase
        if self._entered_protocol_phase is phase.phase:
            return
        self._entered_protocol_phase = phase.phase
        rest_seconds = float(phase.rest_before_seconds)
        self.last_rest_observation = {
            "phase": phase.phase.value,
            "requested_seconds": rest_seconds,
            "observed_seconds": 0.0,
            "observed_ticks": 0,
            "simulator_skipped": False,
        }
        if rest_seconds <= 0:
            return
        if is_cl_simulator():
            # The official simulator contains no stimulation-responsive
            # plasticity, so a wall-clock retention delay has no experimental
            # meaning and would only make software validation take 45 minutes.
            self.last_rest_observation["simulator_skipped"] = True
            return

        observed_ticks = 0
        started_at = time.monotonic()
        for _ in self.neurons.loop(
            ticks_per_second=1,
            stop_after_seconds=rest_seconds,
            ignore_jitter=True,
        ):
            observed_ticks += 1
        self.last_rest_observation["observed_seconds"] = float(
            time.monotonic() - started_at
        )
        self.last_rest_observation["observed_ticks"] = observed_ticks

    def run_episode(self, max_steps=MAX_STEPS, record=False, ep_num=0):
        # Seed both numpy/random and Gym environment to guarantee exact paired layouts
        seed = 42 + ep_num
        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.current_protocol_phase = self.protocol.phase_for_episode(ep_num)
        self._enter_phase_and_observe_rest()
        obs, _ = self.env.reset(seed=seed)
        if GENERALIZATION_ENABLED:
            self.active_scenario = self.scenario_schedule.select(
                ep_num,
                frozen_evaluation=(
                    self.current_protocol_phase.heldout_evaluation
                    or self.current_protocol_phase.is_evaluation
                ),
            )
            self.contact_perturbation.apply(self.active_scenario)
        else:
            self.active_scenario = None
        obs_info = extract_obs(obs, raw_env=self.raw_env)
        episode_budget_start = self.neurons.budget_snapshot()
        self.contact_encoder.reset()
        self.contact_decoder.reset()
        self.nominal_controller.reset()
        self.contact_feedback.reset(float(obs_info["nut_peg_distance_m"]))
        self.last_control_report = None
        self.last_contact_encoding = None
        self.last_feedback_event = None
        self.last_outcome_feedback_event = None
        episode_delivered_feedback = []
        self.last_episode_control_summary = summarize_contact_reports([])
        total_reward = 0.0
        frames_list = []
        episode_success = False
        time_to_success_steps = None
        ep_force_safe = []
        force_history = []
        torque_history = []
        step_rewards = deque(maxlen=50)
        cur_fr = np.zeros(64)
        ep_firing_acc = []
        control_reports = []

        for step in range(max_steps):
            baseline = self.nominal_controller.propose(obs_info)
            sensory_sequence_start = self.neurons.stim_calls
            encoding = self.contact_encoder.encode(
                obs_info,
                self.nominal_controller.phase,
                stimulation_enabled=self.sensory_stimulation_enabled,
            )
            self.last_contact_encoding = encoding.to_dict()
            sensory_stim_events = self.neurons.stim_events_since(
                sensory_sequence_start
            )
            trigger_stim_timestamps = tuple(
                int(event["timestamp"])
                for event in sensory_stim_events
                if event["timestamp"] is not None
            )
            spikes, cur_fr, spike_counts = self._detect_spikes(
                trigger_stim_timestamps
            )
            motor_spikes = [
                channel
                for channel in spikes
                if channel in self.channel_map.motor_channels
            ]
            ep_firing_acc.append(cur_fr.copy())
            pdi_val = 0.0
            raw = self.contact_decoder.decode_counts(
                spike_counts,
                pdi_boost=0.0,
            )
            action, control_report = self.contact_controller.compose(
                baseline,
                raw,
                phase=self.nominal_controller.phase,
                residual_confidence=spike_confidence(motor_spikes),
                force_vector_n=obs_info["force"],
                residual_enabled=(
                    self.current_protocol_phase.phase
                    is not LearningPhase.CALIBRATION
                ),
            )
            self.last_control_report = control_report.to_dict()
            control_reports.append(dict(self.last_control_report))
            force_before = np.asarray(obs_info["force"], dtype=float).copy()
            torque_before = np.asarray(obs_info["torque"], dtype=float).copy()
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs_info = extract_obs(obs, raw_env=self.raw_env)
            total_reward += reward

            force_mag = np.linalg.norm(obs_info["force"])
            torque_mag = np.linalg.norm(obs_info["torque"])
            depth, cur_dist = compute_placement_metrics(obs_info)
            force_safe = force_mag < FORCE_SAFETY_THRESHOLD
            torque_safe = torque_mag < TORQUE_SAFETY_THRESHOLD
            official_success = bool(obs_info["placement_success"])
            task_success = bool(
                official_success
                and self.nominal_controller.phase is TaskPhase.COMPLETE
            )
            if task_success and not episode_success:
                episode_success = True
                time_to_success_steps = step + 1

            ep_force_safe.append(1 if force_safe else 0)
            force_history.append(float(force_mag))
            torque_history.append(float(torque_mag))
            step_rewards.append(reward)

            outcome_feedback_event = self.contact_feedback.evaluate(
                distance_m=cur_dist,
                force_n=float(force_mag),
                success=task_success,
                enabled=(
                    task_success
                    or self.nominal_controller.phase.value
                    in self.contact_controller.config.enabled_phases
                ),
            )
            delivered_feedback_event = (
                event_for_step(
                    self.yoked_feedback_schedule,
                    ep_num,
                    step,
                )
                if self.ablation_stim_mode == "yoked"
                else outcome_feedback_event
            )
            self.last_outcome_feedback_event = (
                outcome_feedback_event.to_dict()
            )
            self.last_feedback_event = delivered_feedback_event.to_dict()
            feedback_stimulated = False
            if (
                self.current_protocol_phase.biological_feedback_enabled
                and self.ablation_stim_mode != "none"
            ):
                feedback_stimulated = self.contact_feedback_stimulator.emit(
                    delivered_feedback_event
                )
            episode_delivered_feedback.append(
                delivered_feedback_event
                if feedback_stimulated
                else ContactFeedbackEvent(
                    kind="not_delivered",
                    valence=0,
                    strength=0.0,
                    distance_delta_m=(
                        delivered_feedback_event.distance_delta_m
                    ),
                    force_n=delivered_feedback_event.force_n,
                )
            )

            self.session_recorder.append({
                "episode": ep_num,
                "step": step,
                "protocol_phase": self.current_protocol_phase.phase.value,
                "encoder_frozen": self.current_protocol_phase.encoder_frozen,
                "decoder_frozen": self.current_protocol_phase.decoder_frozen,
                "feedback_enabled": (
                    self.current_protocol_phase.biological_feedback_enabled
                ),
                "retention_rest": self.last_rest_observation,
                "scenario": (
                    None
                    if self.active_scenario is None
                    else self.active_scenario.to_dict()
                ),
                "contact_encoding": self.last_contact_encoding,
                "sensory_stim_events": sensory_stim_events,
                "feedback_event": self.last_feedback_event,
                "outcome_feedback_event": self.last_outcome_feedback_event,
                "feedback_stimulated": feedback_stimulated,
                "feedback_delivery_mode": self.ablation_stim_mode,
                "sensory_stimulation_enabled": (
                    self.sensory_stimulation_enabled
                ),
                "stimulation_budget": self.neurons.budget_snapshot(),
                "spike_total": int(np.sum(spike_counts)),
                "active_channels": spikes,
                "motor_active_channels": motor_spikes,
                "spike_window": self.last_spike_window,
                "control": self.last_control_report,
                "force_before": force_before.tolist(),
                "torque_before": torque_before.tolist(),
                "force_after": np.asarray(
                    obs_info["force"],
                    dtype=float,
                ).tolist(),
                "torque_after": np.asarray(
                    obs_info["torque"],
                    dtype=float,
                ).tolist(),
                "task_metrics": {
                    "episode_success": episode_success,
                    "step_success": task_success,
                    "official_placement_success": official_success,
                    "time_to_success_steps": time_to_success_steps,
                    "force_safe": force_safe,
                    "torque_safe": torque_safe,
                    "nut_peg_xy_error_m": obs_info["nut_peg_xy_error_m"],
                    "nut_peg_distance_m": obs_info["nut_peg_distance_m"],
                    "nut_height_above_table_m": (
                        obs_info["nut_height_above_table_m"]
                    ),
                    "placement_depth_margin_m": (
                        obs_info["placement_depth_margin_m"]
                    ),
                    "nut_peg_yaw_error_deg": (
                        obs_info["nut_peg_yaw_error_deg"]
                    ),
                    "grasp_confirmed": obs_info["grasp_confirmed"],
                },
            })

            if record:
                try:
                    frame = self.raw_env.sim.render(width=RENDER_W, height=RENDER_H, camera_name="frontview")
                    if frame is not None:
                        frame = frame[::-1]
                    else:
                        frame = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
                except Exception:
                    frame = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
                if frame is not None and frame.size > 0:
                    health_full = self.neurons.get_health() if hasattr(self.neurons, 'get_health') else None
                    min_h = float(health_full.min()) if health_full is not None else 1.0
                    sr = 100.0 if episode_success else 0.0
                    fsr = np.mean(ep_force_safe) * 100
                    frame = draw_overlay(frame, ep_num, total_reward, pdi_val, min_h, cur_fr,
                                         step_rewards, distance=cur_dist, force_mag=force_mag,
                                         torque_mag=torque_mag, depth=depth,
                                         success_rate=sr, force_safe_rate=fsr,
                                         force_vec=obs_info["force"], health_arr=health_full,
                                         force_threshold=FORCE_SAFETY_THRESHOLD, is_sim=is_cl_simulator())
                    frames_list.append(frame)

            # Nominal modes finish release and retreat before accepting the
            # official RoboSuite placement signal as a binary endpoint.
            if episode_success or terminated or truncated:
                break

        if total_reward > self.best_reward:
            self.best_reward = total_reward

        if ep_firing_acc:
            ep_avg_fr = np.mean(ep_firing_acc, axis=0)
            hud.episode_firing_history.append(ep_avg_fr)

        self.last_episode_control_summary = summarize_contact_reports(
            control_reports
        )
        force_array = np.asarray(force_history, dtype=float)
        torque_array = np.asarray(torque_history, dtype=float)
        success_rate = 100.0 if episode_success else 0.0
        force_safe_rate = np.mean(ep_force_safe) * 100 if ep_force_safe else 100.0
        health = culture_health_metrics(self.neurons)
        budget_end = self.neurons.budget_snapshot()
        episode_dose = stimulation_budget_delta(
            episode_budget_start,
            budget_end,
        )
        self.last_episode_metrics = {
            "episode_success": episode_success,
            "time_to_success_steps": time_to_success_steps,
            "step_count": len(force_history),
            "force_safe_rate": float(force_safe_rate),
            "episode_force_safe": bool(all(ep_force_safe)),
            "peak_force_n": (
                float(np.max(force_array)) if force_array.size else 0.0
            ),
            "p95_force_n": (
                float(np.percentile(force_array, 95)) if force_array.size else 0.0
            ),
            "peak_torque_nm": (
                float(np.max(torque_array)) if torque_array.size else 0.0
            ),
            "p95_torque_nm": (
                float(np.percentile(torque_array, 95))
                if torque_array.size
                else 0.0
            ),
            "final_nut_peg_xy_error_m": float(
                obs_info["nut_peg_xy_error_m"]
            ),
            "final_nut_peg_distance_m": float(
                obs_info["nut_peg_distance_m"]
            ),
            "final_placement_depth_margin_m": float(
                obs_info["placement_depth_margin_m"]
            ),
            "final_nut_peg_yaw_error_deg": float(
                obs_info["nut_peg_yaw_error_deg"]
            ),
            "final_grasp_confirmed": bool(obs_info["grasp_confirmed"]),
            **budget_end,
            "episode_stim_calls": episode_dose["stim_calls"],
            "episode_channel_pulses": episode_dose["channel_pulses"],
            "episode_abs_charge_nc": episode_dose["abs_charge_nc"],
            **health,
        }
        self.recorded_feedback_schedule[int(ep_num)] = tuple(
            episode_delivered_feedback
        )
        return total_reward, 0.0, frames_list, success_rate, force_safe_rate

    def train(self, num_episodes=EPISODES, record_last_n=RECORD_LAST_N):
        print("\n" + "=" * 60)
        mode_str = "Simulator Mode" if is_cl_simulator() else "Pure Wetware Mode"
        print("  CL1 Bio-Computer Training (" + mode_str + ")")
        print("=" * 60)
        print(f"  Episodes: {num_episodes} | Env: {ENV_NAME} ({ROBOT})")
        print(f"  Control: {self.control_mode}")
        print(f"  Protocol: {self.protocol.protocol_id}")
        print(f"  Spikes: {self.spike_pipeline}")
        
        all_frames = []
        all_sr = []
        all_fsr = []
        record_start = max(0, num_episodes - record_last_n)

        self.session_recorder.start()
        try:
            pbar = tqdm(range(num_episodes), desc="CL1", ncols=90)
            for ep in pbar:
                rec = (ep >= record_start)
                reward, pdi_val, frames, sr, fsr = self.run_episode(record=rec, ep_num=ep)
                self.episode_rewards.append(reward)
                all_sr.append(sr)
                all_fsr.append(fsr)
                if rec:
                    all_frames.extend(frames)

                avg = np.mean(self.episode_rewards[-20:])
                phase = self.current_protocol_phase.phase.value
                pbar.set_postfix(R=f"{reward:.1f}", avg20=f"{avg:.1f}",
                                 phase=phase, PDI=f"{pdi_val:.2f}",
                                 SR=f"{sr:.0f}%", FSR=f"{fsr:.0f}%")
        finally:
            self.session_recorder.stop()

        final = np.mean(self.episode_rewards[-20:])
        final_sr = np.mean(all_sr[-20:])
        final_fsr = np.mean(all_fsr[-20:])
        print(f"\n  CL1 Done | avg20={final:.2f} SR={final_sr:.1f}% FSR={final_fsr:.1f}%")
        return all_frames

# ═══ Main Entry Point ═══
def main():
    np.random.seed(SEED)
    base_provenance = project_provenance(protocol_id=PROTOCOL_ID)
    hud.reset()
    print("+" + "=" * 58 + "+")
    if is_cl_simulator():
        print("|  Senxe Cerebellum v5.0 — cl-sdk Simulator Mode            |")
        print("|  [!] WARNING: Real CL1 hardware not detected.             |")
        print("|      Falling back to official cl-sdk Poisson simulation.  |")
    else:
        print("|  Senxe Cerebellum v5.0 — Bounded Contact Skill            |")
    print("+" + "=" * 58 + "+\n")

    print("-" * 60)
    print("  Phase 0: Channel Warm-up Calibration")
    print("-" * 60)
    with cl_open() as neurons:
        require_stimulation_approval()
        require_hardware_evidence_recording()
        experiment_neurons = BudgetedNeurons(
            neurons,
            safety_limits=STIM_SAFETY_LIMITS,
        )
        calibration_recorder = CLSessionRecorder(
            experiment_neurons,
            SessionRecordingConfig(
                enabled=RECORD_CL_SESSION,
                file_location=RECORDING_LOCATION,
                file_suffix="senxe_calibration_v2",
            ),
            attributes={
                "protocol_id": PROTOCOL_ID,
                "stage": "timestamped_calibration",
                **base_provenance,
            },
        )
        calibration_recorder.start()
        try:
            calibration = timestamped_contact_calibration(
                experiment_neurons,
                WARMUP_SECONDS,
                artifact_wait_ms=float(
                    ARTIFACT_WAIT_OVERRIDE
                    if ARTIFACT_WAIT_OVERRIDE is not None
                    else 50.0
                ),
                collect_window_ms=float(
                    COLLECT_WINDOW_OVERRIDE
                    if COLLECT_WINDOW_OVERRIDE is not None
                    else 50.0
                ),
                max_stim_amplitude=MAX_STIM_AMPLITUDE_UA,
            )
            resolved_map, health_report = resolve_contact_channel_map(
                calibration.input_responsiveness,
                calibration.output_responsiveness,
                input_threshold=MIN_INPUT_RESPONSE_DELTA,
                output_threshold=MIN_OUTPUT_RESPONSE_DELTA,
                allow_reserve_remap=ALLOW_CALIBRATION_REMAP,
            )
            if not is_cl_simulator():
                require_channel_health(health_report)
            provenance = project_provenance(
                resolved_map,
                protocol_id=PROTOCOL_ID,
            )
            ranking = calibration.channel_ranking
            resp = calibration.output_responsiveness
            calibration_recorder.append({
                "event": "calibration_complete",
                "calibration": calibration.to_dict(),
                "channel_health_report": health_report.to_dict(),
                "resolved_channel_map": channel_map_metadata(resolved_map),
                "resolved_provenance": provenance,
                "stimulation_budget": (
                    experiment_neurons.budget_snapshot()
                ),
            })
        finally:
            calibration_recorder.stop()

        print("-" * 60)
        print("  Phase 1: Bio-Agent Control Loop")
        print("-" * 60)
        env, raw_env = make_robosuite_env(render=True)
        agent = CL1Agent(
            env,
            raw_env,
            experiment_neurons,
            channel_ranking=ranking,
            responsiveness=resp,
            experiment_metadata=provenance,
            channel_map=resolved_map,
            channel_health_report=health_report,
        )
        cl1_frames = agent.train(num_episodes=EPISODES, record_last_n=RECORD_LAST_N)
        env.close()

    print("\n" + "-" * 60)
    print("  Phase 2: Generating Video")
    print("-" * 60)
    save_video(cl1_frames, VIDEO_CL1, fps=VIDEO_FPS, target_seconds=20)

    print("\n" + "=" * 60)
    print("  Execution Complete.")
    print(f"  Video saved: {VIDEO_CL1}")
    print("=" * 60)

if __name__ == "__main__":
    main()
