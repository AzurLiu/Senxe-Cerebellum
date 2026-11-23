#!/usr/bin/env python3
"""Run culture-aware, counterbalanced Senxe contact-skill ablations.

Real CL1 hardware is restricted to one condition per invocation by default.
This prevents a fresh Python agent from being mistaken for a fresh biological
replicate. Multi-condition execution is available for the non-learning SDK
simulator, where it is useful only for software and paired-seed validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from tqdm import tqdm

from core.channel_health import (
    ChannelHealthReport,
    health_thresholds_from_env,
    require_channel_health,
    resolve_contact_channel_map,
)
from core.channel_map import ContactChannelMap, channel_map_metadata
from core.learning_protocol import default_protocol_path, load_protocols
from core.provenance import project_provenance
from core.neurons import (
    cl_open,
    BACKEND_SIMULATOR,
    evidence_backend,
    simulator_mode_hint,
    timestamped_contact_calibration,
)
from core.yoked_feedback import (
    FeedbackSchedule,
    load_feedback_schedule,
    load_feedback_schedule_metadata,
    save_feedback_schedule,
)
from core.session_recording import CLSessionRecorder, SessionRecordingConfig
from senxe_demo_robosuite import (
    MAX_STEPS,
    MAX_STIM_AMPLITUDE_UA,
    MAX_STIM_ABS_CHARGE_NC,
    MAX_STIM_CALLS,
    MAX_STIM_CHANNEL_PULSES,
    ALLOW_CALIBRATION_REMAP,
    RECORD_CL_SESSION,
    RECORDING_LOCATION,
    STIM_SAFETY_LIMITS,
    CL1Agent,
    make_robosuite_env,
)


SEED = 42
BENCHMARK_PROTOCOL_ID = os.getenv(
    "SENXE_BENCHMARK_PROTOCOL_ID",
    os.getenv("SENXE_PROTOCOL_ID", "senxe_contact_skill_v2"),
).strip()
CONDITIONS = (
    {
        "name": "contact_skill",
        "spike_mode": "none",
        "stim_mode": "full",
        "sensory_stimulation": True,
    },
    {
        "name": "baseline_only",
        "spike_mode": "zero",
        "stim_mode": "none",
        "sensory_stimulation": False,
    },
    {
        "name": "zero_spikes",
        "spike_mode": "zero",
        "stim_mode": "full",
        "sensory_stimulation": True,
    },
    {
        "name": "shuffled_spikes",
        "spike_mode": "shuffled",
        "stim_mode": "full",
        "sensory_stimulation": True,
    },
    {
        "name": "no_feedback",
        "spike_mode": "none",
        "stim_mode": "none",
        "sensory_stimulation": True,
    },
    {
        "name": "yoked_feedback",
        "spike_mode": "none",
        "stim_mode": "yoked",
        "sensory_stimulation": True,
    },
)
CONDITION_BY_NAME = {
    condition["name"]: condition for condition in CONDITIONS
}
CSV_FIELDS = (
    "RunID",
    "Backend",
    "CultureID",
    "SequenceIndex",
    "ConditionOrder",
    "Condition",
    "CodeHash",
    "ProtocolHash",
    "ChannelMapVersion",
    "ChannelMapHash",
    "ChannelHealthReport",
    "GitCommit",
    "RuntimeVersions",
    "Episode",
    "ProtocolPhase",
    "RetentionDelaySeconds",
    "Scenario",
    "ScenarioSplit",
    "Reward",
    "EpisodeSuccess",
    "TimeToSuccessSteps",
    "ForceSafeRate",
    "EpisodeForceSafe",
    "PeakForceN",
    "P95ForceN",
    "PeakTorqueNm",
    "P95TorqueNm",
    "ResetSeed",
    "InitialNutXM",
    "InitialNutYM",
    "InitialNutYawDeg",
    "FinalNutPegXYErrorM",
    "FinalNutPegDistanceM",
    "FinalPlacementDepthMarginM",
    "FinalNutPegYawErrorDeg",
    "FinalGraspConfirmed",
    "CultureHealthMin",
    "CultureHealthMean",
    "CultureHealthActiveChannels",
    "ResidualAppliedRate",
    "MeanResidualNorm",
    "MeanComplianceScale",
    "RetractCount",
    "HardStopCount",
    "StimSafetyLimits",
    "MaxStimAmplitudeUa",
    "MaxStimCalls",
    "MaxStimChannelPulses",
    "MaxStimAbsChargeNc",
    "StimCalls",
    "StimChannelPulses",
    "StimAbsChargeNc",
    "EpisodeStimCalls",
    "EpisodeStimChannelPulses",
    "EpisodeStimAbsChargeNc",
    "SensoryStimulationEnabled",
    "FeedbackDeliveryMode",
    "YokedFeedbackSource",
)


def counterbalanced_condition_order(
    sequence_index: int,
    *,
    seed: int = SEED,
) -> list[str]:
    """Return a seeded rotation / reversal counterbalancing sequence."""

    names = [condition["name"] for condition in CONDITIONS]
    rng = np.random.default_rng(seed)
    base = list(rng.permutation(names))
    block, offset = divmod(int(sequence_index), len(base))
    if block % 2:
        base.reverse()
    return base[offset:] + base[:offset]


def parse_args() -> argparse.Namespace:
    protocol = load_protocols(default_protocol_path())[BENCHMARK_PROTOCOL_ID]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--condition",
        choices=("all", *CONDITION_BY_NAME),
        default=os.getenv("SENXE_ABLATION_CONDITION", "all"),
    )
    parser.add_argument(
        "--culture-id",
        default=os.getenv("SENXE_CULTURE_ID", "simulator"),
    )
    parser.add_argument(
        "--sequence-index",
        type=int,
        default=int(os.getenv("SENXE_SEQUENCE_INDEX", "0")),
    )
    parser.add_argument(
        "--order-seed",
        type=int,
        default=int(os.getenv("SENXE_ABLATION_ORDER_SEED", str(SEED))),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=int(os.getenv(
            "SENXE_ABLATION_EPISODES",
            str(protocol.total_episodes),
        )),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.getenv(
            "SENXE_ABLATION_OUTPUT",
            "ablation_results.csv",
        )),
    )
    parser.add_argument(
        "--yoked-feedback",
        type=Path,
        default=(
            Path(os.environ["SENXE_YOKED_FEEDBACK_PATH"])
            if os.getenv("SENXE_YOKED_FEEDBACK_PATH")
            else None
        ),
    )
    parser.add_argument(
        "--feedback-output-dir",
        type=Path,
        default=Path("feedback_schedules"),
    )
    parser.add_argument(
        "--restart-interrupted-run",
        action="store_true",
        help=(
            "Explicitly restart a run that the ledger shows was started "
            "before for this culture, condition and sequence index. The "
            "culture is calibrated (stimulated) again, and the restart is "
            "recorded in the ledger."
        ),
    )
    parser.add_argument(
        "--allow-hardware-crossover",
        action="store_true",
        help=(
            "Explicitly allow multiple conditions on one culture. Intended "
            "only for a separately approved crossover protocol."
        ),
    )
    return parser.parse_args()


def _validate_hardware_scope(args: argparse.Namespace, simulator: bool) -> None:
    if simulator:
        return
    if not args.culture_id or args.culture_id == "simulator":
        raise RuntimeError(
            "Real CL1 runs require a non-placeholder --culture-id."
            + simulator_mode_hint()
        )
    if not RECORD_CL_SESSION:
        raise RuntimeError(
            "Real CL1 ablations require SENXE_RECORD_SESSION=1."
            + simulator_mode_hint()
        )
    prior_conditions: set[str] = set()
    if args.output.exists() and args.output.stat().st_size > 0:
        with args.output.open(newline="", encoding="utf-8") as handle:
            prior_rows = list(csv.DictReader(handle))
        prior_conditions |= {
            str(row.get("Condition", ""))
            for row in prior_rows
            if str(row.get("CultureID", "")) == args.culture_id
        }
    # Runs that started but crashed before writing any episode row still
    # stimulated the culture; the ledger keeps them visible here.
    for entry in _read_run_ledger(args.output):
        if str(entry.get("culture_id", "")) == args.culture_id:
            prior_conditions |= {
                str(name) for name in entry.get("conditions", [])
            }
    requested_conditions = (
        set(CONDITION_BY_NAME)
        if args.condition == "all"
        else {args.condition}
    )
    if (
        prior_conditions
        and len(prior_conditions | requested_conditions) > 1
        and not args.allow_hardware_crossover
    ):
        raise RuntimeError(
            "This culture ID already appears under another condition; "
            "allocate an independent culture or use an explicitly "
            "approved crossover design"
        )
    if args.condition == "all" and not args.allow_hardware_crossover:
        raise RuntimeError(
            "Real CL1 runs default to one condition per culture invocation. "
            "Select --condition and allocate independent cultures, or use "
            "--allow-hardware-crossover only with an approved carryover design."
        )
    includes_yoked = args.condition in {"all", "yoked_feedback"}
    if includes_yoked and args.yoked_feedback is None:
        raise RuntimeError(
            "Any real-hardware yoked_feedback condition requires "
            "--yoked-feedback from an independent donor run"
        )
    if args.yoked_feedback is not None:
        donor = load_feedback_schedule_metadata(args.yoked_feedback)
        donor_culture = str(donor.get("culture_id", ""))
        if not donor_culture:
            raise RuntimeError(
                "The yoked donor schedule must identify its culture_id"
            )
        if donor_culture == args.culture_id:
            raise RuntimeError(
                "The yoked recipient culture must differ from the donor "
                "culture recorded in the schedule"
            )
        if donor.get("condition") != "contact_skill":
            raise RuntimeError(
                "The yoked donor schedule must come from contact_skill"
            )
        if donor.get("protocol_id") != BENCHMARK_PROTOCOL_ID:
            raise RuntimeError(
                "The yoked donor protocol must match the recipient protocol"
            )
        current_provenance = project_provenance(
            protocol_id=BENCHMARK_PROTOCOL_ID,
        )
        if donor.get("protocol_hash") != current_provenance["protocol_hash"]:
            raise RuntimeError(
                "The yoked donor protocol hash must match the recipient"
            )
        if donor.get("code_hash") != current_provenance["code_hash"]:
            raise RuntimeError(
                "The yoked donor runtime-source hash must match the recipient"
            )
        if (
            donor.get("runtime_versions")
            != current_provenance["runtime_versions"]
        ):
            raise RuntimeError(
                "The yoked donor runtime versions must match the recipient"
            )


def _selected_conditions(args: argparse.Namespace) -> list[str]:
    order = counterbalanced_condition_order(
        args.sequence_index,
        seed=args.order_seed,
    )
    if args.condition == "all":
        return order
    return [args.condition]


def _validate_append_rows(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                raise RuntimeError(
                    "Existing ablation CSV schema does not match V2; choose "
                    "a new --output path instead of mixing schemas"
                )
            existing_rows = list(reader)
            existing_keys = {
                (
                    row["RunID"],
                    row["Condition"],
                    row["Episode"],
                )
                for row in existing_rows
            }
        incoming_keys = {
            (
                str(row["RunID"]),
                str(row["Condition"]),
                str(row["Episode"]),
            )
            for row in rows
        }
        duplicates = existing_keys & incoming_keys
        if duplicates:
            raise RuntimeError(
                "Refusing to append duplicate run / condition / episode rows"
            )
        for field in (
            "Backend",
            "CodeHash",
            "ProtocolHash",
            "ChannelMapHash",
            "RuntimeVersions",
        ):
            incoming_values = {
                str(row.get(field, "")).strip()
                for row in rows
                if str(row.get(field, "")).strip()
            }
            if not incoming_values:
                continue
            existing_values = {
                str(row[field]).strip()
                for row in existing_rows
            }
            if existing_values != incoming_values:
                raise RuntimeError(
                    f"Refusing to mix different {field} values in one "
                    "evidence table"
                )


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _validate_append_rows(path, rows)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def run_ledger_path(output: Path) -> Path:
    """Sidecar file listing every run that was started against a culture."""

    return output.with_name(output.name + ".runs.jsonl")


def _read_run_ledger(output: Path) -> list[dict[str, Any]]:
    path = run_ledger_path(output)
    if not path.exists():
        return []
    entries = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def _record_run_started(
    output: Path,
    *,
    culture_id: str,
    sequence_index: int,
    conditions: list[str],
    backend: str,
    restart_of: int = 0,
) -> None:
    """Durably note that a culture is about to be stimulated.

    Written before calibration delivers the first pulse, so a run that
    crashes before its first episode row still blocks reuse of the culture
    under another condition.
    """

    path = run_ledger_path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "event": "run_restarted" if restart_of else "run_started",
        "prior_starts": int(restart_of),
        "culture_id": culture_id,
        "sequence_index": int(sequence_index),
        "conditions": list(conditions),
        "backend": backend,
        "started_at_unix": time.time(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _donor_schedule_path(args: argparse.Namespace) -> Path:
    return args.feedback_output_dir / (
        f"{args.culture_id}_seq{args.sequence_index}_contact_skill.json"
    )


def _prior_starts(
    args: argparse.Namespace,
    selected: list[str],
) -> list[dict[str, Any]]:
    """Ledger entries for this culture, sequence index and any selected condition."""

    return [
        entry
        for entry in _read_run_ledger(args.output)
        if str(entry.get("culture_id", "")) == str(args.culture_id)
        and int(entry.get("sequence_index", -1)) == int(args.sequence_index)
        and set(map(str, entry.get("conditions", []))) & set(selected)
    ]


def _preflight_outputs(
    args: argparse.Namespace,
    *,
    selected: list[str],
    backend: str,
    provenance: Mapping[str, str],
) -> None:
    """Reject output conflicts that are knowable before any stimulation.

    Covers duplicate run / condition / episode keys, mixing backends, code,
    protocol or runtime versions in one table, and an existing donor
    feedback schedule. The channel-map hash is only known after calibration
    and is checked again there.
    """

    for condition_name in selected:
        run_id = f"{args.culture_id}:seq{args.sequence_index}:{condition_name}"
        _validate_append_rows(
            args.output,
            [
                {
                    "RunID": run_id,
                    "Condition": condition_name,
                    "Episode": episode,
                    "Backend": backend,
                    "CodeHash": provenance["code_hash"],
                    "ProtocolHash": provenance["protocol_hash"],
                    "RuntimeVersions": provenance["runtime_versions"],
                }
                for episode in range(args.episodes)
            ],
        )
    # The CSV only knows about runs that finished at least one episode. A
    # run that crashed after calibration but before its first row is only
    # visible in the ledger, and silently starting it again would calibrate
    # (stimulate) the same culture a second time.
    earlier = _prior_starts(args, selected)
    if earlier and not getattr(args, "restart_interrupted_run", False):
        raise RuntimeError(
            "The run ledger shows this culture, sequence index and "
            f"condition were already started ({len(earlier)} time(s)). "
            "A started run without complete results is an interrupted run: "
            "it is not restarted automatically. Use a new culture or "
            "sequence index, or pass --restart-interrupted-run to restart "
            "it deliberately."
        )
    if "contact_skill" in selected and _donor_schedule_path(args).exists():
        raise RuntimeError(
            "Refusing to overwrite an existing donor feedback schedule: "
            f"{_donor_schedule_path(args)}"
        )


def _scenario_value(agent: CL1Agent, name: str, fallback: str = "none") -> str:
    if agent.active_scenario is None:
        return fallback
    return str(getattr(agent.active_scenario, name))


def _run_condition(
    *,
    args: argparse.Namespace,
    neurons: Any,
    ranking: np.ndarray,
    responsiveness: np.ndarray,
    condition_name: str,
    condition_order: list[str],
    yoked_schedule: FeedbackSchedule,
    channel_map: ContactChannelMap,
    channel_health_report: ChannelHealthReport,
    provenance: Mapping[str, str] | None = None,
    backend: str,
    on_feedback_schedule: Any = None,
) -> tuple[list[dict[str, Any]], FeedbackSchedule]:
    condition = CONDITION_BY_NAME[condition_name]
    provenance = dict(provenance or project_provenance(
        channel_map,
        protocol_id=BENCHMARK_PROTOCOL_ID,
    ))
    if condition["stim_mode"] == "yoked" and not yoked_schedule:
        raise RuntimeError(
            "yoked_feedback cannot run before a donor schedule is available"
        )

    env, raw_env = make_robosuite_env(render=False)
    agent: CL1Agent | None = None
    completed = False
    try:
        run_id = (
            f"{args.culture_id}:seq{args.sequence_index}:{condition_name}"
        )
        agent = CL1Agent(
            env,
            raw_env,
            neurons,
            channel_ranking=ranking,
            responsiveness=responsiveness,
            ablation_spike_mode=condition["spike_mode"],
            ablation_stim_mode=condition["stim_mode"],
            yoked_feedback_schedule=yoked_schedule,
            sensory_stimulation_enabled=condition[
                "sensory_stimulation"
            ],
            channel_map=channel_map,
            channel_health_report=channel_health_report,
            protocol_id=BENCHMARK_PROTOCOL_ID,
            experiment_metadata={
                "run_id": run_id,
                "culture_id": args.culture_id,
                "sequence_index": args.sequence_index,
                "condition": condition_name,
                "condition_order": condition_order,
                "backend": backend,
                **provenance,
                "file_suffix": (
                    f"{args.culture_id}_seq{args.sequence_index}"
                    f"_{condition_name}_v2"
                ),
            },
        )
        agent.session_recorder.start()
        rows: list[dict[str, Any]] = []
        progress = tqdm(
            range(args.episodes),
            desc=condition_name,
            ncols=96,
        )
        for episode in progress:
            reward, _, _, success_pct, force_safe_rate = agent.run_episode(
                max_steps=MAX_STEPS,
                record=False,
                ep_num=episode,
            )
            metrics = agent.last_episode_metrics
            control = agent.last_episode_control_summary
            phase = agent.current_protocol_phase
            row = {
                "RunID": run_id,
                "Backend": backend,
                "CultureID": args.culture_id,
                "SequenceIndex": args.sequence_index,
                "ConditionOrder": "|".join(condition_order),
                "Condition": condition_name,
                "CodeHash": provenance["code_hash"],
                "ProtocolHash": provenance["protocol_hash"],
                "ChannelMapVersion": provenance["channel_map_version"],
                "ChannelMapHash": provenance["channel_map_hash"],
                "ChannelHealthReport": json.dumps(
                    channel_health_report.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "GitCommit": provenance["git_commit"],
                "RuntimeVersions": provenance["runtime_versions"],
                "Episode": episode,
                "ProtocolPhase": phase.phase.value,
                "RetentionDelaySeconds": phase.rest_before_seconds,
                "Scenario": _scenario_value(agent, "scenario_id"),
                "ScenarioSplit": _scenario_value(agent, "split"),
                "Reward": reward,
                "EpisodeSuccess": int(bool(metrics["episode_success"])),
                "TimeToSuccessSteps": (
                    ""
                    if metrics["time_to_success_steps"] is None
                    else metrics["time_to_success_steps"]
                ),
                "ForceSafeRate": force_safe_rate,
                "EpisodeForceSafe": int(metrics["episode_force_safe"]),
                "PeakForceN": metrics["peak_force_n"],
                "P95ForceN": metrics["p95_force_n"],
                "PeakTorqueNm": metrics["peak_torque_nm"],
                "P95TorqueNm": metrics["p95_torque_nm"],
                "ResetSeed": metrics["reset_seed"],
                "InitialNutXM": metrics["initial_nut_x_m"],
                "InitialNutYM": metrics["initial_nut_y_m"],
                "InitialNutYawDeg": metrics["initial_nut_yaw_deg"],
                "FinalNutPegXYErrorM": (
                    metrics["final_nut_peg_xy_error_m"]
                ),
                "FinalNutPegDistanceM": (
                    metrics["final_nut_peg_distance_m"]
                ),
                "FinalPlacementDepthMarginM": (
                    metrics["final_placement_depth_margin_m"]
                ),
                "FinalNutPegYawErrorDeg": (
                    metrics["final_nut_peg_yaw_error_deg"]
                ),
                "FinalGraspConfirmed": int(
                    metrics["final_grasp_confirmed"]
                ),
                "CultureHealthMin": (
                    ""
                    if metrics["health_min"] is None
                    else metrics["health_min"]
                ),
                "CultureHealthMean": (
                    ""
                    if metrics["health_mean"] is None
                    else metrics["health_mean"]
                ),
                "CultureHealthActiveChannels": (
                    ""
                    if metrics["health_active_channels"] is None
                    else metrics["health_active_channels"]
                ),
                "ResidualAppliedRate": control["residual_applied_rate"],
                "MeanResidualNorm": control.get(
                    "mean_residual_norm",
                    control["mean_abs_applied_residual"],
                ),
                "MeanComplianceScale": control.get(
                    "mean_compliance_scale",
                    1.0,
                ),
                "RetractCount": control.get("retract_count", 0),
                "HardStopCount": control["hard_stop_count"],
                "StimSafetyLimits": json.dumps(
                    STIM_SAFETY_LIMITS.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "MaxStimAmplitudeUa": MAX_STIM_AMPLITUDE_UA,
                "MaxStimCalls": metrics["max_stim_calls"],
                "MaxStimChannelPulses": metrics[
                    "max_channel_pulses"
                ],
                "MaxStimAbsChargeNc": MAX_STIM_ABS_CHARGE_NC,
                "StimCalls": metrics["stim_calls"],
                "StimChannelPulses": metrics["channel_pulses"],
                "StimAbsChargeNc": metrics["abs_charge_nc"],
                "EpisodeStimCalls": metrics["episode_stim_calls"],
                "EpisodeStimChannelPulses": metrics[
                    "episode_channel_pulses"
                ],
                "EpisodeStimAbsChargeNc": metrics[
                    "episode_abs_charge_nc"
                ],
                "SensoryStimulationEnabled": int(
                    condition["sensory_stimulation"]
                ),
                "FeedbackDeliveryMode": condition["stim_mode"],
                "YokedFeedbackSource": (
                    str(args.yoked_feedback or "in_run_donor")
                    if condition["stim_mode"] == "yoked"
                    else ""
                ),
            }
            # Every episode is durable before the next one stimulates the
            # culture: a later crash must not erase delivered-dose evidence.
            _append_rows(args.output, [row])
            rows.append(row)
            progress.set_postfix(
                R=f"{reward:.2f}",
                success=int(success_pct > 0),
                FSR=f"{force_safe_rate:.0f}%",
                phase=phase.phase.value,
            )
        completed = True
        return rows, dict(agent.recorded_feedback_schedule)
    finally:
        if agent is not None:
            if not completed and on_feedback_schedule is not None:
                on_feedback_schedule(dict(agent.recorded_feedback_schedule))
            agent.session_recorder.stop()
        env.close()


def main() -> None:
    args = parse_args()
    provenance = project_provenance(
        protocol_id=BENCHMARK_PROTOCOL_ID,
    )
    condition_order = counterbalanced_condition_order(
        args.sequence_index,
        seed=args.order_seed,
    )
    selected = _selected_conditions(args)
    yoked_schedule = (
        load_feedback_schedule(args.yoked_feedback)
        if args.yoked_feedback is not None
        else {}
    )

    # Everything that can be decided without the device is decided before
    # it is opened. A mistaken re-run of a finished command must fail here,
    # not after calibration has already stimulated the culture again.
    backend = evidence_backend()
    simulator = backend == BACKEND_SIMULATOR
    _validate_hardware_scope(args, simulator)
    input_z_threshold, output_z_threshold = health_thresholds_from_env(
        simulator=simulator,
    )
    _preflight_outputs(
        args,
        selected=selected,
        backend=backend,
        provenance=provenance,
    )

    with cl_open(STIM_SAFETY_LIMITS) as experiment_neurons:
        _record_run_started(
            args.output,
            culture_id=args.culture_id,
            sequence_index=args.sequence_index,
            conditions=selected,
            backend=backend,
            restart_of=len(_prior_starts(args, selected)),
        )
        calibration_recorder = CLSessionRecorder(
            experiment_neurons,
            SessionRecordingConfig(
                enabled=RECORD_CL_SESSION,
                file_location=RECORDING_LOCATION,
                file_suffix=(
                    f"{args.culture_id}_seq{args.sequence_index}"
                    "_calibration_v2"
                ),
            ),
            attributes={
                "protocol_id": BENCHMARK_PROTOCOL_ID,
                "stage": "timestamped_calibration",
                "backend": backend,
                "culture_id": args.culture_id,
                "sequence_index": args.sequence_index,
                "max_stim_amplitude_ua": MAX_STIM_AMPLITUDE_UA,
                "max_stim_calls": MAX_STIM_CALLS,
                "max_stim_channel_pulses": (
                    MAX_STIM_CHANNEL_PULSES
                ),
                "stimulation_safety_limits": (
                    STIM_SAFETY_LIMITS.to_dict()
                ),
                **provenance,
            },
        )
        calibration_recorder.start()
        try:
            calibration = timestamped_contact_calibration(
                experiment_neurons,
                duration_sec=5.0,
                max_stim_amplitude=MAX_STIM_AMPLITUDE_UA,
            )
            resolved_map, health_report = resolve_contact_channel_map(
                calibration.input_response_z,
                calibration.output_response_z,
                input_threshold=input_z_threshold,
                output_threshold=output_z_threshold,
                allow_reserve_remap=ALLOW_CALIBRATION_REMAP,
            )
            if not simulator:
                require_channel_health(health_report)
            provenance = project_provenance(
                resolved_map,
                protocol_id=BENCHMARK_PROTOCOL_ID,
            )
            ranking = calibration.channel_ranking
            responsiveness = calibration.output_responsiveness
            if args.yoked_feedback is not None:
                donor = load_feedback_schedule_metadata(
                    args.yoked_feedback
                )
                if (
                    donor.get("channel_map_hash")
                    != provenance["channel_map_hash"]
                ):
                    raise RuntimeError(
                        "The yoked donor channel map must match the "
                        "calibrated recipient map"
                    )
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

        pending = list(selected)
        # Simulator-only convenience: acquire the donor schedule before a
        # yoked condition if no external donor file was supplied. Real hardware
        # never reaches this path because independent donor data is mandatory.
        if (
            simulator
            and "yoked_feedback" in pending
            and not yoked_schedule
            and "contact_skill" in pending
        ):
            pending.remove("contact_skill")
            pending.insert(0, "contact_skill")

        for condition_name in pending:
            recorded_order = (
                pending if args.condition == "all" else condition_order
            )
            planned_run_id = (
                f"{args.culture_id}:seq{args.sequence_index}:"
                f"{condition_name}"
            )
            _validate_append_rows(
                args.output,
                [
                    {
                        "RunID": planned_run_id,
                        "Condition": condition_name,
                        "Episode": episode,
                        "Backend": backend,
                        "CodeHash": provenance["code_hash"],
                        "ProtocolHash": provenance["protocol_hash"],
                        "ChannelMapHash": provenance["channel_map_hash"],
                        "RuntimeVersions": provenance["runtime_versions"],
                    }
                    for episode in range(args.episodes)
                ],
            )
            schedule_path = None
            if condition_name == "contact_skill":
                schedule_path = _donor_schedule_path(args)
                if schedule_path.exists():
                    raise RuntimeError(
                        "Refusing to overwrite an existing donor feedback "
                        f"schedule: {schedule_path}"
                    )
            def _save_partial_schedule(partial_schedule) -> None:
                # A crashed donor run is not a valid donor. It is kept under
                # a different name and condition label so it stays on disk
                # as evidence but can never be loaded for a yoked recipient.
                if condition_name != "contact_skill" or not partial_schedule:
                    return
                partial_path = args.feedback_output_dir / (
                    f"{args.culture_id}_seq{args.sequence_index}"
                    f"_contact_skill_incomplete_{int(time.time())}.json"
                )
                save_feedback_schedule(
                    partial_path,
                    partial_schedule,
                    metadata={
                        "culture_id": args.culture_id,
                        "sequence_index": args.sequence_index,
                        "condition": "contact_skill_incomplete",
                        "protocol_id": BENCHMARK_PROTOCOL_ID,
                        "backend": backend,
                        **provenance,
                    },
                )

            _, recorded_schedule = _run_condition(
                args=args,
                backend=backend,
                on_feedback_schedule=_save_partial_schedule,
                neurons=experiment_neurons,
                ranking=ranking,
                responsiveness=responsiveness,
                condition_name=condition_name,
                condition_order=recorded_order,
                yoked_schedule=yoked_schedule,
                provenance=provenance,
                channel_map=resolved_map,
                channel_health_report=health_report,
            )
            if condition_name == "contact_skill":
                assert schedule_path is not None
                save_feedback_schedule(
                    schedule_path,
                    recorded_schedule,
                    metadata={
                        "culture_id": args.culture_id,
                        "sequence_index": args.sequence_index,
                        "condition": condition_name,
                        "protocol_id": BENCHMARK_PROTOCOL_ID,
                        "backend": backend,
                        **provenance,
                    },
                )
                if simulator and not yoked_schedule:
                    yoked_schedule = recorded_schedule
                print(f"Donor feedback schedule: {schedule_path}")

    print(f"Benchmark rows appended to {args.output}")


if __name__ == "__main__":
    main()
