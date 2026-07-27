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
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from tqdm import tqdm

from core.channel_health import (
    ChannelHealthReport,
    require_channel_health,
    resolve_contact_channel_map,
)
from core.channel_map import ContactChannelMap, channel_map_metadata
from core.learning_protocol import default_protocol_path, load_protocols
from core.provenance import project_provenance
from core.neurons import (
    BudgetedNeurons,
    cl_open,
    is_cl_simulator,
    require_stimulation_approval,
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
    MIN_INPUT_RESPONSE_DELTA,
    MIN_OUTPUT_RESPONSE_DELTA,
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
            "Real CL1 runs require a non-placeholder --culture-id"
        )
    if not RECORD_CL_SESSION:
        raise RuntimeError(
            "Real CL1 ablations require SENXE_RECORD_SESSION=1"
        )
    if args.output.exists() and args.output.stat().st_size > 0:
        with args.output.open(newline="", encoding="utf-8") as handle:
            prior_rows = list(csv.DictReader(handle))
        prior_conditions = {
            str(row.get("Condition", ""))
            for row in prior_rows
            if str(row.get("CultureID", "")) == args.culture_id
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
            rows.append(row)
            progress.set_postfix(
                R=f"{reward:.2f}",
                success=int(success_pct > 0),
                FSR=f"{force_safe_rate:.0f}%",
                phase=phase.phase.value,
            )
        return rows, dict(agent.recorded_feedback_schedule)
    finally:
        if agent is not None:
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

    with cl_open() as neurons:
        simulator = is_cl_simulator()
        _validate_hardware_scope(args, simulator)
        require_stimulation_approval()
        experiment_neurons = BudgetedNeurons(
            neurons,
            safety_limits=STIM_SAFETY_LIMITS,
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
                calibration.input_responsiveness,
                calibration.output_responsiveness,
                input_threshold=MIN_INPUT_RESPONSE_DELTA,
                output_threshold=MIN_OUTPUT_RESPONSE_DELTA,
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
                schedule_path = args.feedback_output_dir / (
                    f"{args.culture_id}_seq{args.sequence_index}"
                    "_contact_skill.json"
                )
                if schedule_path.exists():
                    raise RuntimeError(
                        "Refusing to overwrite an existing donor feedback "
                        f"schedule: {schedule_path}"
                    )
            rows, recorded_schedule = _run_condition(
                args=args,
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
                        **provenance,
                    },
                )
                if simulator and not yoked_schedule:
                    yoked_schedule = recorded_schedule
                print(f"Donor feedback schedule: {schedule_path}")
            _append_rows(args.output, rows)

    print(f"Benchmark rows appended to {args.output}")


if __name__ == "__main__":
    main()
