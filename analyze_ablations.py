#!/usr/bin/env python3
"""Generate culture-level confidence intervals and falsification checks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from core.experiment_statistics import (
    bootstrap_difference,
    bootstrap_mean,
    culture_endpoint_values,
    is_retention_phase,
)


CONTROLS = (
    "baseline_only",
    "zero_spikes",
    "shuffled_spikes",
    "no_feedback",
    "yoked_feedback",
)
MIN_CONFIRMATORY_CULTURES = 6


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path, default=Path(
        "ablation_results.csv"
    ))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis"),
    )
    args = parser.parse_args()
    rows = _read_rows(args.input)
    if not rows:
        raise RuntimeError("ablation input contains no rows")
    required_fields = {
        "ChannelMapHash",
        "ChannelMapVersion",
        "ChannelHealthReport",
        "CodeHash",
        "Condition",
        "CultureID",
        "EpisodeForceSafe",
        "EpisodeSuccess",
        "ProtocolHash",
        "ProtocolPhase",
        "RuntimeVersions",
        "StimSafetyLimits",
        "EpisodeStimAbsChargeNc",
        "EpisodeStimCalls",
        "EpisodeStimChannelPulses",
    }
    missing_fields = required_fields - set(rows[0])
    if missing_fields:
        raise RuntimeError(
            "Ablation input is not a complete V2 evidence table; missing: "
            + ", ".join(sorted(missing_fields))
        )
    provenance_consistent = all(
        len({str(row[field]) for row in rows}) == 1
        and bool(str(rows[0][field]).strip())
        for field in (
            "ChannelMapHash",
            "CodeHash",
            "ProtocolHash",
            "RuntimeVersions",
            "StimSafetyLimits",
        )
    )
    conditions_by_culture: dict[str, set[str]] = {}
    for row in rows:
        conditions_by_culture.setdefault(
            str(row["CultureID"]),
            set(),
        ).add(str(row["Condition"]))
    independent_culture_design = not any(
        len(conditions) > 1
        for conditions in conditions_by_culture.values()
    )

    endpoints = {
        "EpisodeSuccess": culture_endpoint_values(
            rows,
            endpoint="EpisodeSuccess",
        ),
        "EpisodeForceSafe": culture_endpoint_values(
            rows,
            endpoint="EpisodeForceSafe",
        ),
    }
    summary_rows = []
    for endpoint, values_by_group in endpoints.items():
        for (condition, phase), values in sorted(values_by_group.items()):
            estimate = bootstrap_mean(values)
            summary_rows.append({
                "Endpoint": endpoint,
                "Condition": condition,
                "ProtocolPhase": phase,
                "CultureMean": estimate.mean,
                "CI95Lower": (
                    "" if estimate.lower is None else estimate.lower
                ),
                "CI95Upper": (
                    "" if estimate.upper is None else estimate.upper
                ),
                "BiologicalReplicates": estimate.biological_replicates,
                "IndependentCultureDesign": int(
                    independent_culture_design
                ),
                "ProvenanceConsistent": int(provenance_consistent),
                "ConfirmatoryValid": int(
                    independent_culture_design
                    and provenance_consistent
                    and estimate.biological_replicates
                    >= MIN_CONFIRMATORY_CULTURES
                ),
            })

    comparison_rows = []
    success_values = endpoints["EpisodeSuccess"]
    force_values = endpoints["EpisodeForceSafe"]
    phases = sorted({
        phase
        for condition, phase in success_values
        if condition == "contact_skill" and is_retention_phase(phase)
    })
    for phase in phases:
        treatment_success = success_values.get(
            ("contact_skill", phase),
            [],
        )
        treatment_force = force_values.get(("contact_skill", phase), [])
        for control in CONTROLS:
            success_difference = bootstrap_difference(
                treatment_success,
                success_values.get((control, phase), []),
                confidence=0.99,
            )
            force_difference = bootstrap_difference(
                treatment_force,
                force_values.get((control, phase), []),
                confidence=0.99,
            )
            valid = (
                success_difference.lower is not None
                and force_difference.lower is not None
            )
            comparison_rows.append({
                "ProtocolPhase": phase,
                "Control": control,
                "SuccessDifference": success_difference.mean,
                "SuccessCI99Lower": (
                    ""
                    if success_difference.lower is None
                    else success_difference.lower
                ),
                "SuccessCI99Upper": (
                    ""
                    if success_difference.upper is None
                    else success_difference.upper
                ),
                "ForceSafetyDifference": force_difference.mean,
                "ForceSafetyCI99Lower": (
                    ""
                    if force_difference.lower is None
                    else force_difference.lower
                ),
                "ForceSafetyCI99Upper": (
                    ""
                    if force_difference.upper is None
                    else force_difference.upper
                ),
                "BiologicalReplicates": (
                    success_difference.biological_replicates
                ),
                "IndependentCultureDesign": int(
                    independent_culture_design
                ),
                "ProvenanceConsistent": int(provenance_consistent),
                # Preregistered superiority plus a 5 percentage-point
                # force-safety non-inferiority margin.
                "PassesFalsificationGate": int(
                    valid
                    and independent_culture_design
                    and provenance_consistent
                    and success_difference.biological_replicates
                    >= MIN_CONFIRMATORY_CULTURES
                    and success_difference.lower > 0.0
                    and force_difference.lower > -0.05
                ),
            })

    _write(
        args.output_dir / "culture_level_summary.csv",
        list(summary_rows[0]) if summary_rows else [],
        summary_rows,
    )
    _write(
        args.output_dir / "preregistered_comparisons.csv",
        list(comparison_rows[0]) if comparison_rows else [],
        comparison_rows,
    )
    print(f"Analysis written to {args.output_dir}")


if __name__ == "__main__":
    main()
