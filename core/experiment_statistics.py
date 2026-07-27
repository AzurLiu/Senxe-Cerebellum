"""Culture-level summaries for preregistered Senxe endpoints."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BootstrapEstimate:
    mean: float
    lower: float | None
    upper: float | None
    biological_replicates: int


def bootstrap_mean(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 5000,
    seed: int = 42,
) -> BootstrapEstimate:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return BootstrapEstimate(float("nan"), None, None, 0)
    mean = float(np.mean(data))
    if data.size < 2:
        return BootstrapEstimate(mean, None, None, int(data.size))
    rng = np.random.default_rng(seed)
    samples = rng.choice(data, size=(resamples, data.size), replace=True)
    means = np.mean(samples, axis=1)
    tail = (1.0 - confidence) / 2.0
    return BootstrapEstimate(
        mean=mean,
        lower=float(np.quantile(means, tail)),
        upper=float(np.quantile(means, 1.0 - tail)),
        biological_replicates=int(data.size),
    )


def bootstrap_difference(
    treatment: Sequence[float],
    control: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 5000,
    seed: int = 42,
) -> BootstrapEstimate:
    treatment_values = np.asarray(treatment, dtype=float)
    control_values = np.asarray(control, dtype=float)
    treatment_values = treatment_values[np.isfinite(treatment_values)]
    control_values = control_values[np.isfinite(control_values)]
    replicate_count = min(treatment_values.size, control_values.size)
    if treatment_values.size == 0 or control_values.size == 0:
        return BootstrapEstimate(float("nan"), None, None, 0)
    difference = float(
        np.mean(treatment_values) - np.mean(control_values)
    )
    if treatment_values.size < 2 or control_values.size < 2:
        return BootstrapEstimate(difference, None, None, replicate_count)
    rng = np.random.default_rng(seed)
    treatment_samples = rng.choice(
        treatment_values,
        size=(resamples, treatment_values.size),
        replace=True,
    )
    control_samples = rng.choice(
        control_values,
        size=(resamples, control_values.size),
        replace=True,
    )
    differences = (
        np.mean(treatment_samples, axis=1)
        - np.mean(control_samples, axis=1)
    )
    tail = (1.0 - confidence) / 2.0
    return BootstrapEstimate(
        mean=difference,
        lower=float(np.quantile(differences, tail)),
        upper=float(np.quantile(differences, 1.0 - tail)),
        biological_replicates=replicate_count,
    )


def culture_endpoint_values(
    rows: Iterable[Mapping[str, Any]],
    *,
    endpoint: str,
) -> dict[tuple[str, str], list[float]]:
    """Aggregate episodes within culture before comparing conditions."""

    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        condition = str(row["Condition"])
        phase = str(row["ProtocolPhase"])
        culture = str(row["CultureID"])
        grouped[(condition, phase, culture)].append(float(row[endpoint]))

    by_condition_phase: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (condition, phase, _culture), values in grouped.items():
        by_condition_phase[(condition, phase)].append(float(np.mean(values)))
    return dict(by_condition_phase)


def is_retention_phase(phase: str) -> bool:
    return phase in {
        "frozen_evaluation",
        "retention_5_min",
        "retention_15_min",
        "retention_45_min",
    }
