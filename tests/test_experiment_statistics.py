import numpy as np

from core.experiment_statistics import (
    bootstrap_difference,
    bootstrap_mean,
    culture_endpoint_values,
)


def test_single_culture_is_not_given_a_false_confidence_interval():
    estimate = bootstrap_mean([0.5])

    assert estimate.mean == 0.5
    assert estimate.lower is None
    assert estimate.upper is None
    assert estimate.biological_replicates == 1


def test_culture_is_the_unit_of_replication():
    rows = [
        {
            "Condition": "contact_skill",
            "ProtocolPhase": "frozen_evaluation",
            "CultureID": "a",
            "EpisodeSuccess": value,
        }
        for value in (1, 1, 0)
    ] + [
        {
            "Condition": "contact_skill",
            "ProtocolPhase": "frozen_evaluation",
            "CultureID": "b",
            "EpisodeSuccess": value,
        }
        for value in (0, 0, 0)
    ]

    values = culture_endpoint_values(rows, endpoint="EpisodeSuccess")

    assert np.allclose(
        sorted(values[("contact_skill", "frozen_evaluation")]),
        [0.0, 2 / 3],
    )


def test_bootstrap_difference_reports_direction():
    estimate = bootstrap_difference(
        [0.8, 0.9, 1.0],
        [0.1, 0.2, 0.3],
        resamples=1000,
    )

    assert estimate.mean > 0
    assert estimate.lower is not None and estimate.lower > 0
