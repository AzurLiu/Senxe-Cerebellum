"""Tests for the private hybrid nominal-plus-residual control slice."""

import numpy as np

from core.hybrid_control import (
    BoundedResidualController,
    NominalControlConfig,
    NominalTaskController,
    ResidualControlConfig,
    TaskPhase,
    spike_confidence,
    summarize_control_reports,
)


def _observation(
    *,
    eef_to_nut=(0.1, 0.0, 0.0),
    peg_to_hole=(0.2, 0.0, 0.0),
    force=(0.0, 0.0, 0.0),
    grasp_confirmed=False,
    nut_on_peg=False,
    placement_success=False,
    nut_height_above_table_m=0.15,
    grasp_yaw_error_rad=0.0,
    nut_peg_yaw_error_rad=0.0,
):
    return {
        "eef_to_nut": np.asarray(eef_to_nut, dtype=float),
        "peg_to_hole": np.asarray(peg_to_hole, dtype=float),
        "force": np.asarray(force, dtype=float),
        "grasp_confirmed": grasp_confirmed,
        "nut_on_peg": nut_on_peg,
        "placement_success": placement_success,
        "nut_height_above_table_m": nut_height_above_table_m,
        "grasp_yaw_error_rad": grasp_yaw_error_rad,
        "nut_peg_yaw_error_rad": nut_peg_yaw_error_rad,
    }


def test_nominal_controller_approaches_nut_with_open_gripper():
    controller = NominalTaskController()

    action = controller.propose(_observation())

    assert controller.phase is TaskPhase.APPROACH_NUT
    assert action.shape == (7,)
    assert action[0] > 0.0
    assert action[-1] == -1.0
    assert np.linalg.norm(action[:3]) <= 0.60 + 1e-9


def test_nominal_controller_keeps_long_horizon_phase_outside_cl_path():
    controller = NominalTaskController(NominalControlConfig(
        grasp_confirm_steps=1,
    ))

    controller.propose(_observation(eef_to_nut=(0.0, 0.0, -0.08)))
    grasp_action = controller.propose(_observation(
        eef_to_nut=(0.0, 0.0, 0.03),
    ))
    assert controller.phase is TaskPhase.GRASP
    assert grasp_action[-1] == 1.0

    transport_action = controller.propose(_observation(
        eef_to_nut=(0.0, 0.0, 0.03),
        peg_to_hole=(0.2, 0.0, -0.1),
        grasp_confirmed=True,
    ))
    assert controller.phase is TaskPhase.TRANSPORT
    assert transport_action[0] > 0.0
    assert transport_action[-1] == 1.0


def test_nominal_controller_never_infers_grasp_from_elapsed_steps():
    controller = NominalTaskController(NominalControlConfig(
        grasp_timeout_steps=2,
    ))
    controller.propose(_observation(eef_to_nut=(0.0, 0.0, -0.08)))
    close = _observation(eef_to_nut=(0.0, 0.0, 0.03))

    controller.propose(close)
    assert controller.phase is TaskPhase.GRASP
    controller.propose(close)
    controller.propose(close)

    assert controller.phase is TaskPhase.APPROACH_NUT


def test_nominal_controller_requires_release_and_retreat_before_complete():
    controller = NominalTaskController(NominalControlConfig(
        release_hold_steps=2,
        retreat_min_steps=2,
    ))

    release = controller.propose(_observation(
        nut_on_peg=True,
        placement_success=True,
    ))
    assert controller.phase is TaskPhase.RELEASE
    assert release[-1] == -1.0

    controller.propose(_observation(
        nut_on_peg=True,
        placement_success=True,
    ))
    assert controller.phase is TaskPhase.RETREAT

    retreat = controller.propose(_observation(
        nut_on_peg=True,
        placement_success=True,
    ))
    assert controller.phase is TaskPhase.RETREAT
    assert retreat[2] > 0.0
    controller.propose(_observation(
        nut_on_peg=True,
        placement_success=True,
    ))
    assert controller.phase is TaskPhase.COMPLETE


def test_residual_controller_only_changes_configured_axis():
    controller = BoundedResidualController(ResidualControlConfig(
        residual_axis=2,
        residual_scale=0.1,
        max_residual_abs=0.05,
    ))
    baseline = np.array([0.1, -0.1, 0.0, 0.0, 0.0, 0.0, 1.0])
    neural = np.array([1.0, 1.0, 0.4, 1.0, 1.0, 1.0, -1.0])

    final, report = controller.compose(
        baseline,
        neural,
        phase=TaskPhase.INSERT,
        residual_confidence=1.0,
        force_n=1.0,
    )

    assert np.allclose(final[[0, 1, 3, 4, 5, 6]], baseline[[0, 1, 3, 4, 5, 6]])
    assert np.isclose(final[2], 0.04)
    assert report.mode == "baseline_plus_cl_residual"
    assert report.reason == "residual_applied"


def test_residual_is_clipped_to_private_authority_budget():
    controller = BoundedResidualController(ResidualControlConfig(
        residual_axis=2,
        residual_scale=1.0,
        max_residual_abs=0.03,
    ))

    final, report = controller.compose(
        np.zeros(7),
        np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0]),
        phase=TaskPhase.INSERT,
        residual_confidence=1.0,
        force_n=0.0,
    )

    assert np.isclose(final[2], 0.03)
    assert np.isclose(report.applied_residual, 0.03)


def test_low_confidence_falls_back_to_nominal_action():
    controller = BoundedResidualController(ResidualControlConfig(
        min_residual_confidence=0.5,
    ))
    baseline = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

    final, report = controller.compose(
        baseline,
        np.ones(7),
        phase=TaskPhase.APPROACH_NUT,
        residual_confidence=0.25,
        force_n=0.0,
    )

    assert np.allclose(final, baseline)
    assert report.mode == "baseline_only"
    assert report.reason == "low_residual_confidence"


def test_soft_force_limit_suppresses_only_cl_residual():
    controller = BoundedResidualController(ResidualControlConfig(
        force_soft_limit_n=10.0,
        force_hard_limit_n=20.0,
    ))
    baseline = np.array([0.05, 0.0, -0.02, 0.0, 0.0, 0.0, 1.0])

    final, report = controller.compose(
        baseline,
        np.ones(7),
        phase=TaskPhase.INSERT,
        residual_confidence=1.0,
        force_n=12.0,
    )

    assert np.allclose(final, baseline)
    assert report.reason == "force_soft_limit"


def test_hard_force_limit_stops_entire_action():
    controller = BoundedResidualController(ResidualControlConfig(
        force_hard_limit_n=20.0,
    ))

    final, report = controller.compose(
        np.ones(7) * 0.1,
        np.ones(7),
        phase=TaskPhase.INSERT,
        residual_confidence=1.0,
        force_n=20.0,
    )

    assert np.allclose(final, 0.0)
    assert report.mode == "hard_stop"
    assert report.reason == "force_hard_limit"


def test_protocol_phase_gate_keeps_calibration_on_baseline():
    controller = BoundedResidualController()
    baseline = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

    final, report = controller.compose(
        baseline,
        np.ones(7),
        phase=TaskPhase.APPROACH_NUT,
        residual_confidence=1.0,
        force_n=0.0,
        residual_enabled=False,
    )

    assert np.allclose(final, baseline)
    assert report.mode == "baseline_only"
    assert report.reason == "protocol_phase_gate"


def test_spike_confidence_is_unique_channel_count_and_bounded():
    assert spike_confidence([]) == 0.0
    assert spike_confidence([1, 1, 2], target_count=4) == 0.5
    assert spike_confidence(range(20), target_count=4) == 1.0


def test_episode_summary_tracks_residual_authority_and_stops():
    reports = [
        {
            "phase": "insert",
            "mode": "baseline_plus_cl_residual",
            "applied_residual": 0.02,
        },
        {
            "phase": "insert",
            "mode": "baseline_only",
            "applied_residual": 0.0,
        },
        {
            "phase": "insert",
            "mode": "hard_stop",
            "applied_residual": 0.0,
        },
    ]

    summary = summarize_control_reports(reports)

    assert summary["step_count"] == 3
    assert np.isclose(summary["residual_applied_rate"], 1 / 3)
    assert np.isclose(summary["mean_abs_applied_residual"], 0.02 / 3)
    assert summary["hard_stop_count"] == 1
    assert summary["phase_counts"] == {"insert": 3}
