from types import SimpleNamespace

import numpy as np

from core.contact_skill import (
    CONTACT_SKILL_DIM,
    ContactFeedbackEvaluator,
    ContactSkillControlConfig,
    ContactSkillController,
    ContactStateEncoder,
    summarize_contact_reports,
)
from core.hybrid_control import TaskPhase


class _FakeNeurons:
    def __init__(self):
        self.stims = []

    def stim(self, channels, design, burst):
        self.stims.append((channels, design, burst))


def test_contact_encoder_uses_compact_position_force_and_phase_channels():
    neurons = _FakeNeurons()
    encoder = ContactStateEncoder(neurons)

    report = encoder.encode(
        {
            "peg_to_hole": np.array([0.01, -0.03, 0.0]),
            "force": np.array([2.0, 0.0, -7.0]),
        },
        TaskPhase.INSERT,
    )

    assert report.phase == "insert"
    assert report.contact_state == "contact"
    assert len(report.normalized_features) == 6
    assert len(report.stimulated_channels) == 6
    assert len(neurons.stims) == 6
    assert all(0 <= channel < 32 for channel in report.stimulated_channels)


def test_contact_skill_applies_bounded_xyz_residual():
    controller = ContactSkillController(ContactSkillControlConfig(
        residual_scales=(0.1, 0.1, 0.1),
        max_residual_abs=(0.04, 0.04, 0.03),
    ))
    baseline = np.array([0.01, -0.01, 0.0, 0.0, 0.0, 0.0, 1.0])
    skill = np.array([1.0, -1.0, 1.0, 0.0, 0.0])

    final, report = controller.compose(
        baseline,
        skill,
        phase=TaskPhase.INSERT,
        residual_confidence=1.0,
        force_vector_n=np.array([0.0, 0.0, 2.0]),
    )

    assert np.allclose(final[:3], [0.05, -0.05, 0.03])
    assert np.allclose(report.applied_residual_vector, [0.04, -0.04, 0.03])
    assert report.mode == "baseline_plus_cl_contact_skill"
    assert final[6] == 1.0


def test_soften_can_only_reduce_nominal_motion_authority():
    controller = ContactSkillController(ContactSkillControlConfig(
        min_compliance_scale=0.4,
    ))
    baseline = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    skill = np.array([0.0, 0.0, 0.0, 1.0, 0.0])

    final, report = controller.compose(
        baseline,
        skill,
        phase=TaskPhase.INSERT,
        residual_confidence=1.0,
        force_vector_n=np.array([1.0, 0.0, 0.0]),
    )

    assert np.isclose(report.compliance_scale, 0.4)
    assert np.isclose(final[0], 0.08)
    assert np.linalg.norm(final[:3]) <= np.linalg.norm(baseline[:3])


def test_retract_selection_executes_deterministic_safe_primitive():
    controller = ContactSkillController(ContactSkillControlConfig(
        retract_threshold=0.5,
        retract_speed=0.08,
    ))
    skill = np.zeros(CONTACT_SKILL_DIM)
    skill[4] = 0.8

    final, report = controller.compose(
        np.zeros(7),
        skill,
        phase=TaskPhase.INSERT,
        residual_confidence=1.0,
        force_vector_n=np.array([3.0, 4.0, 0.0]),
    )

    assert np.allclose(final[:3], [-0.048, -0.064, 0.0])
    assert report.retract_applied
    assert report.mode == "cl_selected_safe_retract"


def test_contact_skill_is_disabled_outside_contact_phases():
    controller = ContactSkillController()
    baseline = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])

    final, report = controller.compose(
        baseline,
        np.ones(CONTACT_SKILL_DIM),
        phase=TaskPhase.APPROACH_NUT,
        residual_confidence=1.0,
        force_vector_n=np.zeros(3),
    )

    assert np.allclose(final, baseline)
    assert report.reason == "outside_contact_phase"


def test_contact_feedback_distinguishes_progress_collision_and_neutral():
    evaluator = ContactFeedbackEvaluator()
    evaluator.reset(0.10)

    progress = evaluator.evaluate(
        distance_m=0.09,
        force_n=2.0,
        success=False,
    )
    collision = evaluator.evaluate(
        distance_m=0.09,
        force_n=20.0,
        success=False,
    )
    neutral = evaluator.evaluate(
        distance_m=0.09,
        force_n=0.0,
        success=False,
    )

    assert progress.kind == "contact_progress" and progress.valence == 1
    assert collision.kind == "collision" and collision.valence == -1
    assert neutral.kind == "neutral" and neutral.valence == 0


def test_contact_feedback_is_neutral_outside_contact_phase():
    evaluator = ContactFeedbackEvaluator()
    evaluator.reset(0.10)

    event = evaluator.evaluate(
        distance_m=0.05,
        force_n=20.0,
        success=False,
        enabled=False,
    )

    assert event.kind == "outside_contact_phase"
    assert not event.active


def test_contact_summary_tracks_xyz_compliance_and_retract():
    reports = [
        {
            "mode": "baseline_plus_cl_contact_skill",
            "phase": "insert",
            "applied_residual_vector": [0.01, 0.02, 0.0],
            "compliance_scale": 0.7,
            "retract_applied": False,
        },
        {
            "mode": "cl_selected_safe_retract",
            "phase": "insert",
            "applied_residual_vector": [0.0, 0.0, 0.0],
            "compliance_scale": 1.0,
            "retract_applied": True,
        },
    ]

    summary = summarize_contact_reports(reports)

    assert summary["residual_applied_rate"] == 0.5
    assert summary["mean_residual_norm"] > 0
    assert summary["mean_compliance_scale"] == 0.85
    assert summary["retract_count"] == 1
