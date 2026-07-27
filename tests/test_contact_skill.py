import numpy as np
import pytest

from core.channel_map import (
    DEFAULT_CONTACT_CHANNEL_MAP,
    NON_STIMULATABLE_CHANNELS,
)
from core.contact_skill import (
    CONTACT_SKILL_DIM,
    ContactEncoderConfig,
    ContactFeedbackConfig,
    ContactFeedbackEvaluator,
    ContactFeedbackStimulator,
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
            "nut_to_peg": np.array([-0.02, 0.01, 0.0]),
            "force": np.array([2.0, 0.0, -7.0]),
        },
        TaskPhase.INSERT,
    )

    assert report.phase == "insert"
    assert report.contact_state == "contact"
    assert len(report.normalized_features) == 6
    assert len(report.stimulated_channels) == 6
    assert len(neurons.stims) == 6
    assert set(report.stimulated_channels) <= set(
        DEFAULT_CONTACT_CHANNEL_MAP.sensory_channels
    )
    assert not (
        set(report.stimulated_channels) & NON_STIMULATABLE_CHANNELS
    )
    assert report.position_error_m == [-0.02, 0.01, 0.0]


def test_three_phase_channels_encode_all_seven_phases_uniquely():
    neurons = _FakeNeurons()
    encoder = ContactStateEncoder(neurons)
    patterns = []

    for phase in TaskPhase:
        report = encoder.encode(
            {
                "nut_to_peg": np.zeros(3),
                "force": np.zeros(3),
            },
            phase,
        )
        patterns.append(tuple(
            channel
            for channel in report.stimulated_channels
            if channel in DEFAULT_CONTACT_CHANNEL_MAP.phase_channels
        ))

    assert len(set(patterns)) == len(TaskPhase)
    assert all(pattern for pattern in patterns)


def test_contact_stimulation_configs_reject_forbidden_channels():
    with pytest.raises(ValueError, match="non-stimulatable"):
        ContactEncoderConfig(
            position_channels=(0, 9, 10, 17, 18, 25),
        )
    with pytest.raises(ValueError, match="non-stimulatable"):
        ContactFeedbackStimulator(
            _FakeNeurons(),
            positive_channels=(19, 20, 56),
        )


def test_contact_encoder_can_compute_without_delivering_sensory_stimulation():
    neurons = _FakeNeurons()
    encoder = ContactStateEncoder(neurons)

    report = encoder.encode(
        {
            "nut_to_peg": np.array([0.01, 0.0, -0.02]),
            "force": np.array([1.0, 0.0, 0.0]),
        },
        TaskPhase.TRANSPORT,
        stimulation_enabled=False,
    )

    assert report.stimulated_channels == []
    assert neurons.stims == []
    assert report.position_error_m == [0.01, 0.0, -0.02]


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
    evaluator = ContactFeedbackEvaluator(ContactFeedbackConfig(
        cooldown_steps=0,
    ))
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


def test_contact_feedback_cooldown_suppresses_dense_repeated_events():
    evaluator = ContactFeedbackEvaluator(ContactFeedbackConfig(
        cooldown_steps=3,
    ))
    evaluator.reset(0.10)

    first = evaluator.evaluate(
        distance_m=0.09,
        force_n=2.0,
        success=False,
    )
    second = evaluator.evaluate(
        distance_m=0.08,
        force_n=2.0,
        success=False,
    )
    evaluator.evaluate(
        distance_m=0.08,
        force_n=0.0,
        success=False,
    )
    fourth = evaluator.evaluate(
        distance_m=0.07,
        force_n=2.0,
        success=False,
    )

    assert first.active
    assert second.kind == "cooldown_suppressed_contact_progress"
    assert not second.active
    assert fourth.active


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
