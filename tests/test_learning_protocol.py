from core.learning_protocol import (
    EvidenceLevel,
    LearningPhase,
    default_protocol_path,
    load_protocols,
)


def test_default_protocol_is_evidence_bounded_and_frozen_for_evaluation():
    protocols = load_protocols(default_protocol_path())
    protocol = protocols["senxe_contact_skill_v2"]

    assert protocol.evidence_level is EvidenceLevel.PROJECT_DESIGN
    assert "do not establish biological learning" in protocol.claim_boundary
    assert protocol.artifact_wait_ms == 50.0
    assert "zero_spikes" in protocol.controls

    evaluation = protocol.phases[-1]
    assert evaluation.phase is LearningPhase.RETENTION_45_MIN
    assert evaluation.encoder_frozen
    assert evaluation.decoder_frozen
    assert not evaluation.biological_feedback_enabled
    assert evaluation.rest_before_seconds == 1800
    assert evaluation.heldout_evaluation


def test_protocol_phase_boundaries_are_external_and_deterministic():
    protocol = load_protocols(default_protocol_path())[
        "senxe_contact_skill_v2"
    ]

    assert protocol.phase_for_episode(0).phase is LearningPhase.CALIBRATION
    assert protocol.phase_for_episode(4).phase is LearningPhase.CALIBRATION
    assert protocol.phase_for_episode(5).phase is LearningPhase.FROZEN_BASELINE
    assert protocol.phase_for_episode(30).phase is LearningPhase.FEEDBACK_TRAINING
    assert protocol.phase_for_episode(80).phase is LearningPhase.FROZEN_EVALUATION
    assert protocol.phase_for_episode(100).phase is LearningPhase.RETENTION_5_MIN
    assert protocol.phase_for_episode(120).phase is LearningPhase.RETENTION_15_MIN
    assert protocol.phase_for_episode(140).phase is LearningPhase.RETENTION_45_MIN
    assert protocol.phase_for_episode(500).phase is LearningPhase.RETENTION_45_MIN
    assert protocol.total_episodes == 160


def test_application_protocol_is_short_and_claim_bounded():
    protocol = load_protocols(default_protocol_path())[
        "senxe_contact_skill_application_v1"
    ]

    assert protocol.total_episodes == 22
    assert len(protocol.phases) == 4
    assert "cannot establish retained biological learning" in (
        protocol.claim_boundary
    )
