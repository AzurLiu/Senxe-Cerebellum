from core.learning_protocol import (
    EvidenceLevel,
    LearningPhase,
    default_protocol_path,
    load_protocols,
)


def test_default_protocol_is_evidence_bounded_and_frozen_for_evaluation():
    protocols = load_protocols(default_protocol_path())
    protocol = protocols["senxe_force_residual_v1"]

    assert protocol.evidence_level is EvidenceLevel.PROJECT_DESIGN
    assert "do not establish biological learning" in protocol.claim_boundary
    assert protocol.artifact_wait_ms == 50.0
    assert "zero_spikes" in protocol.controls

    evaluation = protocol.phases[-1]
    assert evaluation.phase is LearningPhase.FROZEN_EVALUATION
    assert evaluation.encoder_frozen
    assert evaluation.decoder_frozen
    assert not evaluation.biological_feedback_enabled


def test_protocol_phase_boundaries_are_external_and_deterministic():
    protocol = load_protocols(default_protocol_path())[
        "senxe_force_residual_v1"
    ]

    assert protocol.phase_for_episode(0).phase is LearningPhase.CALIBRATION
    assert protocol.phase_for_episode(4).phase is LearningPhase.CALIBRATION
    assert protocol.phase_for_episode(5).phase is LearningPhase.FROZEN_BASELINE
    assert protocol.phase_for_episode(15).phase is LearningPhase.FEEDBACK_TRAINING
    assert protocol.phase_for_episode(55).phase is LearningPhase.FROZEN_EVALUATION
    assert protocol.phase_for_episode(500).phase is LearningPhase.FROZEN_EVALUATION

