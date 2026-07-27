"""Evidence-bounded, frozen learning protocol definitions.

Protocol configuration is data, not an implicit collection of training-loop
side effects.  This lets simulator, hardware and ablation runs share the same
phase boundaries and makes post-hoc changes visible.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class EvidenceLevel(str, Enum):
    PEER_REVIEWED = "peer_reviewed"
    REAL_CL1_CODE = "real_cl1_code"
    REAL_CL1_RESPONSE = "real_cl1_response"
    CONCEPTUAL_TRANSFER = "conceptual_transfer"
    PROJECT_DESIGN = "project_design"


class LearningPhase(str, Enum):
    CALIBRATION = "calibration"
    READOUT_FIT = "readout_fit"
    FROZEN_BASELINE = "frozen_baseline"
    FEEDBACK_TRAINING = "feedback_training"
    FROZEN_EVALUATION = "frozen_evaluation"


@dataclass(frozen=True)
class PhaseSpec:
    phase: LearningPhase
    episodes: int
    encoder_frozen: bool
    decoder_frozen: bool
    biological_feedback_enabled: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PhaseSpec":
        episodes = int(value["episodes"])
        if episodes <= 0:
            raise ValueError("phase episodes must be positive")
        return cls(
            phase=LearningPhase(value["phase"]),
            episodes=episodes,
            encoder_frozen=bool(value["encoder_frozen"]),
            decoder_frozen=bool(value["decoder_frozen"]),
            biological_feedback_enabled=bool(
                value["biological_feedback_enabled"]
            ),
        )


@dataclass(frozen=True)
class ProtocolSpec:
    protocol_id: str
    evidence_level: EvidenceLevel
    source_url: str | None
    claim_boundary: str
    artifact_wait_ms: float
    collect_window_ms: float
    encoder_policy: str
    decoder_policy: str
    feedback_policy: str
    controls: tuple[str, ...]
    required_recordings: tuple[str, ...]
    phases: tuple[PhaseSpec, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ProtocolSpec":
        protocol_id = str(value["protocol_id"]).strip()
        if not protocol_id:
            raise ValueError("protocol_id cannot be empty")
        artifact_wait_ms = float(value["artifact_wait_ms"])
        collect_window_ms = float(value["collect_window_ms"])
        if artifact_wait_ms < 0 or collect_window_ms <= 0:
            raise ValueError("protocol response windows are invalid")
        phases = tuple(
            PhaseSpec.from_mapping(phase) for phase in value["phases"]
        )
        if not phases:
            raise ValueError("protocol must declare at least one phase")
        return cls(
            protocol_id=protocol_id,
            evidence_level=EvidenceLevel(value["evidence_level"]),
            source_url=value.get("source_url"),
            claim_boundary=str(value["claim_boundary"]),
            artifact_wait_ms=artifact_wait_ms,
            collect_window_ms=collect_window_ms,
            encoder_policy=str(value["encoder_policy"]),
            decoder_policy=str(value["decoder_policy"]),
            feedback_policy=str(value["feedback_policy"]),
            controls=tuple(str(item) for item in value["controls"]),
            required_recordings=tuple(
                str(item) for item in value["required_recordings"]
            ),
            phases=phases,
        )

    def phase_for_episode(self, episode_index: int) -> PhaseSpec:
        if episode_index < 0:
            raise ValueError("episode_index must be non-negative")
        cursor = 0
        for phase in self.phases:
            cursor += phase.episodes
            if episode_index < cursor:
                return phase
        return self.phases[-1]


def load_protocols(path: str | Path) -> dict[str, ProtocolSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    protocols = {
        item["protocol_id"]: ProtocolSpec.from_mapping(item)
        for item in payload["protocols"]
    }
    if len(protocols) != len(payload["protocols"]):
        raise ValueError("protocol_id values must be unique")
    return protocols


def default_protocol_path() -> Path:
    return Path(__file__).resolve().parent.parent / "config" / "cl1_protocols.json"
