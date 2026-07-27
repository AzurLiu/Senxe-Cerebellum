"""
Senxe Cerebellum — Core Modules
=============================
Shared biological neural interface components.

Modules:
    contact_skill — Default compact encoder and five-output skill boundary
    contact_scenarios — Train/held-out MuJoCo physics perturbations
    spike_pipeline — Timestamped, artifact-separated CL spike features
    hybrid_control — Nominal task planner and legacy one-axis residual
"""

from core.contact_skill import (
    CONTACT_SKILL_DIM,
    ContactFeedbackEvaluator,
    ContactSkillController,
    ContactStateEncoder,
    summarize_contact_reports,
)
from core.decoder import AntagonisticDecoder
from core.hybrid_control import (
    BoundedResidualController,
    HybridControlReport,
    NominalControlConfig,
    NominalTaskController,
    ResidualControlConfig,
    TaskPhase,
    spike_confidence,
    summarize_control_reports,
)
__all__ = [
    "CONTACT_SKILL_DIM",
    "ContactFeedbackEvaluator",
    "ContactSkillController",
    "ContactStateEncoder",
    "summarize_contact_reports",
    "AntagonisticDecoder",
    "BoundedResidualController",
    "HybridControlReport",
    "NominalControlConfig",
    "NominalTaskController",
    "ResidualControlConfig",
    "TaskPhase",
    "spike_confidence",
    "summarize_control_reports",
]
