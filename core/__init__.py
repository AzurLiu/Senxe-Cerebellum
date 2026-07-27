"""
Senxe Cerebellum — Core Modules
=============================
Shared biological neural interface components for v3.0 and v4.0 demos.

Modules:
    decoder   — Antagonistic motor decoding (flexor/extensor differential)
    hybrid_control — White-box nominal control with bounded CL1 residuals
    pdi       — Physical Disturbance Index (FEP-inspired explore/exploit gate)
    curiosity — Neural intrinsic curiosity (firing-pattern novelty detection)
    video     — Video generation utilities
"""

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
from core.pdi import PDI
from core.curiosity import NeuralCuriosity

__all__ = [
    "AntagonisticDecoder",
    "BoundedResidualController",
    "HybridControlReport",
    "NominalControlConfig",
    "NominalTaskController",
    "ResidualControlConfig",
    "TaskPhase",
    "spike_confidence",
    "summarize_control_reports",
    "PDI",
    "NeuralCuriosity",
]
