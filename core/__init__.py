"""
Senxe Cerebellum — Core Modules
=============================
Shared biological neural interface components for v3.0 and v4.0 demos.

Modules:
    neurons   — CL1 hardware interface
    decoder   — Antagonistic motor decoding (flexor/extensor differential)
    pdi       — Physical Disturbance Index (FEP-inspired explore/exploit gate)
    curiosity — Neural intrinsic curiosity (firing-pattern novelty detection)
    video     — Video generation utilities
"""

from core.neurons import (
    cl_open, warmup_calibration,
    ChannelSet, StimDesign, BurstDesign,
    CL_AVAILABLE,
)
from core.decoder import AntagonisticDecoder
from core.pdi import PDI
from core.curiosity import NeuralCuriosity

__all__ = [
    "cl_open", "warmup_calibration",
    "ChannelSet", "StimDesign", "BurstDesign",
    "CL_AVAILABLE",
    "AntagonisticDecoder",
    "PDI",
    "NeuralCuriosity",
]
