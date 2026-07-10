"""
Senxe Cerebellum — Neural Interface Module
========================================
CL1 biological neural interface: Pure hardware mode.

This module provides the hardware abstraction layer for interfacing with
Cortical Labs CL1 biological neural organoids on a 64-channel MEA.
"""

from __future__ import annotations

import os
import numpy as np
from contextlib import contextmanager
from typing import Tuple

# Compatibility shim: cl-sdk uses np.bool which was removed in numpy 2.0
if not hasattr(np, 'bool'):
    np.bool = np.bool_
if not hasattr(np, 'concat'):
    np.concat = np.concatenate

try:
    import cl as _cl_sdk
    from cl import ChannelSet, StimDesign, BurstDesign
    from cl.neurons import Neurons
    CL_AVAILABLE = True
except ImportError:
    raise RuntimeError(
        "FATAL ERROR: cl-sdk is not installed. \n"
        "This project requires the official Cortical Labs SDK (cl-sdk) to run. \n"
        "Please run: pip install cl-sdk"
    )

# Monkeypatch cl.neurons.Neurons.close to be idempotent
if CL_AVAILABLE:
    _original_close = Neurons.close
    def _idempotent_close(self):
        if getattr(self, "_closed", False):
            return
        _original_close(self)
        self._closed = True
    Neurons.close = _idempotent_close

def is_cl_simulator() -> bool:
    """Returns True if cl-sdk is running in simulation mode (no real CL1 hardware).
    
    Detection: The official cl-sdk ships as a mock/simulator that replays
    Poisson-sampled recordings. We detect this by checking whether the
    Neurons class has the mock-only '_replay_file' attribute in its
    annotations, which only exists in the simulator implementation.
    """
    try:
        from cl.neurons import Neurons
        return '_replay_file' in getattr(Neurons, '__annotations__', {})
    except Exception:
        return False

@contextmanager
def cl_open():
    """Unified CL1 entry point: connects to real Cortical Labs biological hardware."""
    with _cl_sdk.open() as neurons:
        yield neurons

def warmup_calibration(
    neurons,
    duration_sec: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Channel warm-up calibration — probe all 64 channels for responsiveness.

    Phase 1 (Baseline): Read spontaneous activity.
    Phase 2 (Probing): Deliver standard biphasic pulse to every channel,
    measure evoked response.

    Returns:
        channel_ranking: Indices sorted by responsiveness.
        responsiveness: Response delta (evoked - baseline).
    """
    print("  [Calibration] Channel warm-up calibrating on real biology...")
    n_rounds = int(duration_sec * 250)  # 10s * 250 rounds of 100 frames @ 25kHz = 250,000 frames
    baseline_rounds = n_rounds // 2
    stim_rounds = n_rounds - baseline_rounds

    baseline_responses = np.zeros(64)
    for _ in range(baseline_rounds):
        frames = neurons.read(100, None)
        baseline_responses += np.mean(np.abs(frames.astype(float)), axis=0)
    baseline_responses /= max(1, baseline_rounds)

    stim = StimDesign(160, -1.0, 160, 1.0)
    burst = BurstDesign(1, 50)  # Single weak pulse to avoid global seizure
    stim_responses = np.zeros(64)
    rounds_per_ch = max(1, stim_rounds // 64)

    # Sequentially stimulate channels to measure network excitability without causing a culture seizure
    for ch in range(64):
        neurons.stim(ChannelSet(ch), stim, burst)
        for _ in range(rounds_per_ch):
            frames = neurons.read(100, None)
            stim_responses += np.mean(np.abs(frames.astype(float)), axis=0)
            
    stim_responses /= max(1, 64 * rounds_per_ch)

    responsiveness = stim_responses - baseline_responses
    channel_ranking = np.argsort(responsiveness)[::-1]

    top8 = channel_ranking[:8]
    print(f"  [Calibration] Done! Top-8: {top8.tolist()} "
          f"range: {responsiveness[top8[0]]:.1f}~{responsiveness[top8[-1]]:.1f}")
    return channel_ranking, responsiveness

