"""
Senxe Cerebellum — Virtual Interference Encoding (VIE)
====================================================
Maps physical sensor data (force, torque, position, goal) to neural
stimulation patterns on a 64-channel MEA using biologically realistic
coding schemes.

Channel Layout (non-overlapping):
    CH  0-5:  Force magnitude — rate coding
    CH  6-11: Force per-axis — directional encoding
    CH 12-15: Force transient — traveling wave (dynamic sliding feedback)
    CH 16-27: Torque/friction — traveling wave temporal coding
    CH 28-31: Velocity supplement (independent)
    CH 32-46: End-effector position
    CH 47-54: Goal direction (peg-to-hole delta vector)
    CH 55-59: Insertion depth progress
    CH 60-63: Reserved

Usage::

    from core.vie import VIE
    vie = VIE(neurons, force_threshold=20.0, depth_threshold=0.02)
    vie.encode(obs_info)
    vie.adapt(firing_rates)
"""

from __future__ import annotations

import numpy as np

from core.neurons import ChannelSet, StimDesign, BurstDesign


class VIE:
    """Virtual Interference Encoding — maps physical sensor data to neural stimulation.

    Encodes force, torque, position, goal direction, and insertion depth onto
    a 64-channel MEA using biologically realistic coding schemes:

    - CH 0-5:   Force magnitude → rate coding
                (higher force → higher burst frequency, mimicking mechanoreceptors)
    - CH 6-11:  Force per-axis → directional encoding
    - CH 12-15: Force transient → traveling wave (dynamic sliding feedback)
    - CH 16-27: Torque and friction → traveling wave temporal coding
                (phase-delayed pulses simulate proprioceptive spindle fibers)
    - CH 28-31: Velocity supplement (independent, not reusing torque channels)
    - CH 32-46: End-effector absolute position → position encoding
    - CH 47-54: Goal direction (peg-to-hole delta vector)
    - CH 55-59: Insertion depth progress encoding
    - CH 60-63: Reserved for future use

    Includes online adaptive gain adjustment: channels with weak responses
    get amplified, over-responsive channels get attenuated.

    Args:
        neurons: CL1 neurons instance (real or mock) with .stim() method.
        force_threshold: Force safety threshold (N) for normalizing force encoding.
        depth_threshold: Insertion depth threshold (m) for normalizing depth encoding.
        raw_env: Optional raw RoboSuite environment reference.
    """

    # ── Non-overlapping channel layout ──
    CH_FORCE_MAG   = list(range(0, 6))     # Force magnitude rate coding
    CH_FORCE_AXIS  = list(range(6, 12))    # Per-axis force direction encoding
    CH_FORCE_WAVE  = list(range(12, 16))   # Force transient traveling wave
    CH_TORQUE      = list(range(16, 28))   # Torque / friction
    CH_VELOCITY    = list(range(28, 32))   # Velocity supplement (independent)
    CH_POSITION    = list(range(32, 47))   # End-effector position
    CH_GOALDIR     = list(range(47, 55))   # Goal direction
    CH_DEPTH       = list(range(55, 60))   # Insertion depth
    CH_RESERVED    = list(range(60, 64))   # Reserved

    def __init__(self, neurons, force_threshold=20.0, depth_threshold=0.02,
                 raw_env=None):
        self.neurons = neurons
        self.raw_env = raw_env
        self.force_threshold = force_threshold
        self.depth_threshold = depth_threshold
        self.channel_gain = np.ones(64)      # Per-channel encoding gain (online-adjusted)
        self.stim_history = np.zeros(64)     # Cumulative stimulation tracker
        self.response_history = np.zeros(64) # Cumulative response tracker
        self.adaptation_rate = 0.005

    def encode(self, obs_info):
        """Encode observation into neural stimulation patterns on the 64-ch MEA.

        Converts force/torque/position/goal sensor readings into charge-balanced
        biphasic pulse trains delivered across the channel groups. Force uses
        rate coding (burst frequency proportional to magnitude), while torque
        uses traveling-wave temporal coding with phase delays.

        Args:
            obs_info: Dictionary with keys 'eef_pos', 'eef_vel', 'force',
                      'torque', 'peg_to_hole' from extract_obs().
        """
        eef_pos = obs_info["eef_pos"]; eef_vel = obs_info["eef_vel"]
        force = obs_info["force"]; torque = obs_info["torque"]
        peg_to_hole = obs_info["peg_to_hole"]
        distance = np.linalg.norm(peg_to_hole)
        direction = peg_to_hole / (distance + 1e-8)
        stim = StimDesign(160, -1.0, 160, 1.0)

        # ══ Force Magnitude — Rate Coding (CH 0-5) ══
        force_mag = np.linalg.norm(force)
        fnorm = np.clip(force_mag / self.force_threshold, 0.0, 1.5)
        fhz = int(np.clip(50 + 350 * fnorm, 50, 400))
        fn = max(1, min(10, int(fnorm * 8 * self.channel_gain[self.CH_FORCE_MAG[0]])))
        self.neurons.stim(ChannelSet(*self.CH_FORCE_MAG), stim, BurstDesign(fn, fhz))

        # ══ Force Axis Encoding (CH 6-11) — per-axis direction ══
        for ax in range(3):
            cb = self.CH_FORCE_AXIS[ax * 2]
            chs = ChannelSet(*[cb, cb + 1])
            f = force[ax]; inten = np.clip(abs(f) / (self.force_threshold / 3), 0.1, 2.0)
            fs = StimDesign(160, -inten * np.sign(f), 160, inten * np.sign(f))
            fb = BurstDesign(max(1, int(abs(f) / 3)), int(50 + abs(f) * 15))
            self.neurons.stim(chs, fs, fb)

        # ══ Force Wave Encoding (CH 12-15) — dynamic sliding feedback ══
        if force_mag > 0.05:
            for wave_i in range(4):
                ch_idx = self.CH_FORCE_WAVE[wave_i]
                phase_delay_ms = 1.0 + 3.0 * (wave_i / 3.0)
                wave_freq = int(np.clip(40 + 160 * fnorm, 40, 200))
                wave_amp = np.clip(fnorm * 0.6, 0.05, 1.2)
                wave_stim = StimDesign(
                    int(160 + phase_delay_ms * 10), -wave_amp,
                    int(160 + phase_delay_ms * 10),  wave_amp
                )
                self.neurons.stim(ChannelSet(ch_idx), wave_stim, BurstDesign(1, wave_freq))

        # ══ Torque/Friction — Traveling Waves (CH 16-27) ══
        tmag = np.linalg.norm(torque)
        if tmag > 0.01:
            for ax in range(3):
                t = torque[ax]
                if abs(t) > 0.01:
                    cb = self.CH_TORQUE[0] + ax * 4
                    chs = ChannelSet(*range(cb, min(cb + 4, 28)))
                    inten = np.clip(abs(t) * 3.0 * self.channel_gain[cb], 0.1, 2.0)
                    ws = StimDesign(160, -inten, 160, inten)
                    whz = int(np.clip(60 * abs(t), 20, 200))
                    self.neurons.stim(chs, ws, BurstDesign(2, whz))

        # ══ Velocity Supplement (CH 28-31) — independent, no reuse ══
        vmag = np.linalg.norm(eef_vel)
        if vmag > 0.003:
            for ax in range(3):
                v = eef_vel[ax]
                if abs(v) > 0.003 and ax < len(self.CH_VELOCITY):
                    cb = self.CH_VELOCITY[ax]
                    vi = np.clip(abs(v) * 5, 0.1, 2.0)
                    vs = StimDesign(160, -vi, 160, vi)
                    vhz = int(np.clip(60 * abs(v), 20, 200))
                    self.neurons.stim(ChannelSet(cb), vs, BurstDesign(2, vhz))

        # ══ Position Encoding (CH 32-46) ══
        for ax in range(3):
            cb = self.CH_POSITION[0] + ax * 5
            chs = ChannelSet(*range(cb, min(cb + 5, 47)))
            p = eef_pos[ax]; g = self.channel_gain[cb]
            phz = int(np.clip((100 + 200 * abs(p)) * g, 50, 350))
            ps = StimDesign(160, -abs(p) * 0.8 * g, 160, abs(p) * 0.8 * g)
            self.neurons.stim(chs, ps, BurstDesign(2, phz))

        # ══ Goal Direction (CH 47-54) ══
        for ax in range(3):
            cb = self.CH_GOALDIR[0] + ax * 2
            chs = ChannelSet(*[cb, min(cb + 1, 54)])
            d = direction[ax]; inten = np.clip((abs(d) * 1.5 + 0.1) * self.channel_gain[cb], 0.1, 2.0)
            ds = StimDesign(160, -inten * np.sign(d), 160, inten * np.sign(d))
            db = BurstDesign(max(1, int(abs(d) * 5)), int(50 + abs(d) * 100))
            self.neurons.stim(chs, ds, db)

        # ══ Insertion Depth (CH 55-59) ══
        depth = self._compute_depth(obs_info)
        dn = np.clip(depth / self.depth_threshold, 0.0, 2.0)
        dg = np.mean(self.channel_gain[self.CH_DEPTH[:5]])
        dhz = int(np.clip((50 + 300 * dn) * dg, 50, 400)); dnn = max(1, int(dn * 6 * dg))
        dstim = StimDesign(160, -0.8, 160, 0.8)
        self.neurons.stim(ChannelSet(*self.CH_DEPTH), dstim, BurstDesign(dnn, dhz))

    def adapt(self, firing_rates):
        """Online adaptation of channel encoding gains.

        Implements homeostatic gain control: under-responsive channels get
        amplified, over-responsive channels get attenuated. Target is uniform
        response (~0.5 normalized) across all channels.

        Args:
            firing_rates: Per-channel firing rates, shape (64,).
        """
        fr_norm = firing_rates / (firing_rates.max() + 1e-6)
        # Target: uniform response across channels (~0.5 normalized)
        error = 0.5 - fr_norm
        self.channel_gain += self.adaptation_rate * error
        self.channel_gain = np.clip(self.channel_gain, 0.3, 3.0)

    @staticmethod
    def _compute_depth(obs_info):
        """Compute insertion depth from peg-to-hole distance."""
        d = np.linalg.norm(obs_info["peg_to_hole"])
        return max(0.0, 0.1 - d)
