"""
Senxe Cerebellum — Virtual Interference Encoding (VIE)
====================================================
Maps physical sensor data to neural stimulation patterns using 
Neuromorphic Event-Driven Sparse Coding and Attention Multiplexing.

Channel Layout (Scattered via permutation):
    CH  0-11: Force Deltas (Sparse event-driven)
    CH 12-23: Torque Deltas (Sparse event-driven)
    CH 24-27: Velocity Supplement
    CH 28-54: Spatial Attention Array (27 ch, multiplexed for Nut/Peg)
    CH 55-59: State Flag / Depth Encoding
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
    """Virtual Interference Encoding — Neuromorphic Implementation.

    Features:
    1. Sparse Event-Driven Coding: Channels only fire on physical deltas (changes)
       to reduce MEA crosstalk and seizure-like global overstimulation.
    2. Attention Multiplexing: A dedicated high-resolution 27-channel spatial 
       array focuses entirely on the CURRENT goal (Nut when searching, Peg when transporting).

    Args:
        neurons: CL1 neurons instance with .stim() method.
        force_threshold: Force safety threshold (N).
        depth_threshold: Insertion depth threshold (m).
        raw_env: Optional raw RoboSuite environment reference.
    """

    # ── Non-overlapping channel layout (Scattered for Minicolumns) ──
    _prng = np.random.RandomState(42)
    _all_channels = _prng.permutation(64).tolist()
    
    CH_FORCE         = _all_channels[0:12]      # Sparse force deltas
    CH_TORQUE        = _all_channels[12:24]     # Sparse torque deltas
    CH_VELOCITY      = _all_channels[24:28]     # Velocity 
    CH_SPATIAL       = _all_channels[28:55]     # 27ch Spatial Attention Array
    CH_STATE         = _all_channels[55:60]     # Attention state / Depth
    CH_RESERVED      = _all_channels[60:64]     # Reserved

    def __init__(self, neurons, force_threshold=20.0, depth_threshold=0.02,
                 raw_env=None):
        self.neurons = neurons
        self.raw_env = raw_env
        self.force_threshold = force_threshold
        self.depth_threshold = depth_threshold
        self.channel_gain = np.ones(64)      
        self.adaptation_rate = 0.005

        # Neuromorphic Delta Tracking
        self.prev_force = np.zeros(3)
        self.prev_torque = np.zeros(3)
        self.prev_eef_pos = np.zeros(3)
        
        # Attention State Machine
        # 0: SEARCHING (Target = Nut)
        # 1: TRANSPORTING (Target = Peg)
        self.attention_state = 0  

    def reset(self):
        """Reset temporal state trackers for a new episode. 
        Note: channel_gain is preserved across episodes for long-term homeostasis."""
        self.prev_force = np.zeros(3)
        self.prev_torque = np.zeros(3)
        self.prev_eef_pos = np.zeros(3)
        self.attention_state = 0

    def encode(self, obs_info):
        """Encode observation using sparse delta coding and attention multiplexing.

        Args:
            obs_info: Dictionary with keys 'eef_pos', 'eef_vel', 'force',
                      'torque', 'peg_to_hole', 'eef_to_nut' from extract_obs().
        """
        eef_pos = obs_info["eef_pos"]
        eef_vel = obs_info["eef_vel"]
        force = obs_info["force"]
        torque = obs_info["torque"]
        peg_to_hole = obs_info["peg_to_hole"]
        eef_to_nut = obs_info.get("eef_to_nut", np.zeros(3))
        
        force_mag = np.linalg.norm(force)
        nut_dist = np.linalg.norm(eef_to_nut)

        # ── 1. Attention State Machine ──
        # Heuristic for grasping: close to nut AND feeling force
        if self.attention_state == 0:
            if nut_dist < 0.05 and force_mag > 5.0:
                self.attention_state = 1  # Switch to TRANSPORTING
        else:
            # If we drop it, switch back to SEARCHING
            if force_mag < 1.0:
                self.attention_state = 0  

        active_target_vec = eef_to_nut if self.attention_state == 0 else peg_to_hole

        # ── 2. Sparse Event-Driven Delta Coding ──
        d_force = force - self.prev_force
        d_torque = torque - self.prev_torque
        
        self.prev_force = force.copy()
        self.prev_torque = torque.copy()
        self.prev_eef_pos = eef_pos.copy()

        # Encode Force Deltas (CH 0-11: 4 channels per axis for pos/neg and mag changes)
        for ax in range(3):
            df = d_force[ax]
            if abs(df) > 0.5:  # Noise threshold
                # Use all 4 channels per axis: 0=small+, 1=large+, 2=small-, 3=large-
                mag_idx = 0 if abs(df) < 3.0 else 1
                ch_idx = self.CH_FORCE[ax * 4 + (0 if df > 0 else 2) + mag_idx]
                gain = self.channel_gain[ch_idx]
                inten = np.clip(abs(df) * 0.5 * gain, 0.1, 2.0)
                hz = int(np.clip(50 + abs(df) * 20 * gain, 50, 300))
                fs = StimDesign(160, -inten, 160, inten)
                self.neurons.stim(ChannelSet(ch_idx), fs, BurstDesign(2, hz))

        # Encode Torque Deltas (CH 12-23: 4 channels per axis)
        for ax in range(3):
            dt = d_torque[ax]
            if abs(dt) > 0.1:
                mag_idx = 0 if abs(dt) < 1.0 else 1
                ch_idx = self.CH_TORQUE[ax * 4 + (0 if dt > 0 else 2) + mag_idx]
                gain = self.channel_gain[ch_idx]
                inten = np.clip(abs(dt) * 2.0 * gain, 0.1, 2.0)
                hz = int(np.clip(50 + abs(dt) * 50 * gain, 50, 300))
                ts = StimDesign(160, -inten, 160, inten)
                self.neurons.stim(ChannelSet(ch_idx), ts, BurstDesign(2, hz))

        # Encode Velocity (Continuous but sparse via threshold) (CH 24-27)
        vmag = np.linalg.norm(eef_vel)
        if vmag > 0.01:
            for ax in range(3):
                v = eef_vel[ax]
                if abs(v) > 0.01:
                    cb = self.CH_VELOCITY[ax]
                    gain = self.channel_gain[cb]
                    vi = np.clip(abs(v) * 3 * gain, 0.1, 2.0)
                    vs = StimDesign(160, -vi, 160, vi)
                    vhz = int(np.clip(60 * abs(v) * gain, 20, 200))
                    self.neurons.stim(ChannelSet(cb), vs, BurstDesign(1, vhz))

        # ── 3. High-Resolution Spatial Attention Array (CH 28-54) ──
        # 27 channels = 9 channels per axis. We distribute the target vector component across them.
        for ax in range(3):
            val = active_target_vec[ax]
            # Which of the 9 channels to fire? based on val tuning curve.
            # Map val from [-0.5, 0.5] to bin index [0, 8]
            bin_idx = int(np.clip((val + 0.5) * 9, 0, 8))
            ch_idx = self.CH_SPATIAL[ax * 9 + bin_idx]
            
            # Fire strongly if we are exactly in this bin
            gain = self.channel_gain[ch_idx]
            hz = int(np.clip((100 + abs(val) * 200) * gain, 50, 350))
            inten = np.clip((0.5 + abs(val)) * gain, 0.1, 1.5)
            ps = StimDesign(160, -inten, 160, inten)
            self.neurons.stim(ChannelSet(ch_idx), ps, BurstDesign(2, hz))

        # ── 4. State Flag & Depth (CH 55-59) ──
        # Let the network know WHICH attention state it's currently in
        state_ch = self.CH_STATE[self.attention_state]
        self.neurons.stim(ChannelSet(state_ch), StimDesign(160, -1.0, 160, 1.0), BurstDesign(2, 100))

        if self.attention_state == 1:
            # If transporting, encode depth explicitly to guide insertion
            depth = self._compute_depth(obs_info)
            if depth > 0.01:
                dn = np.clip(depth / self.depth_threshold, 0.0, 2.0)
                ch_idx = self.CH_STATE[2]
                gain = self.channel_gain[ch_idx]
                dhz = int(np.clip((50 + 300 * dn) * gain, 50, 400))
                dstim = StimDesign(160, -0.8 * gain, 160, 0.8 * gain)
                self.neurons.stim(ChannelSet(ch_idx), dstim, BurstDesign(2, dhz))

    def adapt(self, firing_rates):
        """Online adaptation of channel encoding gains (Homeostasis)."""
        fr_norm = firing_rates / (firing_rates.max() + 1e-6)
        error = 0.5 - fr_norm
        self.channel_gain += self.adaptation_rate * error
        self.channel_gain = np.clip(self.channel_gain, 0.3, 3.0)

    @staticmethod
    def _compute_depth(obs_info):
        """Compute insertion depth from peg-to-hole distance."""
        d = np.linalg.norm(obs_info["peg_to_hole"])
        return max(0.0, 0.1 - d)

