#!/usr/bin/env python3
"""
Senxe Cerebellum v3.0 — FetchPickAndPlace Benchmark (Legacy)
==========================================================
CL1 Bio-Computer — Pure Wetware Interface

Demonstrates key biological control components:
  - VIE  (Virtual Interference Encoding)   — Doom-style visual + tactile encoding
  - Antagonistic Decoding                  — Flexor/extensor motor output
  - PDI  (Physical Disturbance Index)      — FEP-inspired explore/exploit gate
  - Predictable stimulus injection         — Structured burst reinforcement
  - Channel Warm-up Calibration            — 10-second responsiveness probing
  - Metabolic Guardrail                    — Per-channel health monitoring

Usage:  python senxe_demo.py
Output: cl1_pickandplace.mp4
"""

import os
import sys
import numpy as np
import gymnasium as gym
import gymnasium_robotics  # Register Fetch environments
import imageio
import cv2
from tqdm import tqdm
from collections import deque

from core.neurons import cl_open, warmup_calibration, ChannelSet, StimDesign, BurstDesign
from core.decoder import AntagonisticDecoder
from core.pdi import PDI
from core.video import save_video

# ═══ Configuration ═══
ENV_NAME          = "FetchPickAndPlace-v4"
REWARD_TYPE       = "dense"
EPISODES          = 300
MAX_STEPS         = 100
VIDEO_FPS         = 30
VIDEO_CL1         = "cl1_pickandplace.mp4"
RECORD_LAST_N     = 100
WARMUP_SECONDS    = 10
ACTION_SCALE      = 0.35

PREDICTABLE_STIM_TOP_K    = 8
PREDICTABLE_BURST_N       = 15
PREDICTABLE_BURST_HZ      = 300

# ═══ VIE: Virtual Interference Encoding ═══
class VIE:
    CH_PRESSURE = list(range(0,  16))
    CH_VELOCITY = list(range(16, 32))
    CH_POSITION = list(range(32, 48))
    CH_GOALDIR  = list(range(48, 64))

    def __init__(self, neurons, env=None):
        self.neurons = neurons
        self.env = env

    def encode(self, obs, goal):
        if isinstance(obs, dict):
            grip_pos = obs.get("observation", np.zeros(25))[:3]
            grip_vel = obs.get("observation", np.zeros(25))[3:6] if len(obs.get("observation", np.zeros(25))) >= 6 else np.zeros(3)
            object_pos = obs.get("object", np.zeros(3))[:3]
        else:
            grip_pos = obs[:3]
            grip_vel = obs[3:6] if len(obs) >= 6 else np.zeros(3)
            object_pos = obs[3:6] if len(obs) >= 15 else grip_pos.copy()

        delta     = goal - grip_pos
        distance  = np.linalg.norm(delta)
        direction = delta / (distance + 1e-8)

        stim = StimDesign(160, -1.0, 160, 1.0)
        burst_hz = int(np.clip(50 + 400 * distance, 50, 400))
        burst_n  = max(1, min(10, int(distance * 15)))
        self.neurons.stim(ChannelSet(*self.CH_PRESSURE[:8]), stim, BurstDesign(burst_n, burst_hz))

        visual_brightness = 0.5
        if self.env is not None:
            try:
                vis_frame = self.env.render()
                if vis_frame is not None and vis_frame.size > 0:
                    visual_brightness = np.mean(vis_frame.astype(np.float32)) / 255.0
            except Exception:
                pass
        vis_hz = int(np.clip(50 + 300 * visual_brightness, 50, 350))
        vis_n  = max(1, int(visual_brightness * 5))
        vis_stim = StimDesign(160, -0.8, 160, 0.8)
        self.neurons.stim(ChannelSet(*self.CH_PRESSURE[8:16]), vis_stim, BurstDesign(vis_n, vis_hz))

        obj_delta = object_pos - grip_pos
        obj_dist  = np.linalg.norm(obj_delta)
        obj_dir   = obj_delta / (obj_dist + 1e-8)
        for axis in range(3):
            ch_base = self.CH_GOALDIR[8] + axis * 2
            chs = ChannelSet(*[ch_base, min(ch_base + 1, 63)])
            d = obj_dir[axis]
            intensity = np.clip(abs(d) * 1.5 + 0.1, 0.1, 2.0)
            obj_stim = StimDesign(160, -intensity * np.sign(d), 160,  intensity * np.sign(d))
            obj_burst = BurstDesign(max(1, int(abs(d) * 5)), int(50 + abs(d) * 100))
            self.neurons.stim(chs, obj_stim, obj_burst)

        vel_mag = np.linalg.norm(grip_vel)
        if vel_mag > 0.003:
            for axis in range(3):
                v = grip_vel[axis]
                if abs(v) > 0.003:
                    ch_base = self.CH_VELOCITY[0] + axis * 5
                    chs = ChannelSet(*range(ch_base, min(ch_base + 5, 32)))
                    intensity = np.clip(abs(v) * 5, 0.1, 2.0)
                    wave_stim = StimDesign(160, -intensity, 160, intensity)
                    wave_hz = int(np.clip(60 * abs(v), 20, 200))
                    self.neurons.stim(chs, wave_stim, BurstDesign(2, wave_hz))

        for axis in range(3):
            ch_base = self.CH_GOALDIR[0] + axis * 2
            chs = ChannelSet(*[ch_base, min(ch_base + 1, 55)])
            d = direction[axis]
            intensity = np.clip(abs(d) * 1.5 + 0.1, 0.1, 2.0)
            dir_stim = StimDesign(160, -intensity * np.sign(d), 160,  intensity * np.sign(d))
            dir_burst = BurstDesign(max(1, int(abs(d) * 5)), int(50 + abs(d) * 100))
            self.neurons.stim(chs, dir_stim, dir_burst)


# ═══ Real-time HUD Overlay (OpenCV) ═══
def _overlay_text(frame, ep, reward, pdi, min_health, distance=0.0, success_rate=0.0, gripper_status=""):
    lines = [
        f"Ep:{ep} R:{reward:.1f}",
        f"PDI:{pdi:.2f} H:{min_health:.2f}",
        f"Dist:{distance:.3f}",
    ]
    if gripper_status:
        lines.append(f"SR:{success_rate:.0f}% Grip:{gripper_status}")
    font      = cv2.FONT_HERSHEY_SIMPLEX
    scale     = 0.38
    thickness = 1
    line_h    = 16
    x0, y0    = 6, 6
    box_w     = 160
    box_h     = len(lines) * line_h + 8

    roi = frame[y0:y0+box_h, x0:x0+box_w].astype(np.float32)
    roi *= 0.25
    frame[y0:y0+box_h, x0:x0+box_w] = roi.astype(np.uint8)

    for i, line in enumerate(lines):
        y = y0 + 13 + i * line_h
        cv2.putText(frame, line, (x0 + 4, y), font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        cv2.putText(frame, line, (x0 + 4, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

def _overlay_bar_chart(frame, firing_rates, top_k=8):
    h, w = frame.shape[:2]
    top_idx = np.argsort(firing_rates)[-top_k:][::-1]
    top_vals = firing_rates[top_idx]
    chart_w    = 110
    chart_h    = 55
    margin_r   = 6
    margin_t   = 6
    x0         = w - chart_w - margin_r
    y0         = margin_t
    bar_w      = chart_w // top_k

    overlay_region = frame[y0:y0+chart_h, x0:x0+chart_w].astype(np.float32)
    overlay_region *= 0.45
    frame[y0:y0+chart_h, x0:x0+chart_w] = overlay_region.astype(np.uint8)

    val_min   = top_vals.min()
    val_max   = top_vals.max()
    val_range = max(val_max - val_min, 1.0)
    usable_h  = chart_h - 16

    for i, val in enumerate(top_vals):
        relative = (val - val_min) / val_range
        absolute = np.clip((val - 50.0) / 350.0, 0.0, 1.0)
        normed   = 0.6 * relative + 0.4 * absolute
        normed   = max(0.08, normed)
        bar_h    = max(2, int(normed * usable_h))
        bx  = x0 + i * bar_w + 2
        by  = y0 + chart_h - bar_h - 12
        bx2 = bx + bar_w - 4

        color = (
            int(255 * min(normed * 2, 1.0)),
            int(255 * min((1.0 - normed) * 2, 1.0)),
            30,
        )
        cv2.rectangle(frame, (bx, by), (bx2, y0 + chart_h - 12), color, -1)
        label = str(top_idx[i])
        cv2.putText(frame, label, (bx, y0 + chart_h - 1), cv2.FONT_HERSHEY_PLAIN, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

def _overlay_reward_curve(frame, reward_history):
    if len(reward_history) < 2: return
    h, w = frame.shape[:2]
    chart_h   = 50
    margin_b  = 6
    margin_lr = 30
    y_top     = h - chart_h - margin_b
    y_bot     = h - margin_b
    x_left    = margin_lr
    x_right   = w - margin_lr

    roi = frame[y_top:y_bot, x_left:x_right].astype(np.float32)
    roi *= 0.35
    frame[y_top:y_bot, x_left:x_right] = roi.astype(np.uint8)

    data = np.array(reward_history, dtype=np.float32)
    r_min, r_max = data.min(), data.max()
    if r_max - r_min < 1e-6: r_max = r_min + 1.0
    chart_pixel_w = x_right - x_left
    chart_pixel_h = y_bot - y_top - 4

    n = len(data)
    xs = np.linspace(0, chart_pixel_w - 1, n).astype(int) + x_left
    ys = y_bot - 2 - ((data - r_min) / (r_max - r_min) * chart_pixel_h).astype(int)

    pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], isClosed=False, color=(255, 80, 80), thickness=2, lineType=cv2.LINE_AA)

    if r_min < 0 < r_max:
        zero_y = int(y_bot - 2 - ((0 - r_min) / (r_max - r_min) * chart_pixel_h))
        for x in range(x_left, x_right, 8):
            cv2.line(frame, (x, zero_y), (min(x + 4, x_right), zero_y), (120, 120, 120), 1)

def draw_overlay(frame, ep, reward, pdi, min_health, firing_rates, reward_history, distance=0.0, success_rate=0.0, gripper_status=""):
    frame = np.ascontiguousarray(frame)
    _overlay_text(frame, ep, reward, pdi, min_health, distance=distance, success_rate=success_rate, gripper_status=gripper_status)
    _overlay_bar_chart(frame, firing_rates)
    _overlay_reward_curve(frame, reward_history)
    return frame

# ═══ CL1 Biological Agent ═══
class CL1Agent:
    def __init__(self, env, neurons, channel_ranking=None):
        self.env     = env
        self.neurons = neurons
        self.action_dim = env.action_space.shape[0]
        self.vie     = VIE(neurons, env=env)
        self.decoder = AntagonisticDecoder(self.action_dim, action_scale=ACTION_SCALE)
        self.pdi     = PDI()
        self.episode_rewards = []
        self.lr          = 0.015
        self.action_bias = np.zeros(self.action_dim)
        self.best_reward = -np.inf
        self.best_bias   = np.zeros(self.action_dim)
        if channel_ranking is not None:
            self.top_channels = channel_ranking[:PREDICTABLE_STIM_TOP_K].tolist()
        else:
            self.top_channels = list(range(PREDICTABLE_STIM_TOP_K))

    def _detect_spikes(self):
        frames = self.neurons.read(250, None)
        abs_frames = np.abs(frames.astype(np.float32))
        threshold = max(50.0, np.percentile(abs_frames, 99.5))
        spike_channels = list(set(np.where(abs_frames > threshold)[1]))
        firing_rates = np.mean(abs_frames, axis=0)
        return spike_channels, firing_rates

    def _predictable_stim_inject(self, reward):
        if reward <= 0: return
        amp = np.clip(reward * 2.0, 0.5, 3.0)
        stim = StimDesign(200, -amp, 200, amp)
        burst = BurstDesign(PREDICTABLE_BURST_N, PREDICTABLE_BURST_HZ)
        self.neurons.stim(ChannelSet(*self.top_channels), stim, burst)

    def run_episode(self, max_steps=MAX_STEPS, record=False, ep_num=0):
        obs_dict, _ = self.env.reset()
        obs  = obs_dict["observation"]
        goal = obs_dict["desired_goal"]

        self.pdi.reset()
        self.decoder.reset()
        self.vie.reset()
        total_reward = 0.0
        frames = []
        episode_successes = []
        step_rewards = deque(maxlen=50)
        cur_firing_rates = np.zeros(64)

        for step in range(max_steps):
            self.vie.encode(obs, goal)
            spikes, cur_firing_rates = self._detect_spikes()
            vel = obs[3:6] if len(obs) >= 6 else np.zeros(3)
            self.pdi.update(vel)
            pdi_val = self.pdi.compute()
            fep_boost = pdi_val * 0.4
            raw = self.decoder.decode(spikes, pdi_boost=fep_boost)
            action = np.clip(raw + self.action_bias, -1.0, 1.0)
            obs_dict, reward, terminated, truncated, info = self.env.step(action)
            obs  = obs_dict["observation"]
            goal = obs_dict["desired_goal"]
            total_reward += reward

            cur_distance = float(np.linalg.norm(goal - obs[:3]))
            object_pos = obs[3:6] if len(obs) >= 6 else obs[:3]
            obj_goal_dist = float(np.linalg.norm(goal - object_pos))
            gripper_open = obs[9:11] if len(obs) >= 11 else np.array([0.0, 0.0])
            gripper_is_closed = float(np.mean(gripper_open)) < 0.035
            success = 1 if (obj_goal_dist < 0.05 and gripper_is_closed) else 0
            episode_successes.append(success)
            step_rewards.append(reward)

            self._predictable_stim_inject(reward)

            if record:
                frame = self.env.render()
                if frame is not None:
                    min_health = 1.0
                    if hasattr(self.neurons, 'get_health'):
                        min_health = float(self.neurons.get_health().min())
                    cur_success_rate = np.mean(episode_successes) * 100 if episode_successes else 0.0
                    grip_status = "CLOSED" if gripper_is_closed else "OPEN"
                    frame = draw_overlay(frame, ep_num, total_reward, pdi_val, min_health, cur_firing_rates, step_rewards, distance=cur_distance, success_rate=cur_success_rate, gripper_status=grip_status)
                    frames.append(frame)

            if reward > -0.5:
                self.action_bias += self.lr * action * (reward + 1.0)
                self.action_bias = np.clip(self.action_bias, -0.5, 0.5)

            if terminated or truncated:
                break

        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_bias   = self.action_bias.copy()

        success_rate = np.mean(episode_successes) * 100 if episode_successes else 0.0
        return total_reward, self.pdi.compute(), frames, success_rate

    def train(self, num_episodes=EPISODES, record_last_n=RECORD_LAST_N):
        print("\n" + "=" * 60)
        print("  CL1 Bio-Computer Training")
        print("=" * 60)
        print(f"  Episodes: {num_episodes} | Env: {ENV_NAME}")
        
        all_frames = []
        all_success_rates = []
        record_start = max(0, num_episodes - record_last_n)

        pbar = tqdm(range(num_episodes), desc="CL1", ncols=90)
        for ep in pbar:
            rec = (ep >= record_start)
            reward, pdi_val, frames, success_rate = self.run_episode(record=rec, ep_num=ep)
            self.episode_rewards.append(reward)
            all_success_rates.append(success_rate)
            if rec: all_frames.extend(frames)

            avg = np.mean(self.episode_rewards[-20:])
            pbar.set_postfix(R=f"{reward:.1f}", avg20=f"{avg:.1f}", PDI=f"{pdi_val:.2f}", SR=f"{success_rate:.0f}%")

        final = np.mean(self.episode_rewards[-20:])
        final_sr = np.mean(all_success_rates[-20:])
        print(f"\n  CL1 Training Done | Final avg20: {final:.2f} | Final SuccessRate: {final_sr:.1f}%")
        self.all_success_rates = all_success_rates
        return all_frames

# ═══ Main Entry Point ═══
def main():
    print("+" + "=" * 58 + "+")
    print("|  Senxe Cerebellum v3.0 -- Pure Wetware Interface           |")
    print("+" + "=" * 58 + "+\n")

    print("-" * 60)
    print("  Phase 0: Channel Warm-up Calibration")
    print("-" * 60)

    with cl_open() as neurons:
        channel_ranking, responsiveness = warmup_calibration(neurons, WARMUP_SECONDS)

        print("-" * 60)
        print("  Phase 1: CL1 Training")
        print("-" * 60)
        env = gym.make(ENV_NAME, render_mode="rgb_array", reward_type=REWARD_TYPE)
        agent = CL1Agent(env, neurons, channel_ranking)
        cl1_frames = agent.train(num_episodes=EPISODES, record_last_n=RECORD_LAST_N)
        env.close()

    print("\n" + "-" * 60)
    print("  Phase 2: Generating Video")
    print("-" * 60)
    save_video(cl1_frames, VIDEO_CL1, fps=VIDEO_FPS, target_seconds=20)

    print("\n" + "=" * 60)
    print("  Senxe Demo Complete!")
    print("=" * 60)
    print(f"  Video: {VIDEO_CL1}")

if __name__ == "__main__":
    main()
