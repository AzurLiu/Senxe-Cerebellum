#!/usr/bin/env python3
"""
Senxe Cerebellum v4.0 — RoboSuite NutAssembly (Native Force/Torque Sensors)
=========================================================================
CL1 Bio-Computer vs PPO vs Random — Industrial Assembly Sample Efficiency

Benchmark comparing biological neural control (Cortical Labs CL1) against
PPO reinforcement learning and random baselines on the RoboSuite NutAssembly
task with a Franka Panda robot arm and native force/torque sensors.

Usage:  python senxe_demo_robosuite.py
Output: cl1_nutassembly.mp4, side_by_side_nutassembly.mp4, learning_curve_nutassembly.png

To use real CL1 hardware: pip install cl-sdk  (auto-detected, zero code changes)
"""
import os, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, imageio, cv2
from tqdm import tqdm
from collections import deque
from contextlib import contextmanager

# ═══ Core Modules (shared with v3.0) ═══
from core.neurons import (
    cl_open, warmup_calibration,
    ChannelSet, StimDesign, BurstDesign,
    MockNeurons, CL_AVAILABLE,
)
from core.decoder import AntagonisticDecoder
from core.pdi import PDI
from core.curiosity import NeuralCuriosity
from core.video import save_video, make_side_by_side
from core.vie import VIE
from core.hud import draw_overlay, hud

# ═══ Configuration ═══
SEED            = 42
AGENT_NAME      = "CL1 Bio" if CL_AVAILABLE else "Mock SNN"
ENV_NAME        = "NutAssembly"
ROBOT           = "Panda"
CL1_EPISODES    = 200;  CL1_MAX_STEPS = 200
PPO_TIMESTEPS   = 200_000;  PPO_EVAL_EPS = 200
RANDOM_EPISODES = 200
VIDEO_FPS       = 30
VIDEO_CL1       = "cl1_nutassembly.mp4"
VIDEO_SIDE      = "side_by_side_nutassembly.mp4"
PLOT_FILE       = "learning_curve_nutassembly.png"
RECORD_LAST_N   = 80;  WARMUP_SECONDS = 10;  ACTION_SCALE = 0.25
RENDER_W        = 720;  RENDER_H = 720        # Render resolution
INSERTION_DEPTH_THRESHOLD = 0.02
FORCE_SAFETY_THRESHOLD    = 20.0
TORQUE_SAFETY_THRESHOLD   = 5.0
PREDICTABLE_STIM_TOP_K  = 8;  PREDICTABLE_BURST_N = 15;  PREDICTABLE_BURST_HZ = 300

# ═══ RoboSuite Environment ═══
def make_robosuite_env(render=False):
    """Create RoboSuite NutAssembly environment with native force/torque sensors."""
    import robosuite as suite
    from robosuite.wrappers import GymWrapper
    raw = suite.make(ENV_NAME, robots=ROBOT, has_renderer=False,
                     has_offscreen_renderer=render, use_camera_obs=False,
                     render_camera="frontview", horizon=CL1_MAX_STEPS, reward_shaping=True)
    return GymWrapper(raw), raw

def extract_obs(obs):
    """Extract force, torque, position, and goal vectors from the observation."""
    if isinstance(obs, dict):
        eef = np.array(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float64).flatten()[:3]
        vel = np.array(obs.get("robot0_eef_vel_lin", obs.get("robot0_eef_vel", np.zeros(3))), dtype=np.float64).flatten()[:3]
        frc = np.array(obs.get("robot0_eef_force", obs.get("ft_force", obs.get("robot0_force", np.zeros(3)))), dtype=np.float64).flatten()[:3]
        trq = np.array(obs.get("robot0_eef_torque", obs.get("ft_torque", obs.get("robot0_torque", np.zeros(3)))), dtype=np.float64).flatten()[:3]
        p2h = np.array(obs.get("peg_to_hole", obs.get("hole_pos", np.zeros(3)) - eef), dtype=np.float64).flatten()[:3]
        jnt = np.array(obs.get("robot0_joint_pos", np.zeros(7)), dtype=np.float64).flatten()
        return dict(eef_pos=eef, eef_vel=vel, force=frc, torque=trq, peg_to_hole=p2h, joint_pos=jnt)
    # flat array 路径：记录警告，使用安全默认值
    import warnings
    warnings.warn(
        "extract_obs received flat array — index mapping may be incorrect. "
        "Prefer dict observations via RoboSuite config.",
        stacklevel=2
    )
    o = np.array(obs, dtype=np.float64).flatten(); n = len(o)
    return dict(
        eef_pos=o[:3] if n >= 3 else np.zeros(3),
        eef_vel=o[3:6] if n >= 6 else np.zeros(3),
        force=np.zeros(3),   # 不猜测，返回零值
        torque=np.zeros(3),  # 不猜测，返回零值
        peg_to_hole=np.zeros(3),
        joint_pos=np.zeros(7),
    )

def compute_insertion_depth(info):
    d = np.linalg.norm(info["peg_to_hole"]); return max(0.0, 0.1 - d), d

# ═══ CL1 Biological Agent ═══
class CL1Agent:
    def __init__(self, env, raw_env, neurons, channel_ranking=None, responsiveness=None):
        self.env = env; self.raw_env = raw_env; self.neurons = neurons
        self.action_dim = env.action_space.shape[0]
        self.vie = VIE(neurons, force_threshold=FORCE_SAFETY_THRESHOLD,
                       depth_threshold=INSERTION_DEPTH_THRESHOLD, raw_env=raw_env)
        # P1: Population Vector — pass calibration responsiveness as decode weights
        resp_weights = responsiveness if channel_ranking is not None else None
        self.decoder = AntagonisticDecoder(self.action_dim, action_scale=ACTION_SCALE,
                                            channel_weights=resp_weights)
        self.pdi = PDI()
        self.curiosity = NeuralCuriosity()
        self.episode_rewards = []
        self.best_reward = -np.inf
        self.action_bias = np.zeros(self.action_dim)
        self.lr = 0.05
        self.top_channels = (channel_ranking[:PREDICTABLE_STIM_TOP_K].tolist()
                             if channel_ranking is not None else list(range(PREDICTABLE_STIM_TOP_K)))

    def _detect_spikes(self):
        frames = self.neurons.read(250, None)
        threshold = np.percentile(frames, 99.5)
        spike_channels = list(set(np.where(frames > threshold)[1]))
        firing_rates = np.mean(np.abs(frames.astype(np.float32)), axis=0)
        return spike_channels, firing_rates

    def _predictable_stim_inject(self, reward):
        """Predictable stimulus injection — positive reinforcement pathway.

        When reward > 0, delivers structured burst stimulation to the top-K
        calibrated channels. Under the Free Energy Principle, predictable
        structured stimulation reinforces the current behavioral policy
        (the neural culture learns to reproduce the rewarded activity pattern).

        Args:
            reward: Step reward from the environment.
        """
        if reward <= 0: return
        if hasattr(self.neurons, 'inject_reward'):
            self.neurons.inject_reward(min(reward, 1.0))
        amp = np.clip(reward * 2.0, 0.5, 3.0)
        s = StimDesign(200, -amp, 200, amp)
        self.neurons.stim(ChannelSet(*self.top_channels), s,
                          BurstDesign(PREDICTABLE_BURST_N, PREDICTABLE_BURST_HZ))

    def _unpredictable_stim_inject(self, penalty):
        """Unpredictable Stimulus noise injection — negative reinforcement pathway.

        When a negative event occurs (force violation, increasing distance,
        large negative reward), delivers unpredictable random-frequency
        stimulation to non-top channels. Under the DishBrain/FEP framework,
        unpredictable noise signals an undesirable state, driving the neural
        culture to modify its activity patterns to avoid this condition.

        Triggers: force > safety threshold, distance increasing, reward < -1.0.

        Args:
            penalty: Negative penalty magnitude (should be <= 0).
        """
        if penalty >= 0: return
        if hasattr(self.neurons, 'inject_reward'):
            self.neurons.inject_reward(max(penalty, -1.0))
        amp = np.clip(abs(penalty) * 1.5, 0.3, 2.0)
        # Select 8 random channels (excluding top_channels — noise, not signal)
        available = [ch for ch in range(64) if ch not in self.top_channels]
        random_chs = np.random.choice(available, size=min(8, len(available)), replace=False).tolist()
        # Irregular pulses: random frequency and burst count
        stim = StimDesign(160, -amp, 160, amp)
        burst = BurstDesign(np.random.randint(3, 10), np.random.randint(50, 300))
        self.neurons.stim(ChannelSet(*random_chs), stim, burst)

    def run_episode(self, max_steps=CL1_MAX_STEPS, record=False, ep_num=0):
        obs, _ = self.env.reset()
        obs_info = extract_obs(obs)
        self.pdi.reset(); self.decoder.reset(); self.curiosity.reset()
        self.action_bias = np.zeros(self.action_dim)
        total_reward = 0.0; frames_list = []
        ep_successes = []; ep_force_safe = []
        step_rewards = deque(maxlen=50); cur_fr = np.zeros(64)
        ep_firing_acc = []  # Accumulate per-step firing rates for evolution heatmap
        prev_dist = None  # Track distance change for unpredictable stimulus

        for step in range(max_steps):
            self.vie.encode(obs_info)
            spikes, cur_fr = self._detect_spikes()
            ep_firing_acc.append(cur_fr.copy())
            # P2: Adaptive VIE encoding — online gain adjustment
            self.vie.adapt(cur_fr)
            vel = obs_info["eef_vel"]
            self.pdi.update(vel); pdi_val = self.pdi.compute()
            novelty = self.curiosity.compute_novelty(cur_fr)
            # PDI + 好奇心联合驱动
            fep_boost = pdi_val * 0.3 + novelty * 0.1
            raw = self.decoder.decode(spikes, pdi_boost=fep_boost)
            # Add learned action bias (Hebbian momentum)
            action = np.clip(raw + self.action_bias, -1.0, 1.0)
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs_info = extract_obs(obs)
            total_reward += reward

            force_mag = np.linalg.norm(obs_info["force"])
            torque_mag = np.linalg.norm(obs_info["torque"])
            depth, cur_dist = compute_insertion_depth(obs_info)
            inserted = depth > INSERTION_DEPTH_THRESHOLD
            force_safe = force_mag < FORCE_SAFETY_THRESHOLD
            success = 1 if (inserted and force_safe) else 0
            ep_successes.append(success)
            ep_force_safe.append(1 if force_safe else 0)
            step_rewards.append(reward)

            # === Dual feedback: Predictable Stimulus (positive) + Unpredictable Stimulus (negative) ===
            # Positive: reward > 0 → structured burst on top-K channels
            self._predictable_stim_inject(reward)
            # Negative: force exceeds safety OR distance increasing → random noise
            penalty = 0.0
            if force_mag > FORCE_SAFETY_THRESHOLD:
                penalty -= (force_mag - FORCE_SAFETY_THRESHOLD) * 0.5
            if prev_dist is not None and cur_dist > prev_dist + 0.005:
                penalty -= (cur_dist - prev_dist) * 10.0
            if reward < -1.0:
                penalty += reward * 0.3  # already negative
            self._unpredictable_stim_inject(penalty)
            prev_dist = cur_dist

            # Reward-modulated learning (Hebbian action momentum)
            if reward > -0.5:
                self.action_bias += self.lr * action * (reward + 1.0)
                self.action_bias = np.clip(self.action_bias, -0.5, 0.5)

            if record:
                try:
                    frame = self.raw_env.sim.render(width=RENDER_W, height=RENDER_H, camera_name="frontview")
                    if frame is not None:
                        frame = frame[::-1]
                    else:
                        frame = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
                except Exception:
                    frame = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
                if frame is None or frame.size == 0:
                    continue
                health_full = self.neurons.get_health() if hasattr(self.neurons, 'get_health') else None
                min_h = float(health_full.min()) if health_full is not None else 1.0
                sr = np.mean(ep_successes) * 100
                fsr = np.mean(ep_force_safe) * 100
                frame = draw_overlay(frame, ep_num, total_reward, pdi_val, min_h, cur_fr,
                                     step_rewards, distance=cur_dist, force_mag=force_mag,
                                     torque_mag=torque_mag, depth=depth,
                                     success_rate=sr, force_safe_rate=fsr,
                                     force_vec=obs_info["force"], health_arr=health_full,
                                     force_threshold=FORCE_SAFETY_THRESHOLD)
                frames_list.append(frame)

            if terminated or truncated: break

        if total_reward > self.best_reward:
            self.best_reward = total_reward

        # Record episode average firing rates for evolution heatmap
        if ep_firing_acc:
            ep_avg_fr = np.mean(ep_firing_acc, axis=0)
            hud.episode_firing_history.append(ep_avg_fr)

        success_rate = np.mean(ep_successes) * 100 if ep_successes else 0.0
        force_safe_rate = np.mean(ep_force_safe) * 100 if ep_force_safe else 100.0
        return total_reward, self.pdi.compute(), frames_list, success_rate, force_safe_rate

    def train(self, num_episodes=CL1_EPISODES, record_last_n=RECORD_LAST_N):
        """Train the CL1 biological agent over multiple episodes."""
        print("\n" + "=" * 60)
        print(f"  {AGENT_NAME}-Computer Training (RoboSuite NutAssembly)")
        print("=" * 60)
        print(f"  Episodes: {num_episodes} | Env: {ENV_NAME} ({ROBOT})")
        backend = "Cortical Labs cl-sdk" if CL_AVAILABLE else "Built-in mock"
        print(f"  Backend:  {backend}")
        print(f"  Modules:  VIE(Adaptive) + PopVector + PDI + Predictable/Unpredictable Stim (FEP dual-feedback)")
        print(f"  Record:   last {record_last_n} mature episodes\n")

        all_frames = []; all_sr = []; all_fsr = []
        record_start = max(0, num_episodes - record_last_n)

        pbar = tqdm(range(num_episodes), desc="CL1", ncols=90)
        for ep in pbar:
            rec = (ep >= record_start)
            reward, pdi_val, frames, sr, fsr = self.run_episode(record=rec, ep_num=ep)
            self.episode_rewards.append(reward)
            all_sr.append(sr); all_fsr.append(fsr)
            if rec: all_frames.extend(frames)

            avg = np.mean(self.episode_rewards[-20:])
            pbar.set_postfix(R=f"{reward:.1f}", avg20=f"{avg:.1f}",
                             PDI=f"{pdi_val:.2f}", SR=f"{sr:.0f}%", FSR=f"{fsr:.0f}%")

            if hasattr(self.neurons, 'get_health') and (ep + 1) % 100 == 0:
                health = self.neurons.get_health()
                min_h = health.min()
                tqdm.write(f"  [Health] Ep {ep+1}: min_health={min_h:.3f} "
                           f"{'OK' if min_h > 0.5 else 'WARNING!'}")

            if (ep + 1) % 50 == 0:
                avg_sr = np.mean(all_sr[-50:]); avg_fsr = np.mean(all_fsr[-50:])
                tqdm.write(f"  CL1 Ep {ep+1}: R={reward:.1f} avg20={avg:.1f} "
                           f"SR={avg_sr:.1f}% FSR={avg_fsr:.1f}%")

        final = np.mean(self.episode_rewards[-20:])
        final_sr = np.mean(all_sr[-20:]); final_fsr = np.mean(all_fsr[-20:])
        print(f"\n  CL1 Done | avg20={final:.2f} SR={final_sr:.1f}% FSR={final_fsr:.1f}%")
        self.all_success_rates = all_sr; self.all_force_safe_rates = all_fsr
        return all_frames

# ═══ PPO Baseline ═══
def train_ppo_baseline(record_last_n=RECORD_LAST_N):
    from stable_baselines3 import PPO
    print("\n" + "=" * 60)
    print("  PPO Baseline (RoboSuite NutAssembly)")
    print("=" * 60)
    print(f"  Training: {PPO_TIMESTEPS} steps | Eval: {PPO_EVAL_EPS} episodes\n")

    train_env, _ = make_robosuite_env(render=False)
    eval_env, eval_raw = make_robosuite_env(render=True)

    model = PPO("MlpPolicy", train_env, verbose=0,
                n_steps=256, batch_size=64, learning_rate=3e-4)

    all_rewards = []; all_frames = []; all_sr = []; all_fsr = []
    n_chunks = 20; chunk_steps = PPO_TIMESTEPS // n_chunks
    evals_per_chunk = PPO_EVAL_EPS // n_chunks
    record_start = PPO_EVAL_EPS - record_last_n; ep_count = 0

    pbar = tqdm(range(n_chunks), desc="PPO ", ncols=90)
    for chunk in pbar:
        model.learn(total_timesteps=chunk_steps, reset_num_timesteps=False)
        for _ in range(evals_per_chunk):
            obs, _ = eval_env.reset()
            ep_r = 0.0; done = False; ep_frames = []
            ep_forces = []; ep_depths = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, _ = eval_env.step(action)
                ep_r += r; done = term or trunc
                oi = extract_obs(obs)
                ep_forces.append(np.linalg.norm(oi["force"]))
                d, _ = compute_insertion_depth(oi)
                ep_depths.append(d)
                if ep_count >= record_start:
                    try:
                        frame = eval_raw.sim.render(width=RENDER_W, height=RENDER_H, camera_name="frontview")
                        if frame is not None:
                            frame = frame[::-1]
                        else:
                            frame = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
                    except Exception:
                        frame = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
                    if frame is not None and frame.size > 0:
                        ep_frames.append(frame)
            all_rewards.append(ep_r)
            max_d = max(ep_depths) if ep_depths else 0
            max_f = max(ep_forces) if ep_forces else 0
            sr = 100.0 if (max_d > INSERTION_DEPTH_THRESHOLD and max_f < FORCE_SAFETY_THRESHOLD) else 0.0
            fsr = 100.0 if max_f < FORCE_SAFETY_THRESHOLD else 0.0
            all_sr.append(sr); all_fsr.append(fsr)
            if ep_count >= record_start: all_frames.extend(ep_frames)
            ep_count += 1
        avg = np.mean(all_rewards[-evals_per_chunk:])
        pbar.set_postfix(avg_R=f"{avg:.1f}", eps=len(all_rewards))

    train_env.close(); eval_env.close()
    final = np.mean(all_rewards[-20:])
    print(f"\n  PPO Done | avg20={final:.2f}")
    return all_rewards, all_frames, all_sr, all_fsr

# ═══ Random Baseline ═══
def run_random_baseline(num_episodes=RANDOM_EPISODES):
    print("\n" + "=" * 60)
    print("  Random Agent Baseline (RoboSuite NutAssembly)")
    print("=" * 60)
    print(f"  Episodes: {num_episodes}\n")

    env, _ = make_robosuite_env(render=False)
    all_rewards = []; all_sr = []; all_fsr = []

    pbar = tqdm(range(num_episodes), desc="RNG ", ncols=90)
    for ep in pbar:
        obs, _ = env.reset()
        total_reward = 0.0; done = False
        ep_forces = []; ep_depths = []
        while not done:
            action = env.action_space.sample()
            obs, r, term, trunc, _ = env.step(action)
            total_reward += r; done = term or trunc
            oi = extract_obs(obs)
            ep_forces.append(np.linalg.norm(oi["force"]))
            d, _ = compute_insertion_depth(oi)
            ep_depths.append(d)
        all_rewards.append(total_reward)
        max_d = max(ep_depths) if ep_depths else 0
        max_f = max(ep_forces) if ep_forces else 0
        sr = 100.0 if (max_d > INSERTION_DEPTH_THRESHOLD and max_f < FORCE_SAFETY_THRESHOLD) else 0.0
        fsr = 100.0 if max_f < FORCE_SAFETY_THRESHOLD else 0.0
        all_sr.append(sr); all_fsr.append(fsr)
        avg = np.mean(all_rewards[-20:])
        pbar.set_postfix(R=f"{total_reward:.1f}", avg20=f"{avg:.1f}")

    env.close()
    print(f"\n  Random Done | avg20={np.mean(all_rewards[-20:]):.2f}")
    return all_rewards, all_sr, all_fsr

# ═══ Video Utilities — now from core/video.py ═══
# save_video() and make_side_by_side() imported at top from core.video

# ═══ Learning Curve ═══
def plot_learning_curves(cl1_r, ppo_r, rnd_r, path=PLOT_FILE,
                         cl1_sr=None, cl1_fsr=None, ppo_sr=None, rnd_sr=None):
    for font in ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']:
        try: plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']; break
        except Exception: continue
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 1, figsize=(13, 10), gridspec_kw={'height_ratios': [3, 2]})
    ax = axes[0]

    def rolling(data, w=20):
        if len(data) < w: return data
        return np.convolve(data, np.ones(w)/w, mode='valid')

    cl1_s = rolling(cl1_r); ppo_s = rolling(ppo_r); rnd_s = rolling(rnd_r)
    off = 19

    ax.plot(cl1_r, alpha=0.12, color='#e74c3c')
    ax.plot(ppo_r, alpha=0.12, color='#3498db')
    ax.plot(rnd_r, alpha=0.12, color='#95a5a6')
    ax.plot(range(off, off+len(cl1_s)), cl1_s, color='#e74c3c', lw=2.5,
            label=f'{AGENT_NAME} (VIE+Force/Torque+Predictable Stimulus)')
    ax.plot(range(off, off+len(ppo_s)), ppo_s, color='#3498db', lw=2.5,
            label='PPO Traditional RL')
    ax.plot(range(off, off+len(rnd_s)), rnd_s, color='#95a5a6', lw=2.0, ls='--',
            label='Random (baseline)')

    ax.set_title("Senxe Cerebellum v4.0 — RoboSuite NutAssembly (Native Force/Torque)\n"
                 f"{AGENT_NAME} vs PPO vs Random: Industrial Assembly Sample Efficiency",
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Episode"); ax.set_ylabel("Reward")
    ax.legend(fontsize=10, loc='lower right'); ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(len(cl1_r), len(ppo_r), len(rnd_r)))

    ax.text(0.02, 0.95,
            f"Senxe Cerebellum v4.0 — Native Force/Torque Sensors\n"
            f"CL1: {CL1_EPISODES} eps | PPO: {PPO_TIMESTEPS} steps\n"
            f"Random: {RANDOM_EPISODES} eps\n"
            f"NutAssembly ({ROBOT}) — ready for real robotic arm\n"
            f"Force safety: <{FORCE_SAFETY_THRESHOLD}N | Depth: >{INSERTION_DEPTH_THRESHOLD}m",
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round,pad=0.4', fc='lightyellow', alpha=0.7))

    # Success Rate + Force Safety Rate subplot
    ax2 = axes[1]
    if cl1_sr and len(cl1_sr) > 0:
        sr_s = rolling(cl1_sr); ax2.plot(range(off, off+len(sr_s)), sr_s,
            color='#2ecc71', lw=2.0, label='CL1 Success Rate (%)')
    if cl1_fsr and len(cl1_fsr) > 0:
        fsr_s = rolling(cl1_fsr); ax2.plot(range(off, off+len(fsr_s)), fsr_s,
            color='#e67e22', lw=2.0, ls='-.', label='CL1 Force Safety Rate (%)')
    if ppo_sr and len(ppo_sr) > 0:
        psr_s = rolling(ppo_sr); ax2.plot(range(off, off+len(psr_s)), psr_s,
            color='#3498db', lw=1.5, ls=':', label='PPO Success Rate (%)')
    if rnd_sr and len(rnd_sr) > 0:
        rsr_s = rolling(rnd_sr); ax2.plot(range(off, off+len(rsr_s)), rsr_s,
            color='#95a5a6', lw=1.5, ls=':', label='Random Success Rate (%)')

    ax2.set_xlabel("Episode"); ax2.set_ylabel("Rate (%)")
    ax2.set_ylim(0, 105); ax2.legend(fontsize=9, loc='upper left'); ax2.grid(True, alpha=0.3)
    ax2.set_title("Success Rate + Force Safety Rate", fontsize=11)

    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f"  Learning curve saved: {path}")

# ═══ Main Entry Point ═══
def main():
    np.random.seed(SEED)
    hud.reset()
    print("+" + "=" * 58 + "+")
    print("|  Senxe Cerebellum v4.0 — RoboSuite NutAssembly            |")
    print("|  Native Force/Torque — Ready for Real Robotic Arm       |")
    print("+" + "=" * 58 + "+\n")

    if CL_AVAILABLE:
        print("  CL SDK: Cortical Labs cl-sdk (real/simulator)")
    else:
        print("  CL SDK unavailable -- using built-in mock")
        print("     (pip install cl-sdk to use real CL1 backend)")
    print(f"  Env: {ENV_NAME} | Robot: {ROBOT}")
    print(f"  Force safety: <{FORCE_SAFETY_THRESHOLD}N | Depth: >{INSERTION_DEPTH_THRESHOLD}m\n")

    # Phase 0: Calibration
    print("-" * 60); print("  Phase 0: Channel Warm-up Calibration"); print("-" * 60)
    with cl_open() as neurons:
        ranking, resp = warmup_calibration(neurons, WARMUP_SECONDS)

        # Phase 1: CL1 Training
        env, raw_env = make_robosuite_env(render=True)
        agent = CL1Agent(env, raw_env, neurons, channel_ranking=ranking, responsiveness=resp)
        cl1_frames = agent.train(num_episodes=CL1_EPISODES, record_last_n=RECORD_LAST_N)
        cl1_rewards = agent.episode_rewards.copy()
        cl1_sr = agent.all_success_rates.copy()
        cl1_fsr = agent.all_force_safe_rates.copy()
        env.close()

    # Phase 2: PPO
    ppo_rewards, ppo_frames, ppo_sr, ppo_fsr = train_ppo_baseline(record_last_n=RECORD_LAST_N)

    # Phase 3: Random
    rnd_rewards, rnd_sr, rnd_fsr = run_random_baseline(num_episodes=RANDOM_EPISODES)

    # Phase 4: Videos
    print("\n" + "-" * 60); print("  Phase 4: Generating Videos"); print("-" * 60)
    save_video(cl1_frames, VIDEO_CL1, fps=VIDEO_FPS, target_seconds=20)
    make_side_by_side(cl1_frames, ppo_frames, VIDEO_SIDE, fps=VIDEO_FPS,
                      left_label=f"{AGENT_NAME} (Force/Torque)",
                      right_label="PPO (Traditional RL)",
                      center_label="NutAssembly")

    # Phase 5: Plot
    print()
    plot_learning_curves(cl1_rewards, ppo_rewards, rnd_rewards,
                         cl1_sr=cl1_sr, cl1_fsr=cl1_fsr, ppo_sr=ppo_sr, rnd_sr=rnd_sr)

    # Done
    print("\n" + "=" * 60)
    print("  Senxe Cerebellum v4.0 Demo Complete!")
    print("=" * 60)
    print(f"  Video (CL1):          {VIDEO_CL1}")
    print(f"  Video (side-by-side): {VIDEO_SIDE}")
    print(f"  Plot:                 {PLOT_FILE}")
    print(f"  CL1 avg20:  {np.mean(cl1_rewards[-20:]):.2f}")
    print(f"  PPO avg20:  {np.mean(ppo_rewards[-20:]):.2f}")
    print(f"  RNG avg20:  {np.mean(rnd_rewards[-20:]):.2f}")
    print(f"  CL1 SR:     {np.mean(cl1_sr[-20:]):.1f}%")
    print(f"  CL1 FSR:    {np.mean(cl1_fsr[-20:]):.1f}%")
    print()
    print("  Switch to real CL1 hardware:")
    print("    pip install cl-sdk   # Auto-detected, zero code changes!")
    print()

if __name__ == "__main__":
    main()
