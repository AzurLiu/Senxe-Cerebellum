#!/usr/bin/env python3
"""
Senxe Cerebellum v4.0 — RoboSuite NutAssembly (Native Force/Torque Sensors)
=========================================================================
CL1 Bio-Computer — Pure Wetware Interface

Usage:  python senxe_demo_robosuite.py
Output: cl1_nutassembly.mp4
"""
import os, sys, numpy as np
import imageio, cv2
from tqdm import tqdm
from collections import deque

from core.neurons import cl_open, warmup_calibration, is_cl_simulator, ChannelSet, StimDesign, BurstDesign
from core.decoder import AntagonisticDecoder
from core.pdi import PDI
from core.curiosity import NeuralCuriosity
from core.video import save_video
from core.vie import VIE
from core.hud import draw_overlay, hud

# ═══ Configuration ═══
SEED            = 42
ENV_NAME        = "NutAssembly"
ROBOT           = "Panda"
EPISODES        = 200
MAX_STEPS       = 200
VIDEO_FPS       = 30
VIDEO_CL1       = "cl1_nutassembly.mp4"
RECORD_LAST_N   = 80
WARMUP_SECONDS  = 10
ACTION_SCALE    = 0.25
RENDER_W        = 720
RENDER_H        = 720
INSERTION_DEPTH_THRESHOLD = 0.02
FORCE_SAFETY_THRESHOLD    = 20.0
TORQUE_SAFETY_THRESHOLD   = 5.0
PREDICTABLE_STIM_TOP_K    = 8
PREDICTABLE_BURST_N       = 15
PREDICTABLE_BURST_HZ      = 300

# ═══ RoboSuite Environment ═══
def make_robosuite_env(render=False):
    import robosuite as suite
    from robosuite.wrappers import GymWrapper
    raw = suite.make(ENV_NAME, robots=ROBOT, has_renderer=False,
                     has_offscreen_renderer=render, use_camera_obs=False,
                     render_camera="frontview", horizon=MAX_STEPS, reward_shaping=True)
    return GymWrapper(raw), raw

def extract_obs(obs, raw_env=None):
    if raw_env is not None:
        try:
            eef = raw_env.sim.data.get_site_xpos("gripper0_right_grip_site")
            vel = raw_env.sim.data.get_site_xvelp("gripper0_right_grip_site")
            peg = raw_env.sim.data.body("peg1").xpos if hasattr(raw_env.sim.data, "body") else raw_env.sim.data.get_body_xpos("peg1")
            p2h = peg - eef
            frc_raw = getattr(raw_env.robots[0], "ee_force", np.zeros(3))
            trq_raw = getattr(raw_env.robots[0], "ee_torque", np.zeros(3))
            frc = frc_raw["right"] if isinstance(frc_raw, dict) and "right" in frc_raw else frc_raw
            trq = trq_raw["right"] if isinstance(trq_raw, dict) and "right" in trq_raw else trq_raw
            frc = np.array(frc).flatten()[:3]
            trq = np.array(trq).flatten()[:3]
            p2h = np.array(p2h).flatten()[:3]
            eef = np.array(eef).flatten()[:3]
            vel = np.array(vel).flatten()[:3]
            jnt = raw_env.robots[0]._joint_positions
            nut = raw_env.sim.data.body("SquareNut_main").xpos if hasattr(raw_env.sim.data, "body") else raw_env.sim.data.get_body_xpos("SquareNut_main")
            e2n = nut - eef
            return dict(eef_pos=eef, eef_vel=vel, force=frc, torque=trq, peg_to_hole=p2h, joint_pos=jnt, eef_to_nut=e2n)
        except Exception:
            pass
    raise RuntimeError("extract_obs requires raw_env")

def compute_insertion_depth(info):
    d = np.linalg.norm(info["peg_to_hole"])
    return max(0.0, 0.1 - d), d

# ═══ CL1 Biological Agent ═══
class CL1Agent:
    def __init__(self, env, raw_env, neurons, channel_ranking=None, responsiveness=None):
        self.env = env; self.raw_env = raw_env; self.neurons = neurons
        self.action_dim = env.action_space.shape[0]
        self.vie = VIE(neurons, force_threshold=FORCE_SAFETY_THRESHOLD,
                       depth_threshold=INSERTION_DEPTH_THRESHOLD, raw_env=raw_env)
        resp_weights = responsiveness if channel_ranking is not None else None
        self.decoder = AntagonisticDecoder(self.action_dim, action_scale=ACTION_SCALE,
                                            channel_weights=resp_weights)
        self.pdi = PDI()
        self.curiosity = NeuralCuriosity()
        self.episode_rewards = []
        self.best_reward = -np.inf
        self.top_channels = (channel_ranking[:PREDICTABLE_STIM_TOP_K].tolist()
                             if channel_ranking is not None else list(range(PREDICTABLE_STIM_TOP_K)))

    def _detect_spikes(self):
        frames = self.neurons.read(250, None)
        abs_frames = np.abs(frames.astype(np.float32))
        # Enforce an absolute minimum threshold (e.g. 50uV) to prevent 
        # the percentile function from hallucinating spikes from Gaussian noise
        # when the culture is silent or dead.
        threshold = max(50.0, np.percentile(abs_frames, 99.5))
        spike_channels = list(set(np.where(abs_frames > threshold)[1]))
        firing_rates = np.mean(abs_frames, axis=0)
        return spike_channels, firing_rates

    def _predictable_stim_inject(self, reward):
        if reward <= 0: return
        amp = np.clip(reward * 2.0, 0.5, 3.0)
        s = StimDesign(200, -amp, 200, amp)
        self.neurons.stim(ChannelSet(*self.top_channels), s,
                          BurstDesign(PREDICTABLE_BURST_N, PREDICTABLE_BURST_HZ))

    def _unpredictable_stim_inject(self, penalty):
        if penalty >= 0: return
        amp = np.clip(abs(penalty) * 1.5, 0.3, 2.0)
        available = [ch for ch in range(64) if ch not in self.top_channels]
        random_chs = np.random.choice(available, size=min(8, len(available)), replace=False).tolist()
        stim = StimDesign(160, -amp, 160, amp)
        burst = BurstDesign(np.random.randint(3, 10), np.random.randint(50, 300))
        self.neurons.stim(ChannelSet(*random_chs), stim, burst)

    def run_episode(self, max_steps=MAX_STEPS, record=False, ep_num=0):
        obs, _ = self.env.reset()
        obs_info = extract_obs(obs, raw_env=self.raw_env)
        self.vie.reset(); self.pdi.reset(); self.decoder.reset(); self.curiosity.reset()
        total_reward = 0.0; frames_list = []
        ep_successes = []; ep_force_safe = []
        step_rewards = deque(maxlen=50); cur_fr = np.zeros(64)
        ep_firing_acc = []; prev_dist = None

        for step in range(max_steps):
            self.vie.encode(obs_info)
            spikes, cur_fr = self._detect_spikes()
            ep_firing_acc.append(cur_fr.copy())
            self.vie.adapt(cur_fr)
            vel = obs_info["eef_vel"]
            self.pdi.update(vel); pdi_val = self.pdi.compute()
            novelty = self.curiosity.compute_novelty(cur_fr)
            fep_boost = pdi_val * 0.3 + novelty * 0.1
            raw = self.decoder.decode(spikes, pdi_boost=fep_boost)
            action = raw
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs_info = extract_obs(obs, raw_env=self.raw_env)
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

            self._predictable_stim_inject(reward)
            penalty = 0.0
            if force_mag > FORCE_SAFETY_THRESHOLD:
                penalty -= (force_mag - FORCE_SAFETY_THRESHOLD) * 0.5
            if prev_dist is not None and cur_dist > prev_dist + 0.005:
                penalty -= (cur_dist - prev_dist) * 10.0
            if reward < -1.0:
                penalty += reward * 0.3
            self._unpredictable_stim_inject(penalty)
            prev_dist = cur_dist

            if record:
                try:
                    frame = self.raw_env.sim.render(width=RENDER_W, height=RENDER_H, camera_name="frontview")
                    if frame is not None:
                        frame = frame[::-1]
                    else:
                        frame = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
                except Exception:
                    frame = np.zeros((RENDER_H, RENDER_W, 3), dtype=np.uint8)
                if frame is not None and frame.size > 0:
                    health_full = self.neurons.get_health() if hasattr(self.neurons, 'get_health') else None
                    min_h = float(health_full.min()) if health_full is not None else 1.0
                    sr = np.mean(ep_successes) * 100
                    fsr = np.mean(ep_force_safe) * 100
                    frame = draw_overlay_wrapper(frame, ep_num, total_reward, pdi_val, min_h, cur_fr,
                                         step_rewards, distance=cur_dist,
                                         success_rate=sr, is_sim=is_cl_simulator())
                    frames_list.append(frame)

            if terminated or truncated: break

        if total_reward > self.best_reward:
            self.best_reward = total_reward

        if ep_firing_acc:
            ep_avg_fr = np.mean(ep_firing_acc, axis=0)
            hud.episode_firing_history.append(ep_avg_fr)

        success_rate = np.mean(ep_successes) * 100 if ep_successes else 0.0
        force_safe_rate = np.mean(ep_force_safe) * 100 if ep_force_safe else 100.0
        return total_reward, self.pdi.compute(), frames_list, success_rate, force_safe_rate

    def train(self, num_episodes=EPISODES, record_last_n=RECORD_LAST_N):
        print("\n" + "=" * 60)
        mode_str = "Simulator Mode" if is_cl_simulator() else "Pure Wetware Mode"
        print("  CL1 Bio-Computer Training (" + mode_str + ")")
        print("=" * 60)
        print(f"  Episodes: {num_episodes} | Env: {ENV_NAME} ({ROBOT})")
        
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

        final = np.mean(self.episode_rewards[-20:])
        final_sr = np.mean(all_sr[-20:]); final_fsr = np.mean(all_fsr[-20:])
        print(f"\n  CL1 Done | avg20={final:.2f} SR={final_sr:.1f}% FSR={final_fsr:.1f}%")
        return all_frames

# ═══ Main Entry Point ═══
def main():
    np.random.seed(SEED)
    hud.reset()
    print("+" + "=" * 58 + "+")
    if is_cl_simulator():
        print("|  Senxe Cerebellum v4.0 — cl-sdk Simulator Mode            |")
        print("|  [!] WARNING: Real CL1 hardware not detected.             |")
        print("|      Falling back to official cl-sdk Poisson simulation.  |")
    else:
        print("|  Senxe Cerebellum v4.0 — Pure Biological Wetware          |")
    print("+" + "=" * 58 + "+\n")

    print("-" * 60); print("  Phase 0: Channel Warm-up Calibration"); print("-" * 60)
    with cl_open() as neurons:
        ranking, resp = warmup_calibration(neurons, WARMUP_SECONDS)

        print("-" * 60); print("  Phase 1: Bio-Agent Control Loop"); print("-" * 60)
        env, raw_env = make_robosuite_env(render=True)
        agent = CL1Agent(env, raw_env, neurons, channel_ranking=ranking, responsiveness=resp)
        cl1_frames = agent.train(num_episodes=EPISODES, record_last_n=RECORD_LAST_N)
        env.close()

    print("\n" + "-" * 60); print("  Phase 2: Generating Video"); print("-" * 60)
    save_video(cl1_frames, VIDEO_CL1, fps=VIDEO_FPS, target_seconds=20)

    print("\n" + "=" * 60)
    print("  Execution Complete.")
    print(f"  Video saved: {VIDEO_CL1}")
    print("=" * 60)

if __name__ == "__main__":
    main()

