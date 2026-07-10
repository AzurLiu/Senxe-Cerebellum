#!/usr/bin/env python3
"""
Senxe Cerebellum — Automated Ablation Benchmark
=============================================
Runs headless simulations of various ablation modes to measure 
the true information content of the biological (or simulated) spikes.

Conditions:
1. none (Control): Real spikes + Real stim
2. zero_spikes: Decoder receives [] + Real stim
3. random_spikes: Decoder receives random channels + Real stim
4. no_stim: Real spikes + No reward/penalty stim
"""

import csv
import numpy as np
from tqdm import tqdm

from senxe_demo_robosuite import make_robosuite_env, CL1Agent
from core.neurons import cl_open, warmup_calibration

SEED = 42
EPISODES_PER_CONDITION = 50

CONDITIONS = [
    {"name": "none (Control)", "spike_mode": "none", "stim_mode": "full"},
    {"name": "zero_spikes", "spike_mode": "zero", "stim_mode": "full"},
    {"name": "random_spikes", "spike_mode": "random", "stim_mode": "full"},
    {"name": "no_stim", "spike_mode": "none", "stim_mode": "none"},
]

def main():
    print("=" * 60)
    print("  CL1 Ablation Benchmark")
    print(f"  Episodes per condition: {EPISODES_PER_CONDITION}")
    print("=" * 60)

    # Use the same seed for reproducible environment initializations
    np.random.seed(SEED)

    # Open neurons (or simulator)
    with cl_open() as neurons:
        # Phase 0: Calibration
        ranking, resp = warmup_calibration(neurons, duration_sec=5.0)

        # Create headless environment
        env, raw_env = make_robosuite_env(render=False)

        results_log = []

        for cond in CONDITIONS:
            name = cond["name"]
            spike_mode = cond["spike_mode"]
            stim_mode = cond["stim_mode"]
            
            print(f"\n---> Starting Condition: {name} <---")
            # Create a fresh agent for each condition to reset internal state (EMA, PDI, etc.)
            agent = CL1Agent(
                env, raw_env, neurons, 
                channel_ranking=ranking, 
                responsiveness=resp,
                ablation_spike_mode=spike_mode,
                ablation_stim_mode=stim_mode
            )

            pbar = tqdm(range(EPISODES_PER_CONDITION), desc=name, ncols=90)
            for ep in pbar:
                # Run headless episode
                reward, pdi_val, _, sr, fsr = agent.run_episode(max_steps=200, record=False, ep_num=ep)
                
                # Log metrics
                results_log.append([ep, name, reward, sr, fsr])

                pbar.set_postfix(R=f"{reward:.1f}", SR=f"{sr:.0f}%", FSR=f"{fsr:.0f}%")

        env.close()

    # Save to CSV
    csv_filename = "ablation_results.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Condition", "Reward", "SuccessRate", "ForceSafeRate"])
        writer.writerows(results_log)
    
    print("\n" + "=" * 60)
    print(f"  Benchmark Complete! Results saved to {csv_filename}")
    print("  Run 'python plot_ablations.py' to visualize the learning curves.")
    print("=" * 60)

if __name__ == "__main__":
    main()
