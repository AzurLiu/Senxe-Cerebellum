#!/usr/bin/env python3
"""
Senxe Cerebellum — Automated Ablation Benchmark
=============================================
Runs headless simulations of various ablation modes to measure 
the true information content of the biological (or simulated) spikes.

Conditions:
1. contact_skill: Real spikes + structured feedback
2. baseline_only: No spike residual + No feedback stim
3. zero_spikes: Decoder receives [] + Real feedback stim
4. shuffled_spikes: Per-window counts are channel-shuffled
5. no_feedback: Real spikes + No reward/penalty feedback stim
"""

import csv
import os
import numpy as np
from tqdm import tqdm

from senxe_demo_robosuite import make_robosuite_env, CL1Agent
from core.neurons import cl_open, timestamped_warmup_calibration

SEED = 42
EPISODES_PER_CONDITION = int(os.getenv("SENXE_ABLATION_EPISODES", "75"))

CONDITIONS = [
    {"name": "contact_skill", "spike_mode": "none", "stim_mode": "full"},
    {"name": "baseline_only", "spike_mode": "zero", "stim_mode": "none"},
    {"name": "zero_spikes", "spike_mode": "zero", "stim_mode": "full"},
    {"name": "shuffled_spikes", "spike_mode": "shuffled", "stim_mode": "full"},
    {"name": "no_feedback", "spike_mode": "none", "stim_mode": "none"},
]

def main():
    print("=" * 60)
    print("  CL1 Ablation Benchmark")
    print(f"  Episodes per condition: {EPISODES_PER_CONDITION}")
    print("=" * 60)

    # Open neurons (or simulator)
    with cl_open() as neurons:
        # Phase 0: Calibration
        ranking, resp = timestamped_warmup_calibration(
            neurons,
            duration_sec=5.0,
        )

        # Create headless environment
        env, raw_env = make_robosuite_env(render=False)

        results_log = []

        for cond in CONDITIONS:
            name = cond["name"]
            spike_mode = cond["spike_mode"]
            stim_mode = cond["stim_mode"]
            
            print(f"\n---> Starting Condition: {name} <---")
            # Reset random seed per condition to guarantee paired episode layouts
            np.random.seed(SEED)
            
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
                control_summary = agent.last_episode_control_summary
                
                # Log metrics
                results_log.append([
                    ep,
                    name,
                    agent.current_protocol_phase.phase.value,
                    (
                        "none"
                        if agent.active_scenario is None
                        else agent.active_scenario.scenario_id
                    ),
                    (
                        "none"
                        if agent.active_scenario is None
                        else agent.active_scenario.split
                    ),
                    reward,
                    sr,
                    fsr,
                    control_summary["residual_applied_rate"],
                    control_summary["mean_abs_applied_residual"],
                    control_summary["hard_stop_count"],
                    control_summary.get("mean_residual_norm", 0.0),
                    control_summary.get("mean_compliance_scale", 1.0),
                    control_summary.get("retract_count", 0),
                ])

                pbar.set_postfix(R=f"{reward:.1f}", SR=f"{sr:.0f}%", FSR=f"{fsr:.0f}%")

        env.close()

    # Save to CSV
    csv_filename = "ablation_results.csv"
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Episode",
            "Condition",
            "ProtocolPhase",
            "Scenario",
            "ScenarioSplit",
            "Reward",
            "SuccessRate",
            "ForceSafeRate",
            "ResidualAppliedRate",
            "MeanAbsAppliedResidual",
            "HardStopCount",
            "MeanResidualNorm",
            "MeanComplianceScale",
            "RetractCount",
        ])
        writer.writerows(results_log)
    
    print("\n" + "=" * 60)
    print(f"  Benchmark Complete! Results saved to {csv_filename}")
    print("  Run 'python plot_ablations.py' to visualize the learning curves.")
    print("=" * 60)

if __name__ == "__main__":
    main()
