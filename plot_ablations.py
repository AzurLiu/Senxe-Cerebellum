#!/usr/bin/env python3
"""
Senxe Cerebellum — Ablation Visualization
=======================================
Reads ablation_results.csv and plots the smoothed learning curves
for each ablation condition using Matplotlib.
"""

import sys
import os
import csv

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("FATAL ERROR: matplotlib or numpy is not installed.")
    print("This script is an optional analysis tool and requires matplotlib.")
    print("Please run: pip install matplotlib numpy")
    sys.exit(1)

def moving_average(data, window_size=20):
    """Computes a simple moving average, padding the start."""
    if len(data) == 0:
        return []
    ma = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        chunk = data[start:i+1]
        ma.append(sum(chunk) / len(chunk))
    return ma

def main():
    csv_file = "ablation_results.csv"
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Please run run_ablation_benchmark.py first.")
        sys.exit(1)

    # Parse CSV: condition -> { ep: {"reward": X, "sr": Y, "fsr": Z} }
    data = {}
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cond = row["Condition"]
            ep = int(row["Episode"])
            if cond not in data:
                data[cond] = {"eps": [], "reward": [], "sr": [], "fsr": []}
            
            data[cond]["eps"].append(ep)
            data[cond]["reward"].append(float(row["Reward"]))
            data[cond]["sr"].append(float(row["SuccessRate"]))
            data[cond]["fsr"].append(float(row["ForceSafeRate"]))

    # Sort each condition by episode
    for cond in data.values():
        zipped = sorted(zip(cond["eps"], cond["reward"], cond["sr"], cond["fsr"]))
        cond["eps"] = [z[0] for z in zipped]
        cond["reward"] = [z[1] for z in zipped]
        cond["sr"] = [z[2] for z in zipped]
        cond["fsr"] = [z[3] for z in zipped]

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Senxe Cerebellum — Information Nullification Ablation Study", fontsize=14, fontweight="bold")

    colors = {
        "none (Control)": "#2ecc71",  # Green
        "zero_spikes": "#e74c3c",     # Red
        "random_spikes": "#f39c12",   # Orange
        "no_stim": "#9b59b6",         # Purple
    }

    for cond, metrics in data.items():
        c = colors.get(cond, "#333333")
        eps = metrics["eps"]
        
        # Plot Success Rate
        sr_ma = moving_average(metrics["sr"], window_size=20)
        ax1.plot(eps, sr_ma, label=cond, color=c, linewidth=2, alpha=0.9)
        ax1.scatter(eps, metrics["sr"], color=c, s=10, alpha=0.1)  # Raw data points faintly in bg

        # Plot Reward
        r_ma = moving_average(metrics["reward"], window_size=20)
        ax2.plot(eps, r_ma, label=cond, color=c, linewidth=2, alpha=0.9)
        ax2.scatter(eps, metrics["reward"], color=c, s=10, alpha=0.1)

    ax1.set_title("Success Rate (20-ep Moving Avg)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Success Rate (%)")
    ax1.set_ylim(-5, 105)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.set_title("Total Reward (20-ep Moving Avg)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    output_png = "ablation_plot.png"
    plt.savefig(output_png, dpi=200)
    print(f"Plot saved to {output_png}")
    
    # Try to show interactively if supported
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
