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
except ImportError:
    print("FATAL ERROR: matplotlib is not installed.")
    print("This script is an optional analysis tool.")
    print("Please run: pip install matplotlib")
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

    # Parse CSV by condition, retaining held-out split and contact metrics.
    data = {}
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cond = row["Condition"]
            ep = int(row["Episode"])
            if cond not in data:
                data[cond] = {
                    "eps": [],
                    "reward": [],
                    "sr": [],
                    "fsr": [],
                    "residual": [],
                    "compliance": [],
                    "retract": [],
                    "split": [],
                }
            
            data[cond]["eps"].append(ep)
            data[cond]["reward"].append(float(row["Reward"]))
            episode_success = row.get("EpisodeSuccess")
            if episode_success not in (None, ""):
                success_percent = 100.0 * float(episode_success)
            else:
                success_percent = float(row["SuccessRate"])
            data[cond]["sr"].append(success_percent)
            data[cond]["fsr"].append(float(row["ForceSafeRate"]))
            data[cond]["residual"].append(float(
                row.get("MeanResidualNorm", 0.0)
            ))
            data[cond]["compliance"].append(float(
                row.get("MeanComplianceScale", 1.0)
            ))
            data[cond]["retract"].append(float(
                row.get("RetractCount", 0.0)
            ))
            data[cond]["split"].append(row.get("ScenarioSplit", "none"))

    # Sort each condition by episode
    for cond in data.values():
        zipped = sorted(zip(
            cond["eps"],
            cond["reward"],
            cond["sr"],
            cond["fsr"],
            cond["residual"],
            cond["compliance"],
            cond["retract"],
            cond["split"],
        ))
        cond["eps"] = [z[0] for z in zipped]
        cond["reward"] = [z[1] for z in zipped]
        cond["sr"] = [z[2] for z in zipped]
        cond["fsr"] = [z[3] for z in zipped]
        cond["residual"] = [z[4] for z in zipped]
        cond["compliance"] = [z[5] for z in zipped]
        cond["retract"] = [z[6] for z in zipped]
        cond["split"] = [z[7] for z in zipped]

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax1, ax2, ax3, ax4 = axes.flatten()
    fig.suptitle(
        "Senxe Cerebellum — Contact Skill Ablation & Generalization",
        fontsize=14,
        fontweight="bold",
    )

    colors = {
        "contact_skill": "#2ecc71",  # Green
        "baseline_only": "#34495e",  # Slate
        "zero_spikes": "#e74c3c",     # Red
        "shuffled_spikes": "#f39c12", # Orange
        "no_feedback": "#9b59b6",     # Purple
        "yoked_feedback": "#16a085",  # Teal
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

        residual_ma = moving_average(metrics["residual"], window_size=10)
        ax3.plot(eps, residual_ma, label=cond, color=c, linewidth=2)

        compliance_ma = moving_average(
            metrics["compliance"],
            window_size=10,
        )
        ax4.plot(
            eps,
            compliance_ma,
            label=f"{cond} compliance",
            color=c,
            linewidth=2,
        )
        retract_ma = moving_average(metrics["retract"], window_size=10)
        ax4.plot(
            eps,
            retract_ma,
            color=c,
            linewidth=1,
            linestyle=":",
            alpha=0.7,
        )

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

    ax3.set_title("CL XYZ Residual Norm (10-ep Moving Avg)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Residual norm")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend()

    ax4.set_title("Compliance Scale (solid) / Retracts (dotted)")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Value per episode")
    ax4.grid(True, linestyle="--", alpha=0.5)

    heldout_starts = [
        metrics["eps"][index]
        for metrics in data.values()
        for index, split in enumerate(metrics["split"])
        if split == "heldout"
    ]
    if heldout_starts:
        heldout_start = min(heldout_starts)
        for axis in (ax1, ax2, ax3, ax4):
            axis.axvline(
                heldout_start,
                color="#2980b9",
                linestyle="--",
                linewidth=1.5,
            )
            axis.axvspan(
                heldout_start,
                max(max(metrics["eps"]) for metrics in data.values()),
                color="#3498db",
                alpha=0.06,
            )
        ax1.text(
            heldout_start,
            100,
            " held-out physics",
            color="#2980b9",
            va="top",
        )

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
