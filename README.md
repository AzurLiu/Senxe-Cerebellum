<p align="center">
  <h1 align="center">Senxe Cerebellum v4.0 (Pure Wetware)</h1>
  <p align="center">
    <strong>100% Biologically-Grounded Motor Control for Industrial Robotics</strong><br>
    Interfacing Cortical Labs CL1 Biological Neurons with Robotic Arms
  </p>
</p>

<p align="center">
  <a href="#what-is-this-project">What is this?</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#core-philosophy">Core Philosophy</a> •
  <a href="#architecture">Architecture</a>
</p>

---

## What is this project?

> *"What if we stopped training artificial networks to mimic biology — and just used the biology itself?"*

Most modern robotics rely on Artificial Intelligence (like Deep Reinforcement Learning) running on silicon chips (GPUs) to control robotic arms. **Senxe Cerebellum takes a radically different approach: it replaces the AI with a living biological brain.**

This project is a software bridge that connects a **Cortical Labs CL1** — a physical microelectrode array (MEA) culturing hundreds of thousands of living biological neurons (a "mini-brain" or organoid) — directly to a 7-DoF industrial robotic arm (Franka Panda). 

Instead of writing code to calculate how the robot should move, we stream physical sensations (force, torque, and spatial distance) directly into the living cells as electrical pulses. The biological neurons process this "pain" and "reward", and their natural electrical firing patterns (spikes) are decoded in real-time to physically move the robotic arm and assemble a nut onto a peg.

**This is not a simulation. This is a 100% Pure Wetware interface.**

---

## Core Philosophy

### 1. Zero Software Fallbacks (Pure Biology)
We do not simulate biology. The codebase strictly requires the Cortical Labs `cl-sdk`. If you do not have physical access to a real biological organoid, the code will refuse to execute. We have violently purged all "Mock" simulators and Deep Learning (PPO) training wheels.

### 2. Neuromorphic Event-Driven Sparse Coding (VIE)
The **Virtual Interference Encoding (VIE)** module translates continuous robotic force/torque sensory data into event-driven Delta Tracking. The culture only receives an electrical shock when physical parameters *change*, drastically improving Signal-to-Noise Ratio (SNR) and completely preventing global overstimulation seizures.

### 3. Closed-Loop Homeostasis
Biological tissue exhausts if constantly stimulated. Channel adaptation (`channel_gain`) is physically hooked into the stimulation amplitude and burst frequency, dynamically preventing cellular fatigue and cell death while preserving sensitive pathways.

### 4. Antagonistic Decoding
Motor output follows the biological **flexor/extensor antagonistic** principle. 
The 64 channels are perfectly balanced and mapped across the 7-DoF spatial and gripper dimensions. It reads the raw electrical spikes from the living cells, balances the competing signals, and outputs smooth mechanical motion.

---

## Quick Start

### Hardware Requirements
You **MUST** have access to Cortical Labs CL1 biological hardware.

```bash
# Clone and install
git clone https://github.com/AzurLiu/Senxe-Cerebellum.git
cd Senxe-Cerebellum
pip install -r requirements.txt
pip install cl-sdk

# Set MuJoCo renderer (macOS)
export MUJOCO_GL=glfw

# Run the biological assembly benchmark
python senxe_demo_robosuite.py
```

---

## Output Artifacts

| File | Description |
|:---:|:---|
| `cl1_nutassembly.mp4` | CL1 bio-agent execution video with Cyberpunk F/T Bloom HUD overlay. |

*(Note: Generating this video requires a successful run on the physical CL1 hardware).*

---

## Architecture

### Biological Control Pipeline

```text
┌─────────────────────────────────────────────────────────────────┐
│                    SENXE CONTROL LOOP                            │
│                                                                 │
│  ┌──────────┐    ┌─────────┐    ┌──────────────┐    ┌────────┐ │
│  │ RoboSuite│───▶│   VIE   │───▶│  64-ch MEA   │───▶│Antagon.│ │
│  │  Sensors │    │ Encoder │    │ (Real CL1)   │    │Decoder │ │
│  │ F/T, Pos │    │(Sparse) │    │              │    │        │ │
│  └──────────┘    └─────────┘    └──────┬───────┘    └───┬────┘ │
│       ▲                                │                │      │
│       │                          ┌─────▼─────┐    ┌─────▼────┐ │
│       │                          │ Predictable / │    │  Action  │ │
│       └──────────────────────────│ Unpredictable Stimulus │◀───│  Output  │ │
│              (env.step)          │ Injection  │    │ (7D OSC) │ │
│                                  └─────┬─────┘    └──────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## Author

**Azur (Jiahao)** — 18-year-old independent developer, incoming University of Alberta student.

Built from scratch as an exploration into biological computation and neural motor control for industrial robotics. This project represents a first-principles approach to neuromorphic engineering: rather than training artificial networks to approximate biological dynamics, Cerebellum interfaces directly with living biological neural tissue to solve real-world motor control problems.

---

Copyright © 2026 Azur (Jiahao). Licensed under the MIT License.
