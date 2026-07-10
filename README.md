# Senxe Cerebellum: Biologically-Grounded Robotic Motor Control

Senxe Cerebellum is an open-source research framework that interfaces living biological neural networks (via the **Cortical Labs CL1** microelectrode array platform) with high-precision industrial robotic manipulators. 

The framework maps multi-modal physical sensor readings (force, torque, kinematics) into closed-loop electrical stimulation patterns and decodes biological firing outputs (spikes) into continuous action trajectories to solve force-sensitive assembly tasks (such as the RoboSuite NutAssembly benchmark).

> [!NOTE]
> **Hardware Fallback**: This framework is built directly on the official Cortical Labs `cl-sdk`. It automatically detects physical hardware; when a CL1 device is not present, it gracefully falls back to the SDK's official Poisson simulation server, enabling developers and researchers to run the entire pipeline locally.

---

## Core Scientific Modules

### 1. Neuromorphic Event-Driven Sparse Coding (VIE)
The **Virtual Interference Encoding (VIE)** module ([core/vie.py](file:///Users/azur/Desktop/github_repo/core/vie.py)) maps continuous environment observations onto the 64-channel microelectrode array (MEA). 
*   **Sparse Encoding**: To minimize signal crosstalk and cellular overstimulation, sensory modalities (e.g., force/torque deltas) are encoded using delta-tracking. Electrodes are only stimulated when physical quantities change significantly.
*   **Attention Multiplexing**: Channels are dynamically reallocated between visual/tactile sensory modalities based on the robot's current task phase (searching vs. inserting).

### 2. Antagonistic Muscle-Pair Decoding
Motor outputs are decoded based on the biological flexor/extensor antagonistic principle ([core/decoder.py](file:///Users/azur/Desktop/github_repo/core/decoder.py)). 
*   **Opposing Populations**: The 64 channels are interleaved into opposing sub-populations (Even/Odd pairs). Each of the 7 action dimensions is driven by the differential activity:
    $$\text{Action}[i] = \frac{\text{flexor\_sum} - \text{extensor\_sum}}{\text{flexor\_sum} + \text{extensor\_sum} + \epsilon}$$
*   **EMA Inertia Filter**: Outputs are smoothed using an Exponential Moving Average (EMA) filter to mimic the biomechanical damping and inertia of physical muscle tissue, producing jerk-free trajectories.

### 3. FEP-Driven Kinematic Gate (PDI)
Rather than using hand-tuned exploration schedules, exploration is regulated by the **Physical Disturbance Index (PDI)** ([core/pdi.py](file:///Users/azur/Desktop/github_repo/core/pdi.py)). Inspired by the Free Energy Principle (FEP):
*   **High PDI** (unstable kinematics, high sensory surprise) increases Gaussian perturbation to force exploration and surprise minimization.
*   **Low PDI** (stable kinematics, low surprise) limits perturbation to exploit the current steady-state control policy.

### 4. Intrinsic Firing-Rate Curiosity
The **Neural Curiosity** module ([core/curiosity.py](file:///Users/azur/Desktop/github_repo/core/curiosity.py)) monitors firing pattern novelty. Novel electrophysiological patterns boost the exploration rate, driving the neural network to escape local minima in silent or repetitive states.

---

## Major Updates (Compared to April 2026 Release)

Since the initial release (`a1057ea` on April 13, 2026), the framework has undergone major refactoring, bug fixing, and scientific alignment:

### 1. Critical Control Loop Fixes
*   **Double Action Scaling Bug**: Resolved an issue where actions were scaled twice in both the Agent loop and the Antagonistic Decoder, which previously caused the robotic arm to stall.
*   **GymWrapper Flattening Fix**: Bypassed GymWrapper observation flattening inside `extract_obs`. This restores access to structured observation dictionaries (native force, torque, and target vector values) from the MuJoCo simulation.
*   **Action Bias Normalization**: Replaced an unconditioned, exponentially growing `action_bias` update with a bounded, clipped heuristic to prevent motor command divergence.

### 2. SDK Integration & Robustness
*   **Idempotent Context Management**: Fixed a double-close bug in the `cl_open()` context manager that caused `ClosedNodeError` crashes in PyTables on exit. The `Neurons.close()` method is now monkeypatched to be fully idempotent.
*   **STDP Plasticity Sign Inversion**: Corrected a biological STDP bug in the mock neuron simulator where Pre-before-Post spikes incorrectly triggered long-term depression (LTD) instead of long-term potentiation (LTP).
*   **NumPy 2.x Compatibility**: Added shims to support running with NumPy 2.x, silencing internal deprecation warnings from the legacy parts of the `cl-sdk`.

### 3. Scientific Rigor & Benchmarking
*   **FEP Terminology Alignment**: Deep-cleaned the codebase to replace reward-centric terminology (like "Dopamine Injection" and "Punishment") with information-theoretic terminology ("Predictable Stimulation" and "Unpredictable Stimulation"), aligning with the Free Energy Principle.
*   **Ablation Benchmark Suite**: Added a dedicated benchmark runner ([run_ablation_benchmark.py](file:///Users/azur/Desktop/github_repo/run_ablation_benchmark.py)) and visualization utility ([plot_ablations.py](file:///Users/azur/Desktop/github_repo/plot_ablations.py)). It runs paired-seed trials to compare the biological agent against control groups (no-stim, zero-spikes, and randomized-spikes).
*   **Fair Baseline Comparison**: Removed hindsight experience replay (HER) reward injection during the PPO evaluation loop to guarantee a scientifically honest comparison between biological and silicon baselines.

---

## Quick Start

### 1. Installation
Install the necessary simulator and reinforcement learning baselines:
```bash
git clone https://github.com/AzurLiu/Senxe-Cerebellum.git
cd Senxe-Cerebellum
pip install -r requirements.txt
pip install cl-sdk
```

### 2. Configure MuJoCo Backend (macOS)
```bash
export MUJOCO_GL=glfw
```

### 3. Run the Biological Benchmark
Run the primary training script:
```bash
python senxe_demo_robosuite.py
```
This script runs the 7-DoF Franka Panda robot arm task, trains the biological agent, and saves a Cyberpunk-styled video overlay `cl1_nutassembly.mp4` displaying the MEA grid, force telemetry, and live status watermarks.

### 4. Run the Ablation Study
To run the automated information-nullification benchmark:
```bash
python run_ablation_benchmark.py
python plot_ablations.py
```
This will run headless control trials and save a learning curve comparison to `ablation_plot.png`.

---

## Author & License

*   **Author**: Azur (Jiahao) — Independent developer, incoming University of Alberta student.
*   **License**: Licensed under the MIT License (changed from AGPL v3 in June 2026).
