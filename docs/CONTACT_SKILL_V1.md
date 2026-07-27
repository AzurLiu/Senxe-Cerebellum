# Contact Skill V1

The default CL1 authority boundary is:

```text
inputs:  alignment_error_xyz, contact_force_xyz, external_phase
outputs: delta_xyz, soften, retract
```

The deterministic system retains target recognition, approach, grasp,
transport, rotation, gripper control, joint control and safety.

## Output semantics

- `delta_x/y/z`: independently scaled and clipped local contact correction.
- `soften`: may reduce nominal translation to 35%; it cannot increase it.
- `retract`: selects a deterministic motion opposite measured force.

CL1 output is ignored during calibration, approach and grasp. Low-confidence
responses fall back to nominal control. The soft force limit removes CL1
authority; the hard force limit emits a zero action.

## Generalization claim boundary

Training and held-out scenarios use different peg positions, yaw angles and
friction values. Simulator performance demonstrates experiment plumbing only.
A real learning claim requires the same culture to:

1. improve during structured feedback;
2. retain improvement with encoder, decoder and feedback frozen;
3. outperform zero-spike, shuffled-spike, no-feedback and nominal controls;
4. preserve the advantage on held-out physical scenarios;
5. remain within force and retract safety limits.

No current simulator result establishes these biological outcomes.
