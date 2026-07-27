# Hybrid Residual V1

This branch is a narrow mechanical-arm experiment, not a general biological
robotics platform.

## Control boundary

The default RoboSuite path uses:

```text
deterministic task phase
-> bounded nominal xyz/gripper action
-> one-axis CL1 residual
-> confidence and force gates
-> final clipped action
```

Long-horizon task phase stays in `NominalTaskController`. The CL1 decoder may
modify only the configured translation axis, which defaults to Z.

Environment variables:

```text
SENXE_CONTROL_MODE=hybrid_residual
SENXE_RESIDUAL_AXIS=2
SENXE_RESIDUAL_SCALE=0.08
SENXE_MAX_RESIDUAL_ABS=0.05
SENXE_MIN_RESIDUAL_CONFIDENCE=0.15
```

The previous direct decoder path remains available only for comparison:

```text
SENXE_CONTROL_MODE=legacy_wetware
```

## Safety behavior

- Empty or low-confidence spike responses contribute no residual.
- The residual is clipped independently of the nominal controller.
- The soft force limit removes CL1 residual authority.
- The hard force limit emits a zero action.
- Translation norm and final action values are clipped.

## Evidence boundary

This stage validates software composition and safety behavior only. It does not
claim biological learning. Hardware claims require synchronized raw recording,
stimulation events, artifact handling, session provenance, and ablation
controls.

The ablation CSV records residual application rate, mean absolute residual,
and hard-stop count in addition to reward and task safety metrics.
