# CL1-Ready Protocol V1

This stage turns the multi-axis contact-skill controller into an auditable experiment.
It does not claim that simulated activity is a living culture or that a CL1 has
learned the robot task.

## Response window

The default data path is:

```text
alignment XYZ + force XYZ + external phase stimulation
-> 50 ms artifact exclusion
-> 50 ms timestamped response collection
-> 10 ms temporal bins + channel counts + first-spike latency
-> fixed zero-bias five-output decoder
-> bounded delta XYZ + safe softening + retract selection
```

`core/spike_pipeline.py` consumes the spikes and timestamps already detected by
the CL SDK. Large raw voltage samples are not reclassified as spikes. The
previous percentile detector is available only with
`SENXE_SPIKE_PIPELINE=legacy_voltage` and cannot support temporal-learning
claims.

## Frozen comparison phases

`config/cl1_protocols.json` is the source of truth for phase boundaries:

1. `calibration`: nominal controller only; encoder and decoder frozen.
2. `frozen_baseline`: CL contact skill enabled; structured feedback disabled.
3. `feedback_training`: CL contact skill and protocol feedback enabled.
4. `frozen_evaluation`: encoder, decoder and feedback frozen again.

Task phase remains external to CL1 in every phase. Feedback stimulation is
disabled automatically outside `feedback_training`. Hardware pulse amplitudes
must be reviewed and approved by the operating lab; the protocol file does not
claim a universal safe dose.

The contact experiment disables PDI/curiosity Gaussian action perturbation,
so its skill is a deterministic function of the recorded spike counts and
decoder state. The perturbation remains available only in the legacy direct
control comparison.

CL1 never controls rotation, gripper intent, joint torque, task phase or hard
safety. `soften` cannot increase nominal movement authority. `retract` selects
a deterministic force-opposing primitive rather than generating an unchecked
escape vector.

## Physical generalization

`config/contact_generalization.json` contains disjoint physics:

- training: known peg offsets, yaw angles and friction scales;
- frozen evaluation: unseen combinations and intermediate/extrapolated values.

The MuJoCo peg body position/orientation and nut collision friction are changed
after reset. Frozen-evaluation episodes automatically select only held-out
scenarios. Scenario identity and split are recorded with every control step.

## Required causal controls

- `baseline_only`: nominal controller, no CL residual and no feedback.
- `zero_spikes`: identical task and stimulation path, decoder input zeroed.
- `shuffled_spikes`: per-window channel counts permuted while total spike count
  is preserved.
- `no_feedback`: real spike path with protocol feedback disabled.

Runs must use paired environment seeds. Report task success, force safety,
residual application rate, hard stops, and phase labels. Do not select only the
best culture, seed, or episode after observing the result.

## Recording and provenance

With `SENXE_RECORD_SESSION=1`, the CL SDK HDF5 recording includes raw samples,
detected spikes, stimulations, and a synchronized `senxe_control` stream. The
project stream contains:

- protocol and phase;
- freeze and feedback flags;
- real spike timestamps, channel counts, temporal bins and artifact count;
- nominal/final actions, XYZ residual, softening and retract decisions;
- force and torque values before and after the action;
- physical scenario and structured feedback event.

The recording is the unit of evidence. Video and aggregate reward alone are
insufficient.

## Simulator source

`core/stim_responsive_source.py` supplies a seeded custom data-source model for
SDK releases that expose `cl.sim`. It produces explicit ground-truth spikes
with a stimulation artifact, response delay, exponential decay, fatigue and a
bounded feedback gain. Its purpose is to test that stimulation causally changes
the software input stream and that artifact exclusion works.

The installed legacy SDK may not expose the custom-source API. In that case the
pure model remains testable, and the adapter raises a clear upgrade error. The
importable factory for a compatible SDK is:

```text
core.stim_responsive_source:create_stim_responsive_source
```

Behavior produced by this model is synthetic and must never be presented as
CL1 adaptation.

## Evidence ledger

The JSON protocol file stores mechanism cards separately from the Senxe
proposal:

- Doom-neuron informs the configurable response-window and simple frozen
  decoder controls. No GPL source is copied.
- DishBrain/Pong supports testing low-dimensional closed-loop structured
  feedback against no-feedback controls. It does not establish transfer to
  mechanical-arm control.
- Senxe's contact-skill protocol is an independent project design awaiting
  real CL1 evaluation.
