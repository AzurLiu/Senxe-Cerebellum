# CL1 Contact-Skill Protocol V2

## Experimental target

The active task is RoboSuite `NutAssemblySquare`: one Panda arm places the
square nut on peg1. Official environment success is the only success endpoint.
No end-effector-distance heuristic is permitted.

The CL path receives nut-to-peg XYZ alignment, force XYZ, externally maintained
phase, and contact severity. It emits:

```text
[delta_x, delta_y, delta_z, soften, retract]
```

Planning, grasp intent, rotation, joints, force hard stops, and success
verification remain deterministic.

## Channel allocation

The confirmatory contact path uses the versioned
`senxe_contact_18_20_6_15_v1` map:

| Region | Count | Encoding / readout |
|---|---:|---|
| sensory | 18 | position XYZ sign pairs, force XYZ sign pairs, three-bit phase, three contact states |
| motor | 20 | five outputs with two positive and two negative electrodes each |
| feedback | 6 | three positive and three negative electrodes |
| reserve | 15 | unused during confirmatory runs |

The exact default assignments are:

| Function | Channels |
|---|---|
| position `X± / Y± / Z±` | `8,9 / 10,17 / 18,25` |
| force `X± / Y± / Z±` | `27,28 / 57,1 / 2,3` |
| phase / contact | `5,6,11 / 12,15,16` |
| `delta_x` positive / negative | `41,42 / 50,51` |
| `delta_y` positive / negative | `13,14 / 45,46` |
| `delta_z` positive / negative | `29,30 / 59,60` |
| `soften` positive / negative | `32,33 / 49,58` |
| `retract` positive / negative | `34,37 / 53,61` |
| positive / negative feedback | `19,20,22 / 23,24,26` |
| reserve | `21,31,35,36,38,39,40,43,44,47,48,52,54,55,62` |

Channels `0`, `4`, `7`, `56`, and `63` are non-stimulatable and excluded.
All four regions are pairwise disjoint and together account for all 59
stimulatable channels. Position and force magnitudes are encoded through
bounded amplitude, not separate small/large electrodes. The motor decoder
ignores every channel outside its 20-electrode readout.

The default channel numbers are fixed before confirmatory data collection.
Any culture-specific replacement must draw from the 15-channel reserve,
follow a prospectively specified calibration rule, create a new channel-map
hash, and may not be selected using task outcomes.

## Response timing

```text
sensory stimulation
-> 50 ms artifact exclusion
-> 50 ms timestamped response collection
-> zero-bias frozen count readout
-> confidence / phase / force supervisor
```

Measured CL timestamps, channel counts, time bins, first-spike latency, and
stimulation timestamps are recorded. Aggregate counts are never expanded into
synthetic spike times.

## Protocol phases

| Phase | Episodes | Feedback | Held-out | Rest before |
|---|---:|---|---|---:|
| calibration | 5 | off | no | 0 |
| frozen baseline | 25 | off | no | 0 |
| feedback training | 50 | on | no | 0 |
| frozen evaluation | 20 | off | yes | 0 |
| retention 5 min | 20 | off | yes | 5 min |
| retention 15 min | 20 | off | yes | +10 min |
| retention 45 min | 20 | off | yes | +30 min |

The delays are cumulative from the end of training. Simulator runs skip the
wall-clock waits because the official simulator is non-learning.

## Conditions

- `contact_skill`: measured spikes and outcome-linked structured feedback;
- `baseline_only`: deterministic controller, no sensory stimulation, zero
  spike residual, and no feedback;
- `zero_spikes`: zeroed readout with real outcome-linked feedback;
- `shuffled_spikes`: count-preserving channel shuffle with real feedback;
- `no_feedback`: measured spikes without task feedback;
- `yoked_feedback`: measured spikes with an independent donor's feedback
  timing and dose.

Real hardware accepts one condition per culture invocation by default. Every
run records culture ID, sequence index, counterbalanced order, protocol phase,
physics scenario, response timestamps, action, force/torque, official success,
time to success, culture health telemetry, the protocol SHA-256, the exact
runtime-source SHA-256, channel-map version and SHA-256, and the current Git
commit.
Python, CL SDK, RoboSuite, MuJoCo, Gymnasium, and NumPy versions are recorded
with the same provenance block.

## Hardware gates

Real stimulation is blocked unless `SENXE_LAB_APPROVED_STIM=1` is supplied
after laboratory review. `SENXE_MAX_STIM_AMPLITUDE_UA` sets the reviewed
maximum used by sensory and feedback stimulation; the default candidate value
is 1.5 µA and is not itself an approval. Real runs must explicitly set the
approved phase width, burst frequency, burst count, call count, channel-pulse
count, and absolute-charge limits. The proxy rejects any SDK call outside this
electrical envelope and rejects all five non-stimulatable channels. Real runs
require `SENXE_RECORD_SESSION=1`; calibration and control are written as
separate raw HDF5 evidence recordings.

Calibration measures input-evoked network response and output-electrode
response. A real run fails before task execution if its assigned map does not
pass the frozen thresholds. Reserve remapping is disabled by default and, when
prospectively enabled, is deterministic, calibration-only, and produces a new
channel-map hash.

Structured feedback has a three-step cooldown. Each response window records
triggering stimulation timestamps, trigger-to-collection delay, expected and
observed ticks, and a timing-valid flag. A stimulated real-hardware step stops
if the artifact-exclusion timing gate fails. Per-episode dose deltas are
recorded alongside cumulative dose.

## Analysis

```bash
python analyze_ablations.py ablation_results.csv
python plot_ablations.py
```

`analyze_ablations.py` aggregates episodes within culture before computing
condition estimates. If a culture appears in more than one condition, the
built-in report remains descriptive but automatically marks the confirmatory
gate invalid because it does not fit a crossover carryover model. The complete
claim and falsification contract is in
`docs/PREREGISTRATION_V2.md`.
