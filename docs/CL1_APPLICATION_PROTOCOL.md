# CL1 Access-Application Protocol

## Purpose

`senxe_contact_skill_application_v1` is the short, default demonstration
profile. It shows that Senxe has bounded biological authority, fail-closed
hardware gates, measurable stimulation/response timing, and a credible path
to a later learning experiment.

It is not a retained-learning experiment and its simulator output is not
biological evidence.

## Scope

The application profile runs 22 episodes:

| Phase | Episodes | Feedback | Scenario |
|---|---:|---|---|
| calibration | 2 | off | train |
| frozen baseline | 5 | off | train |
| feedback pilot | 10 | cooldown-limited | train |
| frozen evaluation | 5 | off | held-out |

The full 160-episode, six-condition confirmatory design remains
`senxe_contact_skill_v2` and is selected explicitly by
`SENXE_BENCHMARK_PROTOCOL_ID`.

## Hardware gates

Before a real task episode can start:

1. the lab must explicitly approve every stimulation limit;
2. calibration must measure input-evoked and output-recorded response;
3. the assigned channel map must pass its frozen health thresholds, or an
   explicitly enabled calibration-only reserve remap must succeed;
4. the map, calibration report and safety limits must be recorded.

Every stimulation SDK call is rejected if it exceeds any approved:

- current amplitude;
- phase width;
- burst frequency;
- burst count;
- cumulative call or channel-pulse count;
- cumulative absolute charge;
- charge-balance tolerance.

Each episode records its own dose delta in addition to cumulative dose.

## Timing gate

The compact sensory encoder records the SDK timestamp of every stimulation.
The response window records the last trigger-to-collection delay, expected and
observed tick counts, and a timing-valid flag. Real CL1 control stops if a
stimulated step does not satisfy the artifact-exclusion interval.

## Default and confirmatory commands

```bash
python senxe_demo_robosuite.py
```

The default command uses the 22-episode application profile and does not
construct the VIE, PDI, Curiosity, legacy decoder or single-axis residual
controller.

```bash
SENXE_BENCHMARK_PROTOCOL_ID=senxe_contact_skill_v2 \
python run_ablation_benchmark.py --condition contact_skill
```

The confirmatory runner retains all preregistered controls and retention
phases. It should only be used after access, dose and sample-size approval.
