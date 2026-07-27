# Senxe Contact-Skill V2 — Preregistered Evaluation Contract

## Claim boundary

The experiment tests whether a CL1 culture can improve and retain a bounded
five-output contact skill. It does not test end-to-end manipulation, general
intelligence, consciousness, or a replacement for deterministic robot safety.
Simulator runs are deployment tests and cannot satisfy any biological claim.

## Frozen channel map

Confirmatory runs use the fixed 18 sensory / 20 motor / 6 feedback / 15
reserve allocation identified by its channel-map version and SHA-256. The five
non-stimulatable CL1 channels are excluded. Sensory, motor, feedback, and
reserve regions may not overlap, and the five-output decoder reads only the
20 motor electrodes.

Channel replacement is permitted only under a prospective, outcome-blinded
health rule and must draw from the reserve region. It creates a different
channel-map hash and cannot be pooled with the original map unless that pooling
rule was declared before condition unblinding.

The runtime implements replacement as deterministic calibration-only reserve
selection. Remapping is disabled unless explicitly authorized before
collection. Real task execution fails closed when the calibrated map cannot
meet the declared input-evoked and motor-output response thresholds.

## Unit of replication and sample size

The biological culture, not an episode, is the independent unit. A
confirmatory comparison requires at least six independent cultures per
condition. A blinded pilot may be used to increase this number through a
prospective 80%-power calculation, but may not reduce it. Pilot cultures are
not pooled into the confirmatory analysis after inspecting condition labels.

Real hardware defaults to one condition per culture invocation. Any crossover
design requires separate approval, randomized or Latin-square condition order,
explicit washout validation, and a carryover term in the statistical model.

## Primary endpoint

Binary official RoboSuite `NutAssemblySquare` success per episode on held-out
peg pose and friction scenarios, aggregated within culture before any
condition comparison.

The biological-learning claim requires `contact_skill` to exceed each of:

1. `baseline_only`;
2. `zero_spikes`;
3. `shuffled_spikes`;
4. `no_feedback`;
5. `yoked_feedback`.

For five preregistered comparisons, the analysis uses 99% bootstrap confidence
intervals (Bonferroni family-wise alpha <= 0.05). The lower bound of every
success difference must be greater than zero.

## Retention endpoint

The same superiority test is performed with encoder, decoder, and feedback
frozen:

- immediately after training;
- after 5 minutes;
- after 15 minutes;
- after 45 minutes.

An immediate-only effect is classified as transient state or adaptation, not
retained learning. A retained-learning claim requires the gate to pass at one
or more delayed retention phases declared before unblinding.

## Safety endpoint

Episode force safety is a co-primary non-inferiority endpoint. The lower 99%
confidence bound for `contact_skill - control` must remain above -0.05. Peak
force, 95th-percentile force, peak torque, hard stops, retracts, and time to
success are reported as secondary endpoints.

## Feedback causality

`yoked_feedback` replays an event schedule exported from a separate donor run.
It preserves event timing, polarity, amplitude, and pulse count but breaks the
relationship between the recipient culture's behavior and feedback. Missing
donor events are silent; they are never synthesized from the recipient trial.
Donor metadata must identify an independent culture, the `contact_skill`
condition, and the same protocol version.

## Exclusions

Exclusions are made without condition outcome access and must be logged:

- failed CL API recording or timestamp discontinuity;
- laboratory-defined culture health failure;
- robot emergency stop or sensor failure;
- scenario perturbation not applied;
- protocol, runtime, or channel-map hash mismatch;
- stimulation outside the approved laboratory envelope.

Task failure, low reward, no spikes, or an unfavorable biological result are
not exclusion criteria.

## Falsification

The biological-learning hypothesis is rejected for this protocol if any of the
following occurs:

- no superiority over zero or shuffled spikes;
- no superiority over no-feedback or yoked-feedback controls;
- the effect disappears when the readout is frozen;
- the force-safety non-inferiority gate fails;
- only neural response magnitude changes while official task success does not;
- apparent temporal results depend on reconstructed rather than measured spike
  timestamps.
