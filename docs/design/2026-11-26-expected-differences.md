# Expected differences between split and central control, and how each is tested

Date: 2026-11-26. Issue: #12. Decided by the maintainer; this record was written with an AI assistant.

> Written for the quadrupeds of Go2 and A2 size. It is to be revised for [the revised scope](2026-11-27-scope-and-comparison.md)
> (#19) before any split form is trained.

gauss does not claim that split control (one controller per leg) is better than one central policy overall
([scope record](2026-11-25-scope.md)). It may claim a specific difference when that difference was written down here
**before** the experiment, was measured under the stated conditions, and is reported together with what split control
costs or loses. This record lists those differences, the advantages and the disadvantages alike. Where a quantity is
the same as in erg ([erg's record](https://github.com/physical-edge-native/erg/blob/main/docs/design/2026-11-25-expected-differences.md)),
the wording follows erg, so that the results at different sizes can be compared.

## Forms compared

The ladder of the scope record, on the same robot, task and training budget (environment steps), with at least 5
training seeds each:

1. no messages between legs;
2. neighbours only (the ring front-left, front-right, hind-right, hind-left);
3. all legs;
4. a body-structured central model (layers connect only neighbouring parts of the body);
5. one central policy (a multilayer perceptron), the baseline.

Robots: Unitree A2 size (main, about 40 kg) and Go2 size (about 15 kg).

## When a difference counts as shown

The difference of the means over seeds is larger than twice its standard error, and has the expected sign. Otherwise
the result is reported as "no difference shown", which is a result too. (Same rule as erg.)

## Walking quality that counts as "reached"

T1 and T2 below need a level of walking quality to measure how long each form takes to reach it. The level is set
**after the central policy on A2 has been trained and before any split form is trained**, from the central policy's
own learning curve, and is added to this record with the commit that sets it. It is a velocity-tracking error
threshold held over a stated number of consecutive evaluations. Setting it before the split forms are trained keeps
it from being chosen to favour either side.

## Expected advantages of split control

| # | Expectation | What is measured | Refuted if | Reported alongside |
|---|---|---|---|---|
| T1 | Split control reaches the set walking quality with fewer environment steps | environment steps until the velocity-tracking error first stays below the set level | the central policy needs as few or fewer steps | the final quality of each form at the end of the budget |
| T2 | Split control needs fewer CPU hours to train | CPU hours to the same point as T1, on the same machine | not fewer | wall-clock time and the machine |
| T3 | The advantage in T1 is larger at A2 size than at Go2 size | the ratio of T1 (split / central) at each size | the ratio does not fall from Go2 to A2 | both ratios with their spread |
| F1 | Keeps walking when one leg's controller stops (erg #1) | falls per minute and velocity-tracking error after one controller stops (its leg holds its last target) | split falls as often as central does when its one controller stops | performance without faults; central stops entirely when its controller stops |
| F2 | A faulty sensor affects only its own leg (erg #6) | the deviation of each leg's commands, and falls, when one leg's sensors are frozen or given noise | the other legs' commands deviate as much under split as under central | the faulty leg's own deviation |
| G1 | Neighbours only generalises better to rough ground not seen in training (found on a small simulated agent by Schilling et al. 2021) | velocity-tracking error and falls per minute on unseen rough ground | neighbours only is no better than all legs and central | performance on flat ground |
| C1 | Computation and communication per controller do not grow with the size of the robot | multiply-adds and bytes per control step per controller, Go2 size versus A2 size | they grow | the totals for the whole robot |

Training-computation claims (T1–T3) are made both per environment step and per CPU hour, since the two can disagree
(a split form can need fewer steps but more computation per step).

## Expected disadvantages, or quantities where either side may come out ahead

| Quantity | Expectation | What is measured |
|---|---|---|
| Final walking quality at a large budget | central may be equal or better (it sees everything) | velocity-tracking error and falls per minute at the end of the budget |
| Fast whole-body motion | split worse: legs see each other at least one control step late | sudden stop, sharp turn, recovery from a push |
| Message delay and loss | split degrades; central has no messages (one computer) | velocity-tracking error versus message loss 0–30 % and delay 0–2 control steps |
| A heavier body (A2) | coordination between legs matters more, so "no messages" falls further behind with size | the gap between forms 1 and 2 at each size |
| Body-structured central model (form 4) | may combine the sample efficiency of split forms with the quality of central | its place on every row above |

## Options considered

- **Keep only some of the expectations.** Fewer claims, fewer tests; but every item above can be measured from the
  training logs or by evaluating trained policies, with little extra training, so all are kept.
- **Fix the walking-quality level now.** Simpler, but without any trained policy a level would be a guess; it is
  fixed instead from the central baseline and before the split forms, which keeps it fair.
- **Claim training computation per environment step only.** Cleaner to compare across machines, but CPU hours are
  what the question from erg (erg#10) is about, so both are reported.
