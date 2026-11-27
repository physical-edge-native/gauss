# Expected differences between split and central control on five robots, and how each is tested

Date: 2026-11-27. Issue: #23. Written by an AI assistant within the maintainer's decisions of
[the revised scope](2026-11-27-scope-and-comparison.md) (#19), for the maintainer to review. Written before any split
form is trained. Replaces [the record of 2026-11-26](2026-11-26-expected-differences.md) (#12), whose expectations are
kept below where they still apply, with the same labels.

gauss does not claim that split control is better than central control, or than GPU-based computing. A specific
difference may be claimed only if it is written down here **before** the experiment, was measured under the stated
conditions, and is reported together with what split control costs or loses. Advantages and disadvantages are listed
with the same weight. Where a quantity is the same as in erg
([erg's record](https://github.com/physical-edge-native/erg/blob/main/docs/design/2026-11-25-expected-differences.md)),
the wording follows erg.

## Forms and robots

- **Split (neighbours only)**: one controller per leg (quadrupeds), per joint (arm) or per limb (humanoid), seeing its
  own part's sensors, the command and the values from its neighbours in the body.
- **Central**: one multilayer perceptron that sees every sensor.
- **All parts** (erg's size only, decided in #39): as neighbours only, but each leg receives messages from all three
  other legs. Run at the width at which neighbours only reaches Q, with 5 seeds.
- The rest of the ladder (no messages, body-structured central model, and all parts on the other robots) where the
  budget allows.

Robots: erg's size, Go2 size, A2 size (quadrupeds, 12 joints each), a robot arm (6–7 joints), a humanoid (19–29
joints). Same model, sensors, command, reward, evaluation and training budget for both forms, at least 5 seeds each.

## Training setup shared by both forms

Added 2026-11-27 (#38) by an AI assistant. This section describes what `gauss/train.py` and `gauss/env.py` do on main
and changes no expectation. Published comparisons of split and central control differ in these choices
([prior work](../research/2026-11-27-prior-work-split-control.md)), so they are stated here.

- **One algorithm and one critic.** Both forms are trained with the same PPO. The critic is one network on the
  whole-body observation, the same in both forms, and is used only in training ("centralized training, decentralized
  execution"). Only the actors would run on the robot.
- **Split form: one actor shared by the four legs.** Each leg applies the same actor network to its own observation,
  which includes a one-hot leg identifier. The robot's log probability is the sum of the four legs', so PPO's ratio,
  entropy and KL are the robot's, as in the central form and as in erg.
- **Messages.** Each leg sends its foot contact and its learned values (clipped) to its two neighbours: the other leg at
  the same end of the body and the other leg on the same side. A message arrives one control step later than the
  receiver's own sensors, plus any extra delay. Loss and delay are drawn from a seed per environment.
- **What M1 counts.** Every controller holds its own copy of the shared actor, so the whole robot's parameters and
  multiply-adds are four times one controller's. The critic and the action standard deviations are reported
  separately as used in training only.
- **Other robots.** On the arm and the humanoid the parts differ (joints along a chain; legs, arms and torso), so
  whether controllers share weights is decided in the issue of each stage, before training.

## Rules already decided

- **Shown**: the difference of the means over seeds is larger than twice its standard error and has the expected
  sign; otherwise "no difference shown", which is reported too.
- **Quality Q**: set per robot after the central policy on that robot is trained and before the split form is
  trained, from the central policy's learning curve, and added to this record with the commit that sets it.
- **Training computation** is reported both per environment step and per CPU hour.

## The smallest network that reaches Q (used by M1)

For each robot and form, networks of two hidden layers of equal width are trained at widths 16, 32, 48, 64, 128 and
256, from the smallest up. A width **reaches Q** when at least 3 of its 5 seeds reach Q within the budget; the
smallest such width is the one reported. To save CPU time the search stops at the first width that reaches Q, and
widths above it are trained only for the central baseline. The per-leg width erg uses (48) is in the list so that
the two repositories can be compared.

## M1. Does the policy's computation fit on a group of microcontrollers? (the core measure)

"Fits" follows the revised scope: within the parameter limit of the reference runtime (Dyne's `mlp-actor` on the
ESP32-S3: 8,192 parameters, 2,048 for the small variant) and, once the time per multiply-add on that board has been
measured (Dyne #156, erg), within half of the 20 ms control period. Until then only multiply-adds are reported.

| # | Expectation | What is measured | Refuted if | Reported alongside |
|---|---|---|---|---|
| E1 | The split controller that reaches Q stays about the same size from erg's size to A2 size | parameters and multiply-adds per controller of the smallest width that reaches Q, at each quadruped size | it grows by more than one step of the width list from erg's size to A2 size | the central policy's growth over the same sizes |
| E2 | The central policy that reaches Q grows with the number of joints | its smallest width, parameters and multiply-adds per control step, against the number of joints (arm, quadrupeds, humanoid) | it does not grow from the 12-joint quadrupeds to the humanoid | the quadrupeds among themselves, where the number of joints does not change and any growth comes from size alone |
| E3 | On every robot where the split form reaches Q, each of its controllers fits on the reference microcontroller | "fits", per controller | a split controller that reaches Q does not fit | the whole robot's multiply-adds and bytes per control step, which may exceed the central policy's |
| E4 | There is a size or kind of robot from which the central policy that reaches Q no longer fits on one reference microcontroller, while the split controllers still do | "fits", for both forms on every robot | the central policy fits on every robot where it reaches Q (then the split form saves no computation on any robot here) | expected **not** to hold at erg's size: erg's central actor (6,784 parameters) already fits |
| C1 | Communication per controller does not grow with the size of the robot (2026-11-26) | bytes per control step per controller, at each size | it grows | the totals for the whole robot |

The central policy's multiply-adds per second at the control rate are reported as the load one computer would have to
carry. GPU boards are not measured.

## M2. Computation needed for training

| # | Expectation | What is measured | Refuted if | Reported alongside |
|---|---|---|---|---|
| T1 | Split reaches Q with fewer environment steps (2026-11-26) | environment steps until Q is first reached, at the width used for both forms | central needs as few or fewer | the final quality of both forms at the end of the budget; the share of seeds that reach Q and the spread of the final quality over seeds (where Schilling et al. 2021 found their difference; added 2026-11-27, #39) |
| T2 | Split needs fewer CPU hours to reach Q (2026-11-26) | CPU hours to the same point, on the same machine | not fewer | wall-clock time and the machine; computation per step of each form |
| T3 | The advantage in T1 grows with size (2026-11-26, widened) | the ratio of T1 (split / central) at erg's size, Go2 size and A2 size | the ratio does not fall from erg's size to A2 size | each ratio with its spread |

## M3. Following the command, and stability

| # | Expectation | What is measured | Refuted if | Reported alongside |
|---|---|---|---|---|
| G1 | Neighbours only generalises better to rough ground not seen in training (2026-11-26; Schilling et al. 2021) | velocity-tracking error and falls per minute on unseen rough ground, legged robots | neighbours only is no better than central | flat-ground results |
| S1 (disadvantage) | At a large budget, central is equal or better on the task | velocity-tracking error and falls per minute (legged robots); position error, settling time and overshoot of the end of the arm (arm), at the end of the budget | — (recorded either way) | — |
| S2 (disadvantage) | On the arm, per-joint control is less accurate at the end of the chain | end-of-arm position error, with and without a payload not seen in training | split is as accurate as central | the error of each joint against its own target |

## M4. Messages delayed or lost, and controllers that stop

| # | Expectation | What is measured | Refuted if | Reported alongside |
|---|---|---|---|---|
| F1 | The robot keeps working when one controller stops (2026-11-26, widened): legged robots keep walking; the arm still reaches with the remaining joints | falls per minute and tracking error (legged), reach error (arm), after one controller stops (its part holds its last target) and after it comes back | split does no better than central when its one computer stops | central stops entirely when its computer stops; results without faults |
| F2 | A faulty sensor affects mainly its own part (2026-11-26) | deviation of every part's commands when one part's sensors are frozen or noisy | the other parts deviate as much under split as under central | the faulty part's own deviation |
| D1 (disadvantage) | Split degrades with message loss and delay; central has no messages | tracking error against loss 0–30 % and delay 0–2 control steps with jitter | — (recorded either way) | central under the same sensor delays |
| D2 (disadvantage) | Split is worse in fast whole-body motion: parts see each other at least one control step late | sudden stop, sharp turn, recovery from a push (legged); fast reach (arm) | split is as good as central | — |

## M5. How easy it is to learn

| # | Expectation | What is measured | Refuted if | Reported alongside |
|---|---|---|---|---|
| L1 | On the robots with the most joints (the humanoid), more split seeds reach Q than central seeds (per-leg policies learned where a central one did not converge: Sartoretti et al. 2019) | share of seeds that reach Q within the budget | the share is not higher | the share on every robot |
| L2 | No expectation: sensitivity to the reward weights and the learning rate | Q reached or not, and steps to Q, under a small stated set of variations | — | — |

## The "all parts" rung at erg's size

Added 2026-11-27 (#42) before any stage 1 training, following the coordinator's answer in #39. It separates two
reasons why split control could fall short of central control: **splitting the computation** (each controller sees
only its own leg's sensors) and **limiting messages to neighbours** (information from the diagonal leg arrives a step
later and only through another leg). Kurin et al. (ICLR 2021) found that, in their tasks, letting every part reach every
other did better than message passing along the body's structure
([prior work](../research/2026-11-27-prior-work-split-control.md)).

| # | Expectation | What is measured | Refuted if | Reported alongside |
|---|---|---|---|---|
| A1 | On the erg-size quadruped, all parts and neighbours only differ little: the only leg added is the diagonal one, which neighbours only reaches in two control steps instead of one | final velocity-tracking error and falls per minute; environment steps to Q; share of seeds that reach Q | a difference between the two is shown in either direction on any of these | the gap between neighbours only and central on the same measures; bytes per control step per controller (3 messages instead of 2) and whether each controller still fits (M1) |

How the result is read:

- If neighbours only is shown worse than central, and all parts closes at least half of that gap, the gap is put down
  to limiting messages to neighbours.
- If all parts does not close half the gap, the gap is put down to splitting the computation.
- If no gap between neighbours only and central is shown, A1 is reported, and the comparison says nothing about the
  cause.

A1 holds only at erg's size. On robots with longer chains (the arm, the humanoid), far parts are more than two steps
apart, and the expectation may differ. It is written in the issue of each stage before that stage is trained, if
the rung is run there.

## Other disadvantages recorded with the same weight

| Quantity | Expectation |
|---|---|
| Total computation and communication of the whole robot | split may be larger than central |
| Computation per environment step in training | split may cost more per step (several controllers and messages) |
| No messages (ladder step 1) on heavier bodies | coordination through the body alone falls further behind with size (2026-11-26) |
| Body-structured central model (ladder step 4) | may combine the sample efficiency of split with the quality of central (2026-11-26) |

## Options considered

- **Report the width erg uses only, without the search.** Cheaper, but E1, E2 and E4 ask for the smallest network that
  reaches Q; a fixed width would show only whether that one width fits.
- **Count a width as reaching Q when any seed does.** Cheaper, but one lucky seed would decide; at least 3 of 5 is
  used instead.
- **Threshold "fits" by parameters only.** Possible now, but leaves out the control period; time per step is added
  once the board measurement exists.
