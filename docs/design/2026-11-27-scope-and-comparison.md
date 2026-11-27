# Scope of gauss (revised): five kinds of robots, and whether the policy fits on a group of microcontrollers

Date: 2026-11-27. Issue: #19. Decided by the maintainer; this record was written with an AI assistant.

This record replaces the robots, the order of work and the main measure of the
[scope record of 2026-11-25](2026-11-25-scope.md). Everything in that record that is not changed here stays as it is
(simulation only, MuJoCo on CPUs, PPO, at least 5 seeds, no superiority claim, message loss and delay modelled in
Python with a seed).

## Decided (2026-11-27)

| Point | Decision |
|---|---|
| Robots and order | erg's size, Go2 size, A2 size, a robot arm, a humanoid, in that order; all public MuJoCo models, no hardware |
| Unit of split control | per leg on quadrupeds, per joint on the arm, per limb on the humanoid (per joint as an option) |
| Forms that must be run | neighbours only and central, on every robot; the rest of the ladder if the budget allows |
| erg's size | erg's robot model fetched at a pinned commit; gauss's own task and training code; results checked against erg's numbers (both wait for the pena-meta decision, below) |
| "Fits on a microcontroller" | ESP32-S3 with Dyne's `mlp-actor` as the reference, as in erg (one shared definition proposed in pena-meta) |
| Robot arm | model and task chosen in the issue of the arm stage |
| Humanoid | not before the report of 2027-02-21; dropped first if the budget runs short |
| Report of 2027-02-21 | M1 and M2 for both forms at erg's size; otherwise the design of the comparison |
| Claim about GPUs | written as a measured, conditional statement, never as "better than" (wording proposed in pena-meta) |

## Purpose

The Physical-Edge Native project measures what its architecture brings to the design and control of robots, and
what it costs, on robots of different sizes and kinds. Within the project:

- [erg](https://github.com/physical-edge-native/erg) compares the two forms of control on a small four-legged robot
  that is built and run;
- [poise](https://github.com/physical-edge-native/poise) studies a soft body used as part of the computation
  (physical reservoir computing);
- **gauss** makes the comparison as a question of architecture, **in simulation only, without hardware**, and
  widens it in size and kind: from erg's size to industrial quadrupeds, a robot arm and a humanoid.

## What is compared

- **Split control (the PENA form)**: one small controller per part of the body (per leg, per joint or per limb,
  depending on the robot; see the table of robots). Each controller sees only its own part's sensors, the command,
  and the values exchanged with its **neighbours** in the body (the parts it is attached to).
- **Central control**: one policy that sees every sensor of the body.

On equal terms, for each robot:

- the same robot model, the same sensors in total (the union of the split controllers' inputs is the central policy's
  input), the same command, the same reward and the same evaluation;
- the same training budget in environment steps, and at least 5 training seeds per form;
- the same training code apart from the networks.

The ladder of the 2026-11-25 record (no messages, neighbours only, all parts, body-structured central model, central
MLP) is kept as optional steps. **Neighbours only versus central** is the pair that must be run on every robot.

No cameras. Sensors: joint angles and speeds (and loads where the model has them), an IMU on the body (and on each
part where the robot has one), foot contact for legged robots.

gauss does not claim that split control is better than central control, or than GPU-based computing. The measured
differences, including where split control is worse, are the result.

## The robots

| Order | Robot | Size | Joints | Unit of split control | Task | Model (license to be confirmed in the implementing issue) |
|---|---|---|---|---|---|---|
| 1 | Small quadruped, erg's size | about 1.5 kg | 12 | per leg (4) | walking that follows a commanded velocity | erg's `models/quadruped.xml` (MIT OR Apache-2.0, same organization), at a pinned commit; needs a pena-meta proposal (see below) |
| 2 | Go2 size | about 15 kg | 12 | per leg (4) | as above | `unitree_go2`, MuJoCo Menagerie (BSD-3-Clause); already in gauss |
| 3 | A2 size, industrial quadruped | about 40 kg | 12 | per leg (4) | as above | `unitree_robots/a2`, unitree_mujoco (BSD-3-Clause); already in gauss, central policy trained (#14) |
| 4 | Robot arm | industrial arm, 6–7 joints | 6–7 | per joint, neighbours along the chain | moving the end of the arm to a commanded position (and orientation), with a payload not seen in training | candidates: Franka Panda, Universal Robots UR5e, Kinova Gen3 (MuJoCo Menagerie) |
| 5 | Humanoid | about 35–50 kg | 19–29 | per limb (two legs, two arms, torso), neighbours in the body tree; per joint as an option | walking that follows a commanded velocity | candidates: Unitree G1 or H1 (MuJoCo Menagerie or unitree_mujoco) |

Why this order: erg's size first, so that gauss's training and evaluation can be checked against erg's simulator on
the same robot before anything new is claimed; then the quadrupeds that grow in size with the same body plan (the
size effect alone); then the arm and the humanoid, which change the kind of robot and the unit of split control.
If the CPU budget runs short, the humanoid is dropped first, then Go2.

## Measures, written down before any experiment

Every measure is reported for both forms, per robot, with the mean and the spread over seeds. Where split control is
worse, the result is written down with the same weight as where it is better.

### M1. Can the policy's computation be carried by a group of cheap microcontrollers? (the core measure)

The question: up to which size and kind of robot, and under which conditions, can the control computation run on
cheap microcontrollers (one per part) instead of one computer with a GPU (a board of the Jetson class).

Measured per controller and for the whole robot, for the **smallest network that reaches the set quality** (Q, below)
in each form:

- parameters, multiply-adds per control step, memory for weights and activations;
- multiply-adds per second at the control rate (50 Hz unless stated);
- bytes sent per control step per controller and for the whole robot;
- **fits on a microcontroller**: a controller fits when its parameters are within the limit of the reference
  runtime and its estimated time per step is within a stated share of the control period. The reference
  microcontroller and runtime are those erg runs on (ESP32-S3; Dyne's `mlp-actor`, limit 8,192 parameters for the
  full variant and 2,048 for the small one), with the time per multiply-add taken from the measurement on that board
  (Dyne #156, erg). Until that measurement exists, only multiply-adds are reported, marked as not yet converted.

Also reported for the central policy: whether it fits on one such microcontroller, and how its size grows with the
number of joints.

Conditions varied: robot (size, kind, number of joints), control rate, message loss and delay, network size.

gauss runs nothing on hardware. Times on a microcontroller are estimates from multiply-adds and the measured time per
multiply-add, and are stated as such. GPU boards are not measured; the central policy's multiply-adds per second are
reported as the load such a computer would have to carry.

Expected (to be shown or refuted):

- the size of one split controller that reaches Q grows little with the size of the robot, while the central
  policy's grows with the number of joints;
- so the split form fits on the reference microcontroller for more of the robots than the central policy does on one
  microcontroller;
- refuted for a robot if the split controller needed to reach Q does not fit, or if the central policy that reaches Q
  fits on one microcontroller as well (then the split form gives no advantage in computation there);
- reported alongside: the total computation and communication of the whole split robot, which may be larger than the
  central policy's.

### M2. Computation needed for training

CPU hours and environment steps until Q is first reached, and at the end of the budget, on the stated machine (CPU
model, threads, memory recorded with every run, including failed runs).

### M3. Following the command, and stability

- Legged robots: velocity-tracking error (forward, sideways, turning) and falls per minute, on flat ground and on
  rough ground not seen in training.
- Arm: error of the end of the arm against the commanded position (and orientation), settling time and overshoot,
  with and without a payload not seen in training.

### M4. Messages delayed or lost, and controllers that stop

- Message loss 0–30 % and delay 0–2 control steps with jitter, from a seed (split form; the central form has no
  messages and is reported under the same sensor delays).
- One controller stops (its part holds its last target) and comes back.
- One part's sensors frozen or noisy.
- Reported alongside: what the same fault does to the central form (one computer stopping stops the whole robot).

### M5. How easy it is to learn

- Environment steps (trials) until Q is first reached.
- The share of seeds that reach Q within the budget.
- Sensitivity to the reward weights and learning rate (a small, stated set of variations).

### The set quality Q

Q is set per robot **after the central policy on that robot has been trained and before the split form is trained**,
from the central policy's learning curve, and is written into this record with the commit that sets it (the rule
already decided in [the expected differences](2026-11-26-expected-differences.md)).

### When a difference counts as shown

The difference of the means over seeds is larger than twice its standard error and has the expected sign; otherwise
it is reported as "no difference shown". (Same rule as erg and the 2026-11-26 record.)

### Expected disadvantages of split control, recorded with the same weight

| Quantity | Expectation |
|---|---|
| Final quality at a large budget | the central policy may be equal or better (it sees everything) |
| Fast whole-body motion | split worse: parts see each other at least one control step late |
| Message delay and loss | split degrades; the central form has no messages |
| Total computation and communication of the whole robot | split may be larger than central |
| Training | split may need more steps (more agents sharing one reward) |
| Arm accuracy | per-joint control may be less accurate at the end of a long chain |

## Order of work and the report of 2027-02-21

| Stage | Content | Target |
|---|---|---|
| 0 | This record and the revised expected differences; pena-meta proposals | December 2026 |
| 1 | erg's size: central and neighbours-only, 5 seeds each, and the "all parts" rung, 5 seeds (#39, #42); checked against erg's simulator; M1, M2, M5 | by early February 2027 |
| 2 | Go2 and A2: the same | after stage 1 |
| 3 | Robot arm | after stage 2 |
| 4 | Humanoid | if the budget allows |

For the report of 2027-02-21: at least one size with the M1 and M2 comparison of the two forms (stage 1). If that is
not possible, the design of the comparison (this record and the expected differences) is what is reported.

**Checking against erg (stage 1).** Before stage 1 results are reported, gauss's central policy and erg's on the same
model are compared on the same commands. The tolerances are written down before the comparison, with their grounds.
If erg's own results are not yet final, the comparison is made with what erg has at that commit and is repeated later.

## Overlaps with erg and the shared rules (proposed in pena-meta)

- gauss now also covers erg's size in simulation; erg's scope record and `repos.md` describe gauss as "at the size of
  industrial robots".
- Using erg's robot model adds a pinned commit to `contracts.md`.
- "Fits on a microcontroller" uses Dyne's and erg's limits and measurements; the definition should be the same in
  erg and gauss.
- The claim about needing fewer GPUs is a public claim; `contracts.md` says no repository claims superiority over
  centralised or GPU-based approaches.

## Other options considered

- **Keep the 2026-11-25 scope (A2 and Go2 only).** Smaller, but it leaves out erg's size (no check against erg) and
  the arm and the humanoid (no change of kind).
- **Start with A2 (where the central policy already walks).** Faster to a first result at industrial size, but with
  nothing to check gauss against.
- **Split per joint everywhere.** One rule for every robot, but the units on a quadruped would no longer match erg's
  per-leg controllers.
