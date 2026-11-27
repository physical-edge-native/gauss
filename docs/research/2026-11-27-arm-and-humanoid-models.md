# Candidate robot arm and humanoid models (stages 3 and 4)

Date: 2026-11-27. Issue: #34. Research note written by an AI assistant (Claude Code) in a scheduled night run;
not a decision. The choice of model stays in the issue of each stage, as the
[scope record](../design/2026-11-27-scope-and-comparison.md) says.

## What was looked at

MuJoCo Menagerie (google-deepmind/mujoco_menagerie) at commit `eea4c7431b63da8278b0ff31638e0b0e941cec0e`
(2026-11-23), the same source as gauss's Go2 model. Each model was loaded with MuJoCo 4.0.1 (the version pinned in
`pyproject.toml`) and its `scene.xml` was stepped from its first keyframe with the keyframe's control held.
Nothing was added to gauss.

## Robot arms

| Model (folder / file) | Joints | Actuators | Mass of the bodies | Actuator type in the model | License (copyright holder) |
|---|---|---|---|---|---|
| Franka Panda (`franka_emika_panda/panda_nohand.xml`) | 7 | 7 | 16.7 kg | position servo, force limit up to 87 N·m | Apache-2.0 (Franka Emika) |
| Franka FR3 (`franka_fr3/fr3.xml`) | 7 | 7 | 18.3 kg | position servo, up to 87 N·m | Apache-2.0 (Franka Robotics) |
| UR5e (`universal_robots_ur5e/ur5e.xml`) | 6 | 6 | 21.0 kg | position servo, up to 150 N·m | BSD-3-Clause (ROS Industrial Consortium) |
| Kinova Gen3 (`kinova_gen3/gen3.xml`) | 7 | 7 | 8.2 kg | position servo, up to 105 N·m | BSD-3-Clause (Kinova inc.) |
| KUKA iiwa 14 (`kuka_iiwa_14/iiwa14.xml`) | 7 | 7 | 30.6 kg | position servo, no force limit in the model | BSD-3-Clause (Drake / Robot Locomotion Group) |
| UFACTORY xArm7 (`ufactory_xarm7/xarm7_nohand.xml`) | 7 | 7 | 11.3 kg | position servo, up to 50 N·m | BSD-3-Clause (UFACTORY) |

Mass is the sum of all body masses in the file, the fixed base link included. "Position servo" means the model's
actuators take a joint angle target with a built-in PD gain, as gauss's quadruped task does.

Notes for gauss:

- **License.** All are permissive. BSD-3-Clause models can be fetched and checked the way `gauss/models.py` does for
  A2 and Go2; Apache-2.0 (Franka) also needs its NOTICE handling checked. None is MIT OR Apache-2.0 like gauss
  itself, so the model files stay outside the repository, as now.
- **6 or 7 joints.** UR5e (6 joints) gives a chain with no redundancy; the 7-joint arms have a null space, which makes
  the end position under-determined by the command and may favour the central form (it sees the whole posture).
  This is a difference to record, not a reason to avoid 7 joints.
- **Payload not seen in training** (M3 for the arm) is added as a mass on the last body at evaluation; every model
  above has a distinct last link or flange where this can be done.
- **Contacts in the home pose.** The Gen3 keyframe already had 4 contacts at rest (between its own links or with the
  floor; not examined further); the others had none. Self-collision settings would have to be checked before
  training on Gen3.
- **Force limits.** iiwa 14 has no force limit on its actuators in this file; a fair comparison of tracking needs
  the real limits added, which changes the model and should be avoided if another candidate works as it is.

## Humanoids

| Model | Joints (besides the free base) | Actuators | Mass | Actuator type | Sensors in the file | License |
|---|---|---|---|---|---|---|
| Unitree G1 (`unitree_g1/g1.xml`) | 29 | 29 | 33.3 kg | position servo, up to 139 N·m | IMU (gyro, accelerometer) on the torso and the pelvis | BSD-3-Clause (Unitree Robotics) |
| Unitree H1 (`unitree_h1/h1.xml`) | 19 | 19 | 51.4 kg | torque motors (ctrl up to ±300 N·m) | none | BSD-3-Clause (Unitree Robotics) |

Notes for gauss:

- **G1 matches gauss's quadruped task more closely**: position actuators (gauss's task outputs joint targets) and
  two IMUs, one on the pelvis and one on the torso, which fits "an IMU on the body (and on each part where the robot
  has one)". H1 takes torques, so its task would need a PD layer written in gauss, a difference from the quadrupeds.
- **Units of split control.** G1's 29 joints split per limb as: two legs (6 each), waist (3), two arms (7 each). The
  arms' wrists add joints that matter little for walking; a 23-joint variant (wrists fixed) would shrink the
  problem. H1's 19 joints split as two legs (5 each), torso (1), two arms (4 each); its torso unit would have a
  single joint.
- **Mass.** G1 (33 kg) sits between Go2 (15 kg) and A2 (40 kg); H1 (51 kg) is above A2. The scope record states
  about 35–50 kg; both are near that range.

## Rough simulation speed (cloud session, not the training PC)

One process, 4-thread Intel Xeon 2.1 GHz, a control step of 10 physics steps of 2 ms (50 Hz), no policy, 3 s each;
measured once while the machine may have been shared, so read as an order of magnitude only.

| Model | Control steps per second |
|---|---|
| UR5e | about 19,000 |
| iiwa 14 | about 14,000 |
| Panda (with hand) | about 8,500 |
| Gen3 | about 5,700 |
| xArm7 (with hand) | about 5,500 |
| H1 | about 3,000 |
| G1 | about 2,300 |

For comparison, the earlier prototype in #11 measured about 2,700 (A2) and 3,500 (Go2) on a similar machine. The
arms are cheap to simulate; the humanoids cost about as much as the quadrupeds per step, and more environment
steps are usually needed to learn walking on two legs, which agrees with the scope record dropping the humanoid
first if the budget runs short.

## What this suggests (to be decided in the stage issues)

- Arm: **UR5e** as the first arm (fewest joints, fastest, BSD-3-Clause, no self-contact at rest), with **Panda or
  FR3** as a 7-joint follow-up if the budget allows; the choice of 6 or 7 joints changes what M1 compares and should
  be written down before training.
- Humanoid: **G1** (position actuators and IMUs like the quadruped task), possibly with the wrists fixed.

These are suggestions from reading the files; no policy has been trained on any of these models.
