# Which network widths fit the reference microcontroller (M1)

Date: 2026-11-28. Issue: #44. Research note written by an AI assistant (Claude Code) in a scheduled night run.
Nothing was trained for it. It is arithmetic on the network sizes, to show in advance where M1 can and cannot tell
the two forms apart. It changes no design record.

## The limit

From the [expected differences](../design/2026-11-27-expected-differences.md): a controller fits when its
parameters are within the limit of Dyne's `mlp-actor` on the ESP32-S3. The limit is 8,192 parameters for the full
variant and 2,048 for the small one. Once the time per multiply-add has been measured on that board (Dyne #156,
erg), the time per step must also be within half of the 20 ms control period.

Parameters and multiply-adds below use the formula in `gauss/train.py` (`mlp_size`): weights plus biases, for two
hidden layers of the same width, as in the width search.

## Quadrupeds (exact sizes from the code)

Sizes from `gauss/env.py`:

- one leg's split controller: 33 inputs, 7 outputs (3 joint targets and 4 learned message values);
- the central policy: 45 inputs (48 with foot contact), 12 outputs.

| Width | Split controller, parameters (multiply-adds) | Central, 45 inputs | Central, 48 inputs (foot contact) |
|---|---|---|---|
| 16 | 935 (896) | 1,212 | 1,260 |
| 32 | 2,375 (2,304) | 2,924 | 3,020 |
| 48 | 4,327 (4,224) | 5,148 | 5,292 |
| 64 | 6,791 (6,656) | 7,884 | 8,076 |
| 128 | 21,767 (21,504) | 23,948 | 24,332 |
| 256 | 76,295 (75,776) | 80,652 | 81,420 |

The robot size (erg's size, Go2, A2) does not change these numbers, because the observation and action sizes are the same on every 12-joint quadruped.
Only the width needed to reach Q can differ.

What follows:

- **Both forms fit up to width 64 and neither fits from width 128.** The central policy at width 64 with foot contact
  (8,076) is within 116 parameters of the limit. At the same width, a split controller is about 15 % smaller.
- **So on the quadrupeds, M1 separates the forms only through the width each needs to reach Q.** E4 ("the central
  policy stops fitting while the split controllers still fit") can hold on a quadruped only if central needs
  width 128 or more and split reaches Q at 64 or less.
- **The small variant (2,048) fits only width 16**, for either form.
- The default central network used so far (256, 128, 64; 53,708 parameters) is far over the limit. It is a training
  baseline, not a candidate for one microcontroller.

## Robot arm and humanoid (rough estimates, not from code)

The arm and humanoid tasks do not exist yet. The inputs below follow the quadruped's pattern: position, speed and
previous action per joint, the base IMU where the robot has a floating base, the command, a one-hot part
identifier, and two messages of 5 values. They are assumptions, to be replaced when the tasks are written.

| Robot and form | Inputs → outputs (assumed) | Width 32 | Width 48 | Width 64 | Width 128 |
|---|---|---|---|---|---|
| Arm, 7 joints, central | 24 → 7 | 2,087 | 3,895 | 6,215 | 20,615 |
| Arm, per joint | 23 → 5 | 1,989 | 3,749 | 6,021 | 20,229 |
| Humanoid G1 (29 joints), central | 98 → 29 | 5,181 | 8,525 | 12,381 | 32,925 |
| Humanoid G1, per limb (an arm, 7 joints) | 48 → 11 | 2,987 | 5,243 | 8,011 | 24,203 |
| Humanoid G1, per limb (a leg, 6 joints) | 46 → 10 | 2,890 | 5,098 | 7,818 | 23,818 |
| Humanoid H1 (19 joints), central | 68 → 19 | 3,891 | 6,595 | 9,811 | 27,795 |

What follows:

- **On the arm, splitting per joint saves almost nothing in parameters.** A per-joint controller has about as many
  inputs as the whole arm's central policy, because the central policy's input is small for 6–7 joints. So
  M1 cannot favour the split form on the arm, whatever the widths turn out to be. The arm's comparison rests on
  M3 and M4.
- **On the humanoid, the central policy stops fitting one width step earlier.** At width 48, the central G1 policy
  (8,525) is over the limit, while per-limb controllers (about 5,100–5,200) fit. This is the one robot where E4 can
  hold at equal widths. It agrees with the expectation in E2 that the central policy grows with the number of joints.

## Time per step (pending the measurement)

At 50 Hz, a split controller at width 64 needs 6,656 multiply-adds per step, about 0.33 million per second. Whether
that is within half of the 20 ms period depends on the time per multiply-add measured on the board. The ESP32-S3's
clock (up to 240 MHz) leaves a wide margin even at tens of cycles per multiply-add, but this is not counted as a
result until Dyne #156 / erg report the measurement, as the record says.

## Points for a later issue (not changed here)

- E4 could say explicitly that, on the quadrupeds, it depends only on the widths reached. A refutation is likely
  if both forms reach Q at width 64 or below.
- The arm could be marked as a robot where M1 is not expected to separate the forms, so that its result is not read
  as a finding about split control.
- The foot-contact input brings the central policy at width 64 to within 116 parameters of the limit. A change to
  the central inputs could move it over. The record could say that "fits" is checked with the inputs actually used.
