# gauss

Split control versus central control of robots, compared as a question of model
architecture, in simulation, on robots of different sizes and kinds. Part of the
[Physical-Edge Native Architecture](https://physical-edge-native.github.io/) project.

**Status: just started. Scope revised to five robots (2026-11-27); a central policy learns to walk on A2 in a short run on CPUs; the split form (per-leg controllers with messages) is in the training code, with no results yet.**

## What this repository is for

A robot is usually controlled by one policy that sees the whole body. In split control,
each part of the body (a leg, a joint or a limb) has its own small controller that sees
only its own part's sensors and the values exchanged with its neighbours, in the same way
that the Physical-Edge Native Architecture splits a computation into regions that exchange
only the values on their borders.

gauss measures how the two differ, in simulation only and without cameras, on public
MuJoCo models of a small quadruped of erg's size, quadrupeds of Unitree Go2 and A2 size,
a robot arm and a humanoid. The main question is up to which size and kind of robot, and
under which conditions, the policy's computation can be carried by a group of cheap
microcontrollers instead of one computer with a GPU. gauss also measures the computation
needed for training, command tracking and stability, behaviour under message delay, loss
and stopped controllers, and how easily each form learns. We make no claim that split
control is better than central control; the measured differences, including where split
control is worse, are the result.

The sister repository [erg](https://github.com/physical-edge-native/erg) makes the same
comparison on a cheap four-legged robot that is built and run. gauss starts at erg's size,
checks its results against erg's simulator, and then widens the size and the kind of robot.

Plan (see [the scope record](docs/design/2026-11-27-scope-and-comparison.md)):

- robots: erg's size, Go2 size (about 15 kg), A2 size (about 40 kg), a robot arm, a humanoid;
- neighbours-only split control against one central policy on every robot, with the rest
  of a ladder of architectures where the budget allows;
- simulation only, in MuJoCo, trained on CPUs.

Work is tracked in GitHub Issues.

## Running

Python 3.10 or later and git are needed. The robot models are not stored in this repository;
they are fetched from their upstream repositories at pinned commits, with their BSD-3-Clause
licenses, and checked against the hashes in `gauss/models.lock.json`.

```bash
python -m pip install -e ".[test]"
python -m gauss.models fetch      # A2 from unitreerobotics/unitree_mujoco, Go2 from MuJoCo Menagerie
python -m pytest                  # loading, standing for 10 s, determinism
python -m gauss.throughput --out throughput.json   # control steps per second on this machine
```

Training the central policy (PPO in PyTorch, CPU build) and evaluating it:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test,train]"
python -m gauss.train --out runs/a2-s0 --seed 0 --workers 4 --minutes 75   # --resume continues a run
python -m gauss.evaluate runs/a2-s0/last.pt --out eval.json                 # tracking error, falls per minute
```

`--form split` trains the neighbours-only split form instead: one controller per leg (the same
weights for the four legs, with a leg identifier), each seeing its own leg's sensors, the body's
IMU, the command and the messages from its two neighbouring legs, with the same critic as the
central form. `--hidden 64,64` sets the widths of the actor's hidden layers; `--msg-loss`,
`--msg-delay` and `--msg-jitter` add seeded message loss and delay; `--foot-contact` gives the
central policy the foot contacts too, so that both forms see the same sensors.

```bash
python -m gauss.train --out runs/a2-split-s0 --form split --hidden 64,64 --seed 0 --workers 4 --minutes 75
python -m gauss.evaluate runs/a2-split-s0/last.pt   # messages passed every step, with no loss
```

`runs/<name>/config.json` records the settings, versions, commit and machine of a run, the
parameters and multiply-adds per controller and for the whole robot, and
`log.jsonl` one line per update (env steps, wall time, CPU time, losses, reward terms).
The same seed gives identical results with the same `--threads`, torch version and kind of CPU.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
