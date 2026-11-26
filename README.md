# gauss

Split control versus central control of robots, compared as a pure question of model
architecture, at the size of industrial robots. Part of the
[Physical-Edge Native Architecture](https://physical-edge-native.github.io/) project.

**Status: just started. A central policy learns to walk on A2 in a short run on CPUs; no split control yet.**

## What this repository is for

A legged robot is usually controlled by one policy that sees the whole body. In split
control, each leg has its own controller that sees only its own leg's sensors and the
values exchanged with other legs, in the same way that the Physical-Edge Native
Architecture splits a computation into regions that exchange only the values on their
borders.

gauss measures how the two differ on four-legged robots of industrial size, in
simulation: walking performance, the computation needed for training, computation and
communication per controller, and behaviour under message loss, delay and faults.
We make no claim that split control is better than central control; the comparison is
the result.

The sister repository [erg](https://github.com/physical-edge-native/erg) makes the same
comparison on a cheap four-legged robot built from hobby parts. Differences that matter at
industrial size may not show on a small, cheap robot, so the architecture comparison is
kept here, separately.

Plan (see [the scope record](docs/design/2026-11-25-scope.md)):

- robots of the size of the Unitree A2 (about 40 kg, the main robot) and Unitree Go2
  (about 15 kg), both with four legs of three joints, like erg's robot;
- a ladder of architectures, from one controller per leg without messages, through
  neighbours-only and all-to-all messages and a body-structured central model, to one
  central policy;
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

`runs/<name>/config.json` records the settings, versions, commit and machine of a run, and
`log.jsonl` one line per update (env steps, wall time, CPU time, losses, reward terms).
The same seed gives identical results with the same `--threads`, torch version and kind of CPU.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
