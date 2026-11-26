"""Measure simulation throughput (control steps per second) on this machine.

    python -m gauss.throughput                 # both robots, 1, 2, 4, ... processes up to all threads
    python -m gauss.throughput --robots a2 --seconds 20 --out docs/records/throughput.json

The work per control step is the standing test with noisy targets (no policy network, no learning).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from datetime import datetime, timezone

from . import machine, sim


def _worker(robot: str, seconds: float, seed: int, barrier, queue) -> None:
    import mujoco
    import numpy as np

    r = sim.make(robot)  # build the model before the timed part
    rng = np.random.default_rng(seed)
    n = int(round(seconds / sim.CONTROL_PERIOD))
    barrier.wait()  # all processes start stepping together
    t0 = time.perf_counter()
    for _ in range(n):
        r.data.ctrl[:] = r.stand_targets + rng.normal(0.0, 0.02, size=r.model.nu)
        mujoco.mj_step(r.model, r.data, nstep=r.steps_per_control)
    queue.put((n, time.perf_counter() - t0))


def measure(robot: str, processes: int, seconds: float) -> dict:
    """Run ``processes`` copies at once; the total rate is all control steps divided by the time
    until the slowest copy finished (model building and process start-up are not counted)."""
    ctx = mp.get_context("spawn")
    barrier, queue = ctx.Barrier(processes), ctx.Queue()
    workers = [ctx.Process(target=_worker, args=(robot, seconds, seed, barrier, queue)) for seed in range(processes)]
    for w in workers:
        w.start()
    results = [queue.get() for _ in workers]
    for w in workers:
        w.join()
    steps = sum(n for n, _ in results)
    slowest = max(t for _, t in results)
    return {
        "robot": robot,
        "processes": processes,
        "control_steps": steps,
        "control_steps_per_s_total": round(steps / slowest),
        "control_steps_per_s_per_process": round(sum(n / t for n, t in results) / processes),
    }


def process_counts(limit: int) -> list[int]:
    counts, n = [], 1
    while n < limit:
        counts.append(n)
        n *= 2
    return counts + [limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robots", nargs="+", default=["a2", "go2"], choices=sorted(sim.STAND_POSE))
    parser.add_argument("--seconds", type=float, default=20.0, help="simulated seconds per process")
    parser.add_argument("--max-processes", type=int, default=os.cpu_count())
    parser.add_argument("--out", help="write the record as JSON to this file")
    args = parser.parse_args()

    record = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "machine": machine.describe(),
        "settings": {
            "physics_timestep_s": sim.PHYSICS_TIMESTEP,
            "control_period_s": sim.CONTROL_PERIOD,
            "simulated_seconds_per_process": args.seconds,
        },
        "results": [],
    }
    print(json.dumps(record["machine"], indent=2))
    for robot in args.robots:
        for p in process_counts(args.max_processes):
            row = measure(robot, p, args.seconds)
            record["results"].append(row)
            print(f"{robot}: {p:3d} processes, {row['control_steps_per_s_total']:8d} control steps/s in total, "
                  f"{row['control_steps_per_s_per_process']:6d} per process")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
            f.write("\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
