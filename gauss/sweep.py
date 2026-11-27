"""Run a set of training runs on one machine: several runs at once, each a separate `python -m gauss.train` process.

    python -m gauss.sweep sweeps/a2-central.json --out runs/a2-central
    python -m gauss.sweep sweeps/a2-central.json --out runs/a2-central --dry-run   # print the plan only

The spec is a JSON file:

    {
      "budget_env_steps": 50000000,          # per run; converted to updates of num_envs * steps_per_env steps
      "num_envs": 512, "steps_per_env": 24,  # optional, train.py's defaults
      "workers_per_run": 4,                  # simulation processes of each run (--workers)
      "threads_per_run": 4,                  # torch threads of each run's update (--threads; default: the workers)
      "max_parallel_runs": null,             # optional; default: physical cores // max(workers, threads), at least 1
      "args": ["--foot-contact"],            # optional, extra train.py options for every run
      "grid": {"robot": ["a2"], "form": ["central"], "hidden": [null, [64, 64]], "seed": [0, 1, 2, 3, 4]},
      "runs": [{"robot": "a2", "form": "split", "hidden": [48, 48], "seed": 0, "args": ["--msg-loss", "0.1"],
                "name": "a2-split-48x48-loss0.1-s0"}]
    }

"grid" (every combination, in the order robot, form, hidden, seed) and "runs" (listed one by one) may both be given.
A hidden of null means train.py's default for the form (named by its widths). Each run goes to <out>/<robot>-<form>-<hidden>-s<seed>/
(hidden written as 64x64; "name" overrides it, and must when two runs would get the same directory).

Every run is called with --resume, so running the sweep again continues it: a finished run (one with done.json for
at least the budget's updates) is skipped, an interrupted run continues from its last.pt (saved every 50 updates and
at the end), and a run that has none starts again. Raising the budget continues finished runs to the new budget.
A run that crashes affects only itself; it is marked failed and the next invocation retries it. A sweep run gives
the same result as running train.py directly with the same options (train.py's determinism: same seed, --threads,
torch version and kind of CPU).

<out>/sweep.json summarises the sweep: per run its status, updates, env steps, wall and CPU time, final training
metrics and parameter digest, with the commit, the spec and the machine. It is rewritten whenever a run ends.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from . import machine
from .train import PPO, git_commit

FINAL_KEYS = ("lin_vel_err", "yaw_rate_err", "fall_rate", "falls_per_1k_steps", "ep_len", "ep_return",
              "vx_when_cmd_ge_0.5", "std", "param_digest")


def resources(spec: dict) -> tuple[int, int]:
    """Workers and torch threads of each run; the threads default to the workers."""
    workers = int(spec.get("workers_per_run", PPO["num_workers"]))
    return workers, int(spec.get("threads_per_run", workers))


def parallel_runs(spec: dict, override: int | None = None) -> int:
    workers, threads = resources(spec)
    return override or spec.get("max_parallel_runs") or default_parallel(workers, threads)


def default_parallel(workers: int, threads: int, cores: int | None = None) -> int:
    """Runs at once: one run needs about max(workers, threads) cores (the workers step the simulation while the
    main process waits; then the main process updates the networks with its threads while the workers wait)."""
    if cores is None:
        cores = machine.describe()["physical_cores"] or os.cpu_count() or 1
    return max(1, cores // max(1, workers, threads))


def _hidden_str(hidden, form="central") -> str:
    if hidden is None:  # train.py's default for the form
        hidden = PPO["hidden"] if form == "central" else PPO["split_hidden"]
    if isinstance(hidden, str):
        hidden = [int(x) for x in hidden.split(",") if x]
    return "x".join(str(int(h)) for h in hidden)


def plan(spec: dict) -> list[dict]:
    """The runs of a spec: name, robot, form, hidden, seed and the full train.py options (without --out)."""
    n_per_update = spec.get("num_envs", PPO["num_envs"]) * spec.get("steps_per_env", PPO["steps_per_env"])
    updates = max(1, -(-int(spec["budget_env_steps"]) // n_per_update))  # rounded up to whole updates
    entries = []
    grid = spec.get("grid")
    if grid:
        for robot, form, hidden, seed in itertools.product(grid.get("robot", ["a2"]), grid.get("form", ["central"]),
                                                           grid.get("hidden", [None]), grid["seed"]):
            entries.append({"robot": robot, "form": form, "hidden": hidden, "seed": seed})
    entries += [dict(r) for r in spec.get("runs", [])]
    runs, names = [], set()
    for e in entries:
        robot, form, hidden, seed = e.get("robot", "a2"), e.get("form", "central"), e.get("hidden"), int(e["seed"])
        name = e.get("name") or f"{robot}-{form}-{_hidden_str(hidden, form)}-s{seed}"
        if name in names:
            raise SystemExit(f"two runs would go to {name}/; give them a 'name'")
        names.add(name)
        args = ["--robot", robot, "--form", form, "--seed", str(seed), "--updates", str(updates),
                "--num-envs", str(spec.get("num_envs", PPO["num_envs"])),
                "--steps-per-env", str(spec.get("steps_per_env", PPO["steps_per_env"])),
                "--workers", str(resources(spec)[0]), "--threads", str(resources(spec)[1])]
        if hidden is not None:
            args += ["--hidden", _hidden_str(hidden).replace("x", ",")]
        args += [str(a) for a in spec.get("args", [])] + [str(a) for a in e.get("args", [])]
        runs.append({"name": name, "robot": robot, "form": form, "hidden": hidden, "seed": seed,
                     "updates": updates, "train_args": args})
    return runs


def _read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _last_log(run_dir):
    last = None
    try:
        with open(os.path.join(run_dir, "log.jsonl")) as f:
            for line in f:
                if line.strip():
                    last = line
    except OSError:
        return None
    try:
        return json.loads(last) if last else None
    except ValueError:  # a line cut short by a crash
        return None


def is_finished(run_dir: str, updates: int) -> bool:
    done = _read_json(os.path.join(run_dir, "done.json"))
    return bool(done) and done.get("updates", 0) >= updates


def run_summary(run: dict, run_dir: str, status: str, returncode=None) -> dict:
    rec = _last_log(run_dir) or {}
    config = _read_json(os.path.join(run_dir, "config.json")) or {}
    return {"name": run["name"], "status": status, "returncode": returncode, "robot": run["robot"],
            "form": run["form"], "hidden": config.get("ppo", {}).get("hidden", run["hidden"]), "seed": run["seed"],
            "budget_updates": run["updates"], "updates": rec.get("update", 0), "env_steps": rec.get("env_steps", 0),
            "wall_s": rec.get("wall_s"), "cpu_s": rec.get("cpu_s"),
            "final": {k: rec.get(k) for k in FINAL_KEYS} if rec else None,
            "train_args": run["train_args"], "commit_at_start": config.get("commit"),
            "params_per_controller": config.get("computation", {}).get("per_controller", {}).get("params")}


def run_sweep(spec: dict, out: str, max_parallel: int | None = None, quiet: bool = False, poll_s: float = 0.5) -> dict:
    runs = plan(spec)
    parallel = parallel_runs(spec, max_parallel)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "sweep_spec.json"), "w") as f:
        json.dump(spec, f, indent=1)
    status = {}
    for r in runs:
        status[r["name"]] = run_summary(r, os.path.join(out, r["name"]),
                                        "finished" if is_finished(os.path.join(out, r["name"]), r["updates"])
                                        else "waiting")
    header = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": git_commit(),
              "python": sys.executable, "max_parallel_runs": parallel, "machine": machine.describe(), "spec": spec}

    def write_summary():
        counts = {}
        for s in status.values():
            counts[s["status"]] = counts.get(s["status"], 0) + 1
        doc = {**header, "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"), "counts": counts,
               "runs": [status[r["name"]] for r in runs]}
        with open(os.path.join(out, "sweep.json.tmp"), "w") as f:
            json.dump(doc, f, indent=1)
        os.replace(os.path.join(out, "sweep.json.tmp"), os.path.join(out, "sweep.json"))
        return doc

    todo = [r for r in runs if status[r["name"]]["status"] != "finished"]
    if not quiet:
        print(f"{len(runs)} runs, {len(runs) - len(todo)} finished already, {parallel} at once, "
              f"{runs[0]['updates'] if runs else 0} updates each", flush=True)
    running = {}  # name -> (Popen, run, log file)
    write_summary()
    try:
        while todo or running:
            while todo and len(running) < parallel:
                r = todo.pop(0)
                d = os.path.join(out, r["name"])
                os.makedirs(d, exist_ok=True)
                resuming = os.path.exists(os.path.join(d, "last.pt"))
                cmd = [sys.executable, "-m", "gauss.train", "--out", d, "--resume", *r["train_args"]]
                logf = open(os.path.join(d, "train.out"), "a")
                logf.write(f"# {datetime.now(timezone.utc).isoformat(timespec='seconds')} {' '.join(cmd)}\n")
                logf.flush()
                # The environment (PYTHONPATH, GAUSS_MODELS_DIR, ...) is passed on unchanged.
                p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=os.environ.copy())
                running[r["name"]] = (p, r, logf)
                status[r["name"]]["status"] = "running"
                if not quiet:
                    print(f"start  {r['name']}" + (" (resuming)" if resuming else ""), flush=True)
                write_summary()
            time.sleep(poll_s)
            for name, (p, r, logf) in list(running.items()):
                rc = p.poll()
                if rc is None:
                    continue
                logf.close()
                del running[name]
                d = os.path.join(out, name)
                rec = _last_log(d) or {}
                ok = rc == 0 and rec.get("update", 0) >= r["updates"]
                if ok:
                    with open(os.path.join(d, "done.json"), "w") as f:
                        json.dump({"updates": rec["update"], "env_steps": rec["env_steps"],
                                   "param_digest": rec.get("param_digest")}, f)
                status[name] = run_summary(r, d, "finished" if ok else "failed", rc)
                if not quiet:
                    s = status[name]
                    print(f"{'done ' if ok else 'FAIL '}  {name} updates={s['updates']} env_steps={s['env_steps']} "
                          f"wall={s['wall_s']}s cpu={s['cpu_s']}s" + ("" if ok else f" (exit {rc}, see train.out)"),
                          flush=True)
                write_summary()
    except KeyboardInterrupt:  # stop the runs; they continue from last.pt next time
        for name, (p, r, logf) in running.items():
            p.terminate()
        for name, (p, r, logf) in running.items():
            p.wait()
            logf.close()
            status[name] = run_summary(r, os.path.join(out, name), "interrupted", p.returncode)
        write_summary()
        raise
    return write_summary()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spec", help="sweep spec (JSON)")
    ap.add_argument("--out", required=True, help="sweep directory; each run goes to a subdirectory")
    ap.add_argument("--max-parallel", type=int, default=None, help="runs at once (overrides the spec)")
    ap.add_argument("--dry-run", action="store_true", help="print the runs and their train.py options only")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    with open(args.spec) as f:
        spec = json.load(f)
    if args.dry_run:
        runs = plan(spec)
        parallel = parallel_runs(spec, args.max_parallel)
        print(f"{len(runs)} runs, {parallel} at once")
        for r in runs:
            d = os.path.join(args.out, r["name"])
            state = "finished" if is_finished(d, r["updates"]) else (
                "resume" if os.path.exists(os.path.join(d, "last.pt")) else "start")
            print(f"  {state:8s} python -m gauss.train --out {d} --resume {' '.join(r['train_args'])}")
        return None
    doc = run_sweep(spec, args.out, args.max_parallel, args.quiet)
    failed = [r["name"] for r in doc["runs"] if r["status"] != "finished"]
    if failed:
        raise SystemExit(f"{len(failed)} run(s) did not finish: {', '.join(failed)}")
    return doc


if __name__ == "__main__":
    main()
