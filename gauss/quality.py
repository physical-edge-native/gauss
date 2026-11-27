"""When a run first reaches a given quality and holds it: the env steps, CPU hours and wall time to that point.

    python -m gauss.quality runs/a2-central --max lin_vel_err=0.15 --max fall_rate=0.02 --hold 20
    python -m gauss.quality runs/a2-central --eval --max lin_vel_err=0.10 --max falls_per_minute=0 --hold 2

Used to set the quality Q of a robot from the central policy's learning curve and to measure T1 and T2 of the
expected differences (docs/design/2026-11-27-expected-differences.md). Q itself is set there, not here.

A threshold is a set of bounds on metrics (--max name=value: at most, --min name=value: at least) and a number of
consecutive points, --hold, over which every bound must be met. The point reported is the first of the held points,
so a run must also stay at the quality for --hold points after it; a run that meets the bounds only in its last
points (fewer than --hold) has not reached the quality yet.

Two sources of points:

- training (default): one point per update from <run>/log.jsonl. The metrics are those of the training rollouts
  (lin_vel_err, yaw_rate_err, fall_rate, falls_per_1k_steps, ep_len, ...), which include the exploration noise.
  fall_rate is missing in an update in which no episode ended; it is then read as 0 (no falls).
- evaluation (--eval): one point per saved checkpoint <run>/model_NNNNN.pt (every 50 updates), evaluated with
  gauss.evaluate (the mean action, fixed commands; metrics lin_vel_err, yaw_rate_err, falls_per_minute). Each
  evaluation is cached next to its checkpoint as model_NNNNN.eval.json and reused when the seconds and seed match.
  Checkpoints are evaluated in order and only until the quality is reached and held (--all evaluates every one).

The env steps, wall and CPU time of a checkpoint are those of the log line of its update; CPU time is the training
process's and its workers', not the evaluation's.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from multiprocessing import get_context

POINT_KEYS = ("update", "env_steps", "wall_s", "cpu_s")


def parse_bounds(maxs=(), mins=()) -> dict:
    """{"max": {name: value}, "min": {name: value}} from "name=value" strings."""
    out = {"max": {}, "min": {}}
    for kind, items in (("max", maxs), ("min", mins)):
        for item in items:
            name, _, value = item.partition("=")
            if not value:
                raise SystemExit(f"bound {item!r} is not name=value")
            out[kind][name.strip()] = float(value)
    return out


def _value(rec: dict, name: str):
    v = rec.get(name)
    if v is None and name == "fall_rate" and "episodes_ended" in rec:
        return 0.0  # no episode ended in this update, so none fell
    return v


def meets(rec: dict, bounds: dict) -> bool:
    for name, hi in bounds.get("max", {}).items():
        v = _value(rec, name)
        if v is None or v > hi:
            return False
    for name, lo in bounds.get("min", {}).items():
        v = _value(rec, name)
        if v is None or v < lo:
            return False
    return True


def first_held(points: list[dict], bounds: dict, hold: int = 1) -> dict | None:
    """The first point from which ``hold`` consecutive points all meet the bounds, or None."""
    hold = max(1, int(hold))
    streak = 0
    for i, p in enumerate(points):
        streak = streak + 1 if meets(p, bounds) else 0
        if streak == hold:
            return points[i - hold + 1]
    return None


def read_log(run_dir: str) -> list[dict]:
    out = []
    with open(os.path.join(run_dir, "log.jsonl")) as f:
        for line in f:
            if line.strip():
                try:
                    out.append(json.loads(line))
                except ValueError:  # a line cut short by a crash
                    break
    return out


def _result(point: dict | None, log: list[dict]) -> dict:
    last = log[-1] if log else {}
    res = {"reached": point is not None, "final_update": last.get("update"), "final_env_steps": last.get("env_steps")}
    if point is not None:
        res.update({k: point.get(k) for k in POINT_KEYS})
        res["cpu_h"] = round(point["cpu_s"] / 3600, 4) if point.get("cpu_s") is not None else None
        res["wall_h"] = round(point["wall_s"] / 3600, 4) if point.get("wall_s") is not None else None
    return res


def training_quality(run_dir: str, bounds: dict, hold: int = 1) -> dict:
    log = read_log(run_dir)
    return _result(first_held(log, bounds, hold), log)


def checkpoints(run_dir: str) -> list[tuple[int, str]]:
    out = []
    for path in glob.glob(os.path.join(run_dir, "model_*.pt")):
        m = re.fullmatch(r"model_(\d+)\.pt", os.path.basename(path))
        if m:
            out.append((int(m.group(1)), path))
    return sorted(out)


def evaluate_cached(path: str, seconds: float = 10.0, seed: int = 123) -> dict:
    """gauss.evaluate on one checkpoint, cached as <checkpoint without .pt>.eval.json."""
    cache = path[:-3] + ".eval.json"
    try:
        with open(cache) as f:
            res = json.load(f)
        if res.get("seconds_per_command") == seconds and res.get("seed") == seed:
            return res
    except (OSError, ValueError):
        pass
    from .evaluate import evaluate  # imports torch only when needed

    res = evaluate(path, seconds, seed)
    with open(cache + ".tmp", "w") as f:
        json.dump(res, f, indent=1)
    os.replace(cache + ".tmp", cache)
    return res


def eval_quality(run_dir: str, bounds: dict, hold: int = 1, seconds: float = 10.0, seed: int = 123,
                 all_checkpoints: bool = False) -> dict:
    log = read_log(run_dir)
    by_update = {rec["update"]: rec for rec in log}
    points, found = [], None
    for update, path in checkpoints(run_dir):
        if update not in by_update:  # a checkpoint past the log (should not happen); its cost is unknown
            continue
        ev = evaluate_cached(path, seconds, seed)
        points.append({**{k: by_update[update].get(k) for k in POINT_KEYS},
                       "lin_vel_err": ev["lin_vel_err"], "yaw_rate_err": ev["yaw_rate_err"],
                       "falls_per_minute": ev["falls_per_minute"]})
        found = first_held(points, bounds, hold)
        if found is not None and not all_checkpoints:
            break
    res = _result(found, log)
    res["checkpoints_evaluated"] = len(points)
    if points:
        res["last_evaluated"] = points[-1]
    return res


def run_dirs(path: str) -> list[str]:
    """``path`` itself if it is a run, else its subdirectories that are runs (a sweep directory)."""
    if os.path.exists(os.path.join(path, "log.jsonl")):
        return [path]
    return sorted(d for d in glob.glob(os.path.join(path, "*")) if os.path.exists(os.path.join(d, "log.jsonl")))


def _one(job):
    d, bounds, hold, use_eval, seconds, seed, all_ckpt = job
    if use_eval:
        res = eval_quality(d, bounds, hold, seconds, seed, all_ckpt)
    else:
        res = training_quality(d, bounds, hold)
    return {"run": os.path.basename(os.path.normpath(d)), **res}


def group_name(run: str) -> str:
    """A run's name without its seed (<robot>-<form>-<hidden>), for counting seeds that reach the quality."""
    return re.sub(r"-s\d+$", "", run)


def summarise(rows: list[dict]) -> list[dict]:
    groups = {}
    for r in rows:
        groups.setdefault(group_name(r["run"]), []).append(r)
    out = []
    for g, rs in groups.items():
        reached = [r for r in rs if r["reached"]]
        steps = [r["env_steps"] for r in reached]
        cpu = [r["cpu_h"] for r in reached if r.get("cpu_h") is not None]
        out.append({"group": g, "runs": len(rs), "reached": len(reached),
                    "env_steps_mean": _mean(steps), "env_steps_sd": _sd(steps),
                    "cpu_h_mean": _mean(cpu), "cpu_h_sd": _sd(cpu)})
    return out


def _mean(xs):
    return round(sum(xs) / len(xs), 4) if xs else None


def _sd(xs):
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return round((sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5, 4)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="a run directory or a sweep directory of runs")
    ap.add_argument("--max", action="append", default=[], metavar="NAME=VALUE", help="metric at most VALUE")
    ap.add_argument("--min", action="append", default=[], metavar="NAME=VALUE", help="metric at least VALUE")
    ap.add_argument("--hold", type=int, default=1, help="consecutive points (updates, or checkpoints with --eval)")
    ap.add_argument("--eval", action="store_true", help="evaluate saved checkpoints instead of the training log")
    ap.add_argument("--seconds", type=float, default=10.0, help="--eval: seconds per command")
    ap.add_argument("--seed", type=int, default=123, help="--eval: evaluation seed")
    ap.add_argument("--all", action="store_true", help="--eval: evaluate every checkpoint, not only up to Q")
    ap.add_argument("--jobs", type=int, default=1, help="--eval: runs evaluated at once (processes)")
    ap.add_argument("--out", help="write the table as JSON here")
    args = ap.parse_args(argv)
    bounds = parse_bounds(args.max, args.min)
    if not bounds["max"] and not bounds["min"]:
        raise SystemExit("give at least one --max or --min")
    dirs = run_dirs(args.path)
    if not dirs:
        raise SystemExit(f"no runs (directories with log.jsonl) in {args.path}")
    jobs = [(d, bounds, args.hold, args.eval, args.seconds, args.seed, args.all) for d in dirs]
    if args.jobs > 1 and args.eval:
        with get_context("spawn").Pool(args.jobs) as pool:
            rows = pool.map(_one, jobs)
    else:
        rows = [_one(j) for j in jobs]
    groups = summarise(rows)

    width = max(len(r["run"]) for r in rows)
    print(f"{'run':{width}s}  {'update':>7s} {'env steps':>12s} {'CPU h':>8s} {'wall h':>8s}  final update")
    for r in rows:
        if r["reached"]:
            print(f"{r['run']:{width}s}  {r['update']:7d} {r['env_steps']:12d} {r['cpu_h']:8.3f} "
                  f"{r['wall_h']:8.3f}  {r['final_update']}")
        else:
            print(f"{r['run']:{width}s}  {'-':>7s} {'-':>12s} {'-':>8s} {'-':>8s}  {r['final_update']}")
    for g in groups:
        print(f"{g['group']}: {g['reached']} of {g['runs']} reach it; env steps {g['env_steps_mean']} "
              f"(sd {g['env_steps_sd']}), CPU h {g['cpu_h_mean']} (sd {g['cpu_h_sd']})")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"bounds": bounds, "hold": args.hold, "source": "evaluation" if args.eval else "training",
                       "runs": rows, "groups": groups}, f, indent=1)
    return rows


if __name__ == "__main__":
    main()
