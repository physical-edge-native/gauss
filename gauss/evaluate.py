"""Evaluate a trained policy on fixed commands, with the mean action (no exploration noise).

    python -m gauss.evaluate runs/a2-s0/last.pt --out eval.json

For each command the robot starts from standing and runs for ``--seconds``. After a fall the episode
restarts from standing with the same command, and the fall is counted. Reported per command and overall:
velocity-tracking error (linear, in the body's horizontal plane, and yaw rate) after the first second
of each episode, falls per minute, distance travelled, touchdowns per foot and the final base height.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from . import machine
from .env import WalkEnv
from .train import ActorCritic

# (vx m/s, vy m/s, yaw rate rad/s)
COMMANDS = [(0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.5, 0.0, 0.0), (0.8, 0.0, 0.0), (1.0, 0.0, 0.0),
            (-0.4, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, 0.0, 0.5)]
SETTLE_S = 1.0  # tracking error is measured after this much time in each episode


def evaluate(path: str, seconds: float = 10.0, seed: int = 123, robot: str | None = None) -> dict:
    s = torch.load(path, weights_only=False)
    robot = robot or s["robot"]
    torch.set_num_threads(1)
    env = WalkEnv(robot, seed, 0, config={"episode_s": seconds + 1.0, "command_resample_s": 1e9})
    ac = ActorCritic(env.obs_dim, env.act_dim, s["hp"]["hidden"], 1.0)
    ac.load_state_dict(s["model"])
    settle = int(round(SETTLE_S / env.dt))
    rows = []
    for cmd in COMMANDS:
        command = np.array(cmd, float)
        env.reset()
        env.command = command.copy()
        obs = env._obs()
        n = int(round(seconds / env.dt))
        lin, yaw, vel = [], [], []
        falls, touch = 0, np.zeros(4, int)
        x0 = env.d.qpos[0:2].copy()
        dist = 0.0
        last = env.last_foot.copy()
        for _ in range(n):
            with torch.no_grad():
                a = ac.actor(torch.from_numpy(obs).float().unsqueeze(0))[0].numpy()
            pos_before = env.d.qpos[0:2].copy()
            obs, _, done, info = env.step(a)
            if done:
                falls += int(info["fell"])
                dist += float(np.linalg.norm(pos_before - x0))
                env.command = command.copy()  # the reset drew a new command; keep ours
                obs = env._obs()
                x0 = env.d.qpos[0:2].copy()
                last = env.last_foot.copy()
                continue
            touch += env.last_foot & ~last
            last = env.last_foot.copy()
            if env.t > settle:
                lin.append(info["lin_err"])
                yaw.append(info["yaw_err"])
                R = env._rot()
                vb = R.T @ env.d.qvel[0:3]
                vel.append((vb[0], vb[1], env.d.qvel[5]))
        dist += float(np.linalg.norm(env.d.qpos[0:2] - x0))
        v = np.mean(vel, axis=0) if vel else np.full(3, np.nan)
        rows.append({
            "command": list(cmd),
            "measured": [round(float(x), 3) for x in v],
            "lin_vel_err": round(float(np.mean(lin)), 4) if lin else None,
            "yaw_rate_err": round(float(np.mean(yaw)), 4) if yaw else None,
            "falls": falls,
            "distance_m": round(dist, 2),
            "touchdowns": touch.tolist(),
            "base_height_m": round(float(env.d.qpos[2]), 3),
        })
    minutes = len(COMMANDS) * seconds / 60.0
    lin_all = [r["lin_vel_err"] for r in rows if r["lin_vel_err"] is not None]
    yaw_all = [r["yaw_rate_err"] for r in rows if r["yaw_rate_err"] is not None]
    return {
        "checkpoint": path, "robot": robot, "seed": seed, "update": s["update"], "env_steps": s["env_steps"],
        "seconds_per_command": seconds,
        "lin_vel_err": round(float(np.mean(lin_all)), 4) if lin_all else None,
        "yaw_rate_err": round(float(np.mean(yaw_all)), 4) if yaw_all else None,
        "falls_per_minute": round(sum(r["falls"] for r in rows) / minutes, 3),
        "commands": rows,
        "machine": machine.describe(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--robot", help="override the robot stored in the checkpoint")
    ap.add_argument("--out", help="write the result as JSON here")
    args = ap.parse_args(argv)
    res = evaluate(args.checkpoint, args.seconds, args.seed, args.robot)
    print(f"{res['robot']}, update {res['update']}, {res['env_steps']} env steps: "
          f"linear error {res['lin_vel_err']} m/s, yaw-rate error {res['yaw_rate_err']} rad/s, "
          f"falls per minute {res['falls_per_minute']}")
    for r in res["commands"]:
        print(" ", r)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
