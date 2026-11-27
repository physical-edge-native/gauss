"""Parallel simulation for training: N subprocess workers, each stepping several WalkEnv instances in turn.

Does not import torch (workers stay light). Every env gets its own seed stream (seed, env_id), so the
result does not depend on how envs are split across workers.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np

INFO_KEYS = ("lin_err", "yaw_err", "fell", "time_out", "ep_len", "ep_return", "vx", "cmd_vx")


def _worker(conn, robot, seed, env_ids, config, weights):
    os.environ["OMP_NUM_THREADS"] = "1"
    from gauss import sim
    from gauss.env import WalkEnv

    model = sim.build_model(robot)  # one compiled model per process, shared by its envs
    envs = [WalkEnv(robot, seed, i, config, weights, model) for i in env_ids]
    k = len(envs)
    while True:
        cmd, arg = conn.recv()
        if cmd == "step":
            obs = np.empty((k, envs[0].obs_dim), np.float32)
            rew = np.empty(k, np.float32)
            done = np.empty(k, bool)
            info = {key: np.empty(k) for key in INFO_KEYS}
            for j, e in enumerate(envs):
                o, r, d, inf = e.step(arg[j])
                obs[j], rew[j], done[j] = o, r, d
                for key in INFO_KEYS:
                    info[key][j] = inf[key]
            conn.send((obs, rew, done, info))
        elif cmd == "dims":
            conn.send((envs[0].obs_dim, envs[0].act_dim, envs[0].critic_dim))
        elif cmd == "obs":
            conn.send(np.stack([e._obs() for e in envs]).astype(np.float32))
        elif cmd == "stats":  # reward term sums since last call, and this process's CPU time
            sums = {key: sum(e.term_sums[key] for e in envs) for key in envs[0].term_sums}
            for e in envs:
                e.term_sums = {key: 0.0 for key in e.term_sums}
            conn.send((sums, time.process_time()))
        elif cmd == "get_state":
            conn.send([e.get_state() for e in envs])
        elif cmd == "set_state":
            for e, s in zip(envs, arg):
                e.set_state(s)
            conn.send(True)
        elif cmd == "close":
            conn.close()
            return


class VecEnv:
    def __init__(self, robot: str, num_envs: int, num_workers: int, seed: int, config=None, weights=None):
        assert num_envs % num_workers == 0
        self.num_envs, self.k = num_envs, num_envs // num_workers
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for w in range(num_workers):
            a, b = ctx.Pipe()
            ids = list(range(w * self.k, (w + 1) * self.k))
            p = ctx.Process(target=_worker, args=(b, robot, seed, ids, config, weights), daemon=True)
            p.start()
            self.conns.append(a)
            self.procs.append(p)

    def _all(self, cmd, args=None):
        for i, c in enumerate(self.conns):
            c.send((cmd, None if args is None else args[i]))
        return [c.recv() for c in self.conns]

    def dims(self):
        """(observation size, action size, size of the whole-body observation at the start of the observation)."""
        return self._all("dims")[0]

    def obs(self) -> np.ndarray:
        return np.concatenate(self._all("obs"))

    def step(self, actions: np.ndarray):
        parts = self._all("step", [actions[i * self.k:(i + 1) * self.k] for i in range(len(self.conns))])
        obs = np.concatenate([p[0] for p in parts])
        rew = np.concatenate([p[1] for p in parts])
        done = np.concatenate([p[2] for p in parts])
        info = {key: np.concatenate([p[3][key] for p in parts]) for key in INFO_KEYS}
        return obs, rew, done, info

    def stats(self):
        parts = self._all("stats")
        sums = {key: sum(p[0][key] for p in parts) for key in parts[0][0]}
        return sums, sum(p[1] for p in parts)

    def get_state(self):
        return self._all("get_state")

    def set_state(self, states):
        self._all("set_state", states)

    def close(self):
        for c in self.conns:
            c.send(("close", None))
        for p in self.procs:
            p.join(timeout=5)
