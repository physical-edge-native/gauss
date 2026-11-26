"""Walking that follows a commanded velocity on flat ground (central baseline: one policy sees every sensor).

Builds on ``gauss.sim`` (position servos with PD gains computed inside MuJoCo, torque limits of the
models kept, physics 2 ms, control 20 ms = 50 Hz).

Observation (45 for a 12-joint robot), all in the base (body) frame:
    base angular velocity * 0.25 (3), gravity direction (3), command * (2, 2, 0.25) (3),
    joint positions - standing pose (12), joint velocities * 0.05 (12), previous action (12)
Action (12): joint-target offsets from the standing pose, target = stand + action_scale * action.
Rewards follow the usual legged-RL terms (legged_gym); each term is weight * value * dt and the total
per step is clipped at 0 from below ("only positive rewards"), as legged_gym does for flat ground.

Deterministic: each env draws everything from its own numpy Generator seeded by (seed, env_id).
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import sim

CONFIG = {
    "episode_s": 20.0,
    "command_resample_s": 5.0,
    "command_ranges": {"vx": (-0.5, 1.0), "vy": (-0.3, 0.3), "yaw": (-0.5, 0.5)},
    "small_command_to_zero": 0.2,  # |(vx, vy)| below this -> (0, 0), as legged_gym
    "action_scale": 0.25,  # rad
    "action_clip": 4.0,  # in action units (= 1 rad)
    "obs_scales": {"ang_vel": 0.25, "dof_vel": 0.05, "command": (2.0, 2.0, 0.25)},
    "init_joint_noise": 0.1,  # rad, uniform
    "init_vel_noise": 0.1,  # m/s and rad/s on the base, uniform
    "tilt_limit_cos": 0.5,  # terminate when the base's up axis is more than 60 deg from vertical
    "tracking_sigma": 0.25,
    "feet_air_time_target": 0.5,  # s
}

# Weights; the reward per step is sum(weight * term * dt), then clipped at 0.
REWARD_WEIGHTS = {
    "tracking_lin_vel": 1.0,  # exp(-|v_xy,cmd - v_xy|^2 / sigma)
    "tracking_ang_vel": 0.5,  # exp(-(yaw_cmd - w_z)^2 / sigma)
    "lin_vel_z": -2.0,  # v_z^2
    "ang_vel_xy": -0.05,  # w_x^2 + w_y^2
    "torques": -1e-5,  # sum tau^2
    "dof_acc": -2.5e-7,  # sum ((dq - dq_prev) / dt)^2
    "action_rate": -0.01,  # sum (a - a_prev)^2
    "feet_air_time": 1.0,  # sum over touchdowns of (air_time - 0.5 s), only when commanded to move
    "termination": 0.0,  # 1 on a fall (kept for recording; 0 as in legged_gym flat)
}


class WalkEnv:
    def __init__(self, robot: str = "a2", seed: int = 0, env_id: int = 0, config=None, weights=None, model=None):
        """``model``: a compiled ``sim.build_model(robot)`` to share between envs in one process (the
        compiled A2 model takes about 250 MB of memory, one MjData about 1 MB)."""
        self.cfg = dict(CONFIG, **(config or {}))
        self.w = dict(REWARD_WEIGHTS, **(weights or {}))
        m = model if model is not None else sim.build_model(robot)
        self.robot, self.m, self.d = robot, m, mujoco.MjData(m)
        self.nstep = int(round(sim.CONTROL_PERIOD / sim.PHYSICS_TIMESTEP))
        self.dt = sim.CONTROL_PERIOD
        self.nu = m.nu
        jids = m.actuator_trnid[:, 0]
        self.qadr = sim.actuated_qpos_addresses(m)
        self.vadr = m.jnt_dofadr[jids].copy()
        self.stand = sim.stand_targets(m, robot)
        self.d.qpos[self.qadr] = self.stand
        self.d.qpos[2] = sim.START_HEIGHT[robot]
        self.d.ctrl[:] = self.stand
        self.max_steps = int(round(self.cfg["episode_s"] / self.dt))
        self.resample_every = int(round(self.cfg["command_resample_s"] / self.dt))
        # Body classes for contacts with the floor: feet (sphere geoms on the calves) and fall bodies.
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        calf_bodies = [b for b in range(m.nbody) if m.body(b).name.endswith("_calf")]
        self.foot_geoms = np.array(
            [g for b in calf_bodies for g in range(m.ngeom)
             if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE and m.geom_contype[g]])
        assert len(self.foot_geoms) == 4, self.foot_geoms
        fall_body = np.zeros(m.nbody, bool)
        for b in range(1, m.nbody):
            name = m.body(b).name
            fall_body[b] = not name.endswith("_calf")  # base, hips, thighs
        self.fall_geom = fall_body[m.geom_bodyid]
        self.foot_index = -np.ones(m.ngeom, int)
        self.foot_index[self.foot_geoms] = np.arange(4)
        # Nominal initial state: the standing pose after settling for 1 s.
        mujoco.mj_step(m, self.d, nstep=500)
        self.qpos0 = self.d.qpos.copy()
        self.qpos0[0:2] = 0.0
        self.qpos0[3:7] = (1, 0, 0, 0)
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, env_id]))
        self.obs_dim = 9 + 3 * self.nu
        self.act_dim = self.nu
        self.term_sums = {k: 0.0 for k in self.w}
        self.reset()

    # ---- helpers -------------------------------------------------------------------------------------
    def _rot(self) -> np.ndarray:
        mat = np.empty(9)
        mujoco.mju_quat2Mat(mat, self.d.qpos[3:7])
        return mat.reshape(3, 3)  # body -> world

    def _sample_command(self) -> None:
        cr = self.cfg["command_ranges"]
        c = np.array([self.rng.uniform(*cr["vx"]), self.rng.uniform(*cr["vy"]), self.rng.uniform(*cr["yaw"])])
        if np.hypot(c[0], c[1]) < self.cfg["small_command_to_zero"]:
            c[0:2] = 0.0
        self.command = c

    def _contacts(self):
        d = self.d
        n = d.ncon
        foot = np.zeros(4, bool)
        fall = False
        if n:
            g = d.contact.geom[:n]
            other = np.where(g[:, 0] == self.floor, g[:, 1], np.where(g[:, 1] == self.floor, g[:, 0], -1))
            other = other[other >= 0]
            fi = self.foot_index[other]
            foot[fi[fi >= 0]] = True
            fall = bool(self.fall_geom[other].any())
        return foot, fall

    def _obs(self) -> np.ndarray:
        d, s = self.d, self.cfg["obs_scales"]
        R = self._rot()
        return np.concatenate([
            d.qvel[3:6] * s["ang_vel"],  # free joint angular velocity is in the body frame
            -R[2, :],  # gravity (0, 0, -1) in the body frame
            self.command * np.asarray(s["command"]),
            d.qpos[self.qadr] - self.stand,
            d.qvel[self.vadr] * s["dof_vel"],
            self.last_action,
        ])

    # ---- API -----------------------------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        m, d, rng, c = self.m, self.d, self.rng, self.cfg
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.qpos0
        d.qpos[self.qadr] = self.stand + rng.uniform(-c["init_joint_noise"], c["init_joint_noise"], self.nu)
        d.qvel[0:6] = rng.uniform(-c["init_vel_noise"], c["init_vel_noise"], 6)
        d.ctrl[:] = self.stand
        mujoco.mj_forward(m, d)
        self.t = 0
        self.last_action = np.zeros(self.nu)
        self.last_dq = d.qvel[self.vadr].copy()
        self.air_time = np.zeros(4)
        self.last_foot = np.zeros(4, bool)
        self.ep_return = 0.0
        self._sample_command()
        return self._obs()

    def step(self, action: np.ndarray):
        """Returns obs (after auto-reset when done), reward, done, info."""
        m, d, c, w = self.m, self.d, self.cfg, self.w
        a = np.clip(np.asarray(action, dtype=float), -c["action_clip"], c["action_clip"])
        d.ctrl[:] = self.stand + c["action_scale"] * a
        mujoco.mj_step(m, d, nstep=self.nstep)
        self.t += 1

        R = self._rot()
        v_body = R.T @ d.qvel[0:3]
        w_body = d.qvel[3:6]
        dq = d.qvel[self.vadr]
        foot, fall_contact = self._contacts()
        fell = fall_contact or (R[2, 2] < c["tilt_limit_cos"])
        time_out = (self.t >= self.max_steps) and not fell

        # feet air time (legged_gym): reward at touchdown
        first_contact = foot & ~self.last_foot & (self.air_time > 0)
        self.air_time += self.dt
        moving = np.hypot(self.command[0], self.command[1]) > 0.1
        air = float(np.sum((self.air_time - c["feet_air_time_target"]) * first_contact)) * moving
        self.air_time[foot] = 0.0
        self.last_foot = foot

        lin_err2 = float(np.sum((self.command[0:2] - v_body[0:2]) ** 2))
        yaw_err2 = float((self.command[2] - w_body[2]) ** 2)
        sig = c["tracking_sigma"]
        terms = {
            "tracking_lin_vel": np.exp(-lin_err2 / sig),
            "tracking_ang_vel": np.exp(-yaw_err2 / sig),
            "lin_vel_z": v_body[2] ** 2,
            "ang_vel_xy": w_body[0] ** 2 + w_body[1] ** 2,
            "torques": float(np.sum(d.actuator_force ** 2)),
            "dof_acc": float(np.sum(((dq - self.last_dq) / self.dt) ** 2)),
            "action_rate": float(np.sum((a - self.last_action) ** 2)),
            "feet_air_time": air,
            "termination": float(fell),
        }
        rew = 0.0
        for k, v in terms.items():
            r = w[k] * v * self.dt
            self.term_sums[k] += r
            rew += r
        rew = max(rew, 0.0)
        self.ep_return += rew
        self.last_action = a
        self.last_dq = dq.copy()

        info = {"lin_err": np.sqrt(lin_err2), "yaw_err": np.sqrt(yaw_err2), "fell": fell, "time_out": time_out,
                "ep_len": self.t, "ep_return": self.ep_return, "vx": v_body[0], "cmd_vx": self.command[0]}
        done = fell or time_out
        if done:
            obs = self.reset()
        else:
            if self.t % self.resample_every == 0:
                self._sample_command()
            obs = self._obs()
        return obs, rew, done, info

    # ---- exact save / restore (for resumable, deterministic runs) ----------------------------------------
    _SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

    def get_state(self) -> dict:
        st = np.empty(mujoco.mj_stateSize(self.m, self._SPEC))
        mujoco.mj_getState(self.m, self.d, st, self._SPEC)
        return {"phys": st, "rng": self.rng.bit_generator.state, "t": self.t, "command": self.command.copy(),
                "last_action": self.last_action.copy(), "last_dq": self.last_dq.copy(),
                "air_time": self.air_time.copy(), "last_foot": self.last_foot.copy(), "ep_return": self.ep_return}

    def set_state(self, s: dict) -> None:
        mujoco.mj_setState(self.m, self.d, s["phys"], self._SPEC)
        mujoco.mj_forward(self.m, self.d)
        self.rng.bit_generator.state = s["rng"]
        for k in ("t", "command", "last_action", "last_dq", "air_time", "last_foot", "ep_return"):
            setattr(self, k, s[k])
