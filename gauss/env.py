"""Walking that follows a commanded velocity on flat ground, for the central form (one policy sees every sensor)
and the neighbours-only split form (one controller per leg, exchanging messages with its two neighbours).

Builds on ``gauss.sim`` (position servos with PD gains computed inside MuJoCo, torque limits of the
models kept, physics 2 ms, control 20 ms = 50 Hz).

Observation (45 for a 12-joint robot), all in the base (body) frame:
    base angular velocity * 0.25 (3), gravity direction (3), command * (2, 2, 0.25) (3),
    joint positions - standing pose (12), joint velocities * 0.05 (12), previous action (12)
    [+ foot contact (4), 1 = touching the floor, in actuator leg order, when config "obs_foot_contact" is set]
Action (12): joint-target offsets from the standing pose, target = stand + action_scale * action.

Split form (config "form": "split"): the legs are numbered in the ring order of LEGS (FL, FR, RR, RL; the upstream
models call the hind legs "R" for rear), and each leg's neighbours are the other leg at the same end of the body and
the other leg on the same side. The observation is the whole-body observation (as above, always with foot contact;
for the critic, used only in training) followed by the four legs' own observations (LEG_OBS values each):
    own joint positions - standing pose (3), own joint velocities * 0.05 (3), own previous joint action (3),
    own foot contact (1), base angular velocity * 0.25 (3), gravity direction (3), command scaled as above (3),
    leg identifier (one-hot, 4), message from the neighbour at the same end (1 + MSG_LEARNED),
    message from the neighbour on the same side (1 + MSG_LEARNED)
The action is the four legs' actions in LEGS order, each 3 joint-target offsets (as above) and MSG_LEARNED learned
values that the leg sends, clipped to [-msg_clip, msg_clip]. A message is the sender's foot contact after the step
and its learned values. Messages sent at control step t arrive at step t + 1 + extra delay (the receiver's
observation after step t + 1 shows them), so a leg sees its neighbours one control step later than its own sensors.
Each directed message is lost with probability msg_loss; the extra delay is msg_delay + a uniform integer in
0..msg_jitter control steps. Both are drawn from the env's own network generator, seeded by (seed, env_id, 1).
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
    "form": "central",  # or "split"
    "obs_foot_contact": False,  # central form: add foot contact (4) to the observation (equal sensors to split)
    "msg_loss": 0.0,  # split form: probability that a message is lost
    "msg_delay": 0,  # split form: extra delay of every message, in control steps
    "msg_jitter": 0,  # split form: a uniform integer in 0..msg_jitter control steps added to the delay
    "msg_clip": 1.0,  # split form: the learned message values are clipped to [-msg_clip, msg_clip]
}

LEGS = ("FL", "FR", "RR", "RL")  # ring order: front-left, front-right, hind-right, hind-left
N_LEGS = 4
LEG_JOINTS = 3
MSG_LEARNED = 4  # learned values in each message
MSG_SIZE = 1 + MSG_LEARNED  # foot contact, learned values
# Each leg's two neighbours in LEGS order: slot 0 the other leg at the same end, slot 1 the other leg on the same side.
NEIGHBOURS = np.array([[i ^ 1, 3 - i] for i in range(N_LEGS)])
LEG_OBS = 3 * LEG_JOINTS + 1 + 9 + N_LEGS + 2 * MSG_SIZE  # 33
LEG_ACT = LEG_JOINTS + MSG_LEARNED  # 7

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
        # Legs: actuator indices of each leg's joints and the index of its foot in _contacts(), in LEGS order.
        prefix = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j).split("_")[0] for j in jids]
        self.leg_act = np.array([[i for i in range(self.nu) if prefix[i] == leg] for leg in LEGS])
        assert self.leg_act.shape == (N_LEGS, LEG_JOINTS), prefix
        foot_leg = [m.body(m.geom_bodyid[g]).name.split("_")[0] for g in self.foot_geoms]
        self.leg_foot = np.array([foot_leg.index(leg) for leg in LEGS])
        # Foot contact in actuator leg order (the order of the legs' joints in the whole-body observation).
        self.act_leg_foot = np.array([self.leg_foot[LEGS.index(prefix[i])] for i in range(0, self.nu, LEG_JOINTS)])
        self.form = self.cfg["form"]
        assert self.form in ("central", "split"), self.form
        self.split = self.form == "split"
        self.with_contact = self.split or bool(self.cfg["obs_foot_contact"])
        self.critic_dim = 9 + 3 * self.nu + 4 * self.with_contact  # the whole-body observation
        if self.split:
            assert self.nu == N_LEGS * LEG_JOINTS
            # The message network has its own generator, so the env's other draws do not depend on it.
            self.net_rng = np.random.default_rng(np.random.SeedSequence([seed, env_id, 1]))
            self.leg_id = np.eye(N_LEGS)
            self.obs_dim = self.critic_dim + N_LEGS * LEG_OBS
            self.act_dim = N_LEGS * LEG_ACT
        else:
            self.obs_dim = self.critic_dim
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
        imu_cmd = [
            d.qvel[3:6] * s["ang_vel"],  # free joint angular velocity is in the body frame
            -R[2, :],  # gravity (0, 0, -1) in the body frame
            self.command * np.asarray(s["command"]),
        ]
        q = d.qpos[self.qadr] - self.stand
        dq = d.qvel[self.vadr] * s["dof_vel"]
        whole = imu_cmd + [q, dq, self.last_action]
        if self.with_contact:
            whole.append(self.contact[self.act_leg_foot].astype(float))
        if not self.split:
            return np.concatenate(whole)
        contact = self.contact[self.leg_foot].astype(float)
        imu_cmd = np.concatenate(imu_cmd)
        legs = [np.concatenate([q[a], dq[a], self.last_action[a], contact[i:i + 1], imu_cmd, self.leg_id[i],
                                self.inbox_values[i].ravel()]) for i, a in enumerate(self.leg_act)]
        return np.concatenate(whole + legs)

    # ---- messages between legs (split form) ------------------------------------------------------------
    def _reset_messages(self) -> None:
        self.learned = np.zeros((N_LEGS, MSG_LEARNED))
        # What each leg last received from each of its two neighbours, and the step that message was sent at.
        self.inbox_values = np.zeros((N_LEGS, 2, MSG_SIZE))
        self.inbox_sent = np.full((N_LEGS, 2), -1)
        self.in_flight = []  # (delivery step, sent step, receiver, slot, values), in the order sent

    def _exchange_messages(self) -> None:
        """At control step t: deliver the messages due by t (sent at t - 1 or earlier), then send this step's
        messages. A receiver keeps the newest message (by sent step) from each neighbour. When a message is lost,
        or has not arrived yet, the receiver keeps the last value it received (zeros before the first), so to the
        controller a lost message looks the same as a late one."""
        t, c = self.t, self.cfg
        keep = []
        for item in self.in_flight:
            at, sent, dst, slot, values = item
            if at > t:
                keep.append(item)
            elif sent > self.inbox_sent[dst, slot]:
                self.inbox_values[dst, slot] = values
                self.inbox_sent[dst, slot] = sent
        self.in_flight = keep
        contact = self.contact[self.leg_foot]
        loss, delay, jitter = float(c["msg_loss"]), int(c["msg_delay"]), int(c["msg_jitter"])
        for dst in range(N_LEGS):
            for slot in range(2):
                src = NEIGHBOURS[dst, slot]
                # Draws only when the network is faulty, so that the ideal network uses no random numbers.
                if loss > 0.0 and self.net_rng.random() < loss:
                    continue
                extra = delay + (int(self.net_rng.integers(0, jitter + 1)) if jitter > 0 else 0)
                values = np.concatenate([[float(contact[src])], self.learned[src]])
                self.in_flight.append((t + 1 + extra, t, dst, slot, values))

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
        self.contact = self._contacts()[0]  # for the observation (last_foot is the reward's, zeros after a reset)
        self.ep_return = 0.0
        if self.split:
            self._reset_messages()
        self._sample_command()
        return self._obs()

    def step(self, action: np.ndarray):
        """Returns obs (after auto-reset when done), reward, done, info."""
        m, d, c, w = self.m, self.d, self.cfg, self.w
        a = np.asarray(action, dtype=float)
        if self.split:  # per-leg actions in LEGS order -> joint actions in actuator order, and learned values
            legs = a.reshape(N_LEGS, LEG_ACT)
            a = np.empty(self.nu)
            a[self.leg_act] = legs[:, :LEG_JOINTS]
            self.learned = np.clip(legs[:, LEG_JOINTS:], -c["msg_clip"], c["msg_clip"])
        a = np.clip(a, -c["action_clip"], c["action_clip"])
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
        self.contact = foot

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

        if self.split:
            self._exchange_messages()
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
        s = {"phys": st, "rng": self.rng.bit_generator.state, "t": self.t, "command": self.command.copy(),
             "last_action": self.last_action.copy(), "last_dq": self.last_dq.copy(),
             "air_time": self.air_time.copy(), "last_foot": self.last_foot.copy(), "contact": self.contact.copy(),
             "ep_return": self.ep_return}
        if self.split:
            s.update(net_rng=self.net_rng.bit_generator.state, learned=self.learned.copy(),
                     inbox_values=self.inbox_values.copy(), inbox_sent=self.inbox_sent.copy(),
                     in_flight=[(at, sent, dst, slot, v.copy()) for at, sent, dst, slot, v in self.in_flight])
        return s

    def set_state(self, s: dict) -> None:
        mujoco.mj_setState(self.m, self.d, s["phys"], self._SPEC)
        mujoco.mj_forward(self.m, self.d)
        self.rng.bit_generator.state = s["rng"]
        for k in ("t", "command", "last_action", "last_dq", "air_time", "last_foot", "ep_return"):
            setattr(self, k, s[k])
        # Checkpoints from before the split form have no "contact"; they never used it in the observation.
        self.contact = s["contact"] if "contact" in s else self.last_foot.copy()
        if self.split:
            self.net_rng.bit_generator.state = s["net_rng"]
            self.learned, self.inbox_values = s["learned"].copy(), s["inbox_values"].copy()
            self.inbox_sent = s["inbox_sent"].copy()
            self.in_flight = [(at, sent, dst, slot, v.copy()) for at, sent, dst, slot, v in s["in_flight"]]
