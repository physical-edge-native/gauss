"""Robots on a flat floor in MuJoCo, held by joint PD control computed inside MuJoCo.

The A2 and Go2 models drive each joint with a torque motor. gauss turns each motor into a position
servo with the same torque limit, so that the PD law runs inside ``mj_step`` (one call per control
step instead of one Python loop per physics step). The torque limits of the models are kept.

erg's model already has position servos (erg's model of its 12 V class servos) and its own floor; gauss uses
them as they are, so that its simulation of erg's robot is the one erg runs.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import mujoco
import numpy as np

from . import models

# Standing pose per leg (the three joints of JOINT_PARTS[robot], in order) in radians, and PD gains
# (N·m/rad, N·m·s/rad) for the models whose torque motors gauss turns into position servos.
# erg: erg.design.stand_angles() at the pinned commit (hip abduction 0, hip flexion, knee).
STAND_POSE = {"a2": (0.0, 0.8, -1.5), "go2": (0.0, 0.9, -1.8), "erg": (0.0, 0.8504476094685183, -1.5371584045999214)}
PD_GAINS = {"a2": (200.0, 5.0), "go2": (40.0, 1.0)}
START_HEIGHT = {"a2": 0.60, "go2": 0.35, "erg": 0.157}  # erg: the torso height in erg's model
NATIVE_SERVOS = {"erg"}  # models whose own position servos and floor are used as they are

PHYSICS_TIMESTEP = 0.002  # s
CONTROL_PERIOD = 0.020  # s (50 Hz), as in erg
JOINT_PARTS = {"a2": ("hip", "thigh", "calf"), "go2": ("hip", "thigh", "calf"), "erg": ("hip_abd", "hip_flex", "knee")}
FOOT_BODY = {"a2": "calf", "go2": "calf", "erg": "shank"}  # the body that carries each foot, as a name suffix


@dataclass
class Robot:
    name: str
    model: mujoco.MjModel
    data: mujoco.MjData
    stand_targets: np.ndarray  # in actuator order
    steps_per_control: int

    @property
    def mass(self) -> float:
        return float(self.model.body_mass.sum())


def build_model(robot: str) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(models.xml_path(robot)))
    spec.option.timestep = PHYSICS_TIMESTEP
    if robot in NATIVE_SERVOS:
        return spec.compile()
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    kp, kd = PD_GAINS[robot]
    for act in spec.actuators:
        torque = np.array(act.ctrlrange, dtype=float)  # the motor's torque limits
        act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        act.gainprm[0] = kp
        act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        act.biasprm[0:3] = [0.0, -kp, -kd]
        act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = torque
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE
    return spec.compile()


def stand_targets(m: mujoco.MjModel, robot: str) -> np.ndarray:
    """The standing pose as joint targets, in actuator order."""
    targets = np.empty(m.nu)
    for i in range(m.nu):
        joint = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0])
        parts = JOINT_PARTS[robot]
        part = next(p for p in parts if f"_{p}_" in f"_{joint}_")
        targets[i] = STAND_POSE[robot][parts.index(part)]
    return targets


def actuated_qpos_addresses(m: mujoco.MjModel) -> np.ndarray:
    return m.jnt_qposadr[m.actuator_trnid[:, 0]].copy()


def make(robot: str) -> Robot:
    m = build_model(robot)
    d = mujoco.MjData(m)
    targets = stand_targets(m, robot)
    d.qpos[actuated_qpos_addresses(m)] = targets
    d.qpos[2] = START_HEIGHT[robot]
    d.ctrl[:] = targets
    mujoco.mj_forward(m, d)
    steps = int(round(CONTROL_PERIOD / PHYSICS_TIMESTEP))
    return Robot(robot, m, d, targets, steps)


def run(robot: str, seconds: float, seed: int = 0, noise: float = 0.02) -> Robot:
    """Hold the standing pose for ``seconds``, adding seeded noise to the targets each control step
    (it stands in for a policy's output)."""
    r = make(robot)
    rng = np.random.default_rng(seed)
    for _ in range(int(round(seconds / CONTROL_PERIOD))):
        r.data.ctrl[:] = r.stand_targets + rng.normal(0.0, noise, size=r.model.nu)
        mujoco.mj_step(r.model, r.data, nstep=r.steps_per_control)
    return r


def state_digest(r: Robot) -> str:
    return hashlib.sha256(r.data.qpos.tobytes() + r.data.qvel.tobytes()).hexdigest()


def upright(r: Robot) -> float:
    """z component of the base's up axis (1 = perfectly upright)."""
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, r.data.qpos[3:7])
    return float(mat[8])


def bodies_on_floor(r: Robot) -> set[str]:
    m, d = r.model, r.data
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    out = set()
    for c in d.contact[: d.ncon]:
        if floor in (c.geom1, c.geom2):
            other = c.geom2 if c.geom1 == floor else c.geom1
            out.add(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[other]))
    return out
