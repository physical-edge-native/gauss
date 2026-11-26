import pytest

from gauss import sim

MASS = {"a2": (38.0, 42.0), "go2": (14.0, 16.5)}  # kg, from the models (A2 about 40 kg, Go2 about 15 kg)


@pytest.mark.parametrize("robot", ["a2", "go2"])
def test_model_has_twelve_joints_and_expected_mass(robot):
    r = sim.make(robot)
    assert r.model.nu == 12
    low, high = MASS[robot]
    assert low <= r.mass <= high


@pytest.mark.parametrize("robot", ["a2", "go2"])
def test_stands_for_ten_seconds(robot):
    r = sim.run(robot, seconds=10.0, seed=1)
    assert sim.upright(r) > 0.98
    assert r.data.qpos[2] > 0.5 * sim.START_HEIGHT[robot]
    feet = sim.bodies_on_floor(r)
    assert feet and all(name.endswith("calf") for name in feet), feet


@pytest.mark.parametrize("robot", ["a2", "go2"])
def test_same_seed_gives_identical_state(robot):
    a = sim.run(robot, seconds=2.0, seed=7)
    b = sim.run(robot, seconds=2.0, seed=7)
    c = sim.run(robot, seconds=2.0, seed=8)
    assert sim.state_digest(a) == sim.state_digest(b)
    assert sim.state_digest(a) != sim.state_digest(c)
