"""The neighbours-only split form: messages arrive one control step after they are sent, message loss and delay are
reproducible from a seed and are restored exactly with the env's state, training is bit-identical for the same seed
and resumes exactly, and a split checkpoint can be evaluated."""
import numpy as np
import pytest

from gauss import sim
from gauss.env import LEG_ACT, LEG_JOINTS, LEG_OBS, N_LEGS, NEIGHBOURS, WalkEnv

FAULTY = {"form": "split", "msg_loss": 0.3, "msg_delay": 1, "msg_jitter": 1}


@pytest.fixture(scope="module")
def model():
    return sim.build_model("a2")


def _actions(n, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.normal(0.0, 0.5, N_LEGS * LEG_ACT) for _ in range(n)]


def _inboxes(env, actions):
    out = []
    for a in actions:
        obs = env.step(a)[0]
        out.append(obs[env.critic_dim:].reshape(N_LEGS, LEG_OBS)[:, -10:].copy())
    return np.array(out)


def test_messages_arrive_one_step_later(model):
    env = WalkEnv("a2", 0, 0, {"form": "split"}, model=model)
    assert env.obs_dim == env.critic_dim + N_LEGS * LEG_OBS and env.critic_dim == 49
    acts = _actions(6)
    got = _inboxes(env, acts)
    assert not got[0].any()  # nothing has arrived after the first step
    for t in range(1, len(acts)):
        sent = np.clip(acts[t - 1].reshape(N_LEGS, LEG_ACT)[:, LEG_JOINTS:], -1.0, 1.0)
        for leg in range(N_LEGS):
            for slot in range(2):
                np.testing.assert_array_equal(got[t, leg, slot * 5 + 1:slot * 5 + 5], sent[NEIGHBOURS[leg, slot]])


def test_message_loss_and_delay_reproducible(model):
    acts = _actions(40)
    a = _inboxes(WalkEnv("a2", 0, 0, FAULTY, model=model), acts)
    b = _inboxes(WalkEnv("a2", 0, 0, FAULTY, model=model), acts)
    c = _inboxes(WalkEnv("a2", 1, 0, FAULTY, model=model), acts)
    ideal = _inboxes(WalkEnv("a2", 0, 0, {"form": "split"}, model=model), acts)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c) and not np.array_equal(a, ideal)

    # Saving and restoring the state (messages in flight, the network's generator) continues exactly.
    env = WalkEnv("a2", 0, 0, FAULTY, model=model)
    _inboxes(env, acts[:20])
    state = env.get_state()
    rest = _inboxes(env, acts[20:])
    other = WalkEnv("a2", 5, 3, FAULTY, model=model)
    other.set_state(state)
    np.testing.assert_array_equal(_inboxes(other, acts[20:]), rest)


def test_central_foot_contact_option(model):
    plain = WalkEnv("a2", 0, 0, model=model)
    with_contact = WalkEnv("a2", 0, 0, {"obs_foot_contact": True}, model=model)
    o1, o2 = plain._obs(), with_contact._obs()
    assert o1.shape == (45,) and o2.shape == (49,)
    np.testing.assert_array_equal(o1, o2[:45])


torch = pytest.importorskip("torch")

from gauss import evaluate, train  # noqa: E402

SMALL = ["--robot", "a2", "--num-envs", "4", "--workers", "2", "--steps-per-env", "8", "--threads", "1", "--quiet",
         "--form", "split", "--hidden", "16,16", "--msg-loss", "0.2", "--msg-delay", "1", "--msg-jitter", "1"]


def _run(out, seed, updates, *extra):
    return train.main(["--out", str(out), "--seed", str(seed), "--updates", str(updates), *SMALL, *extra])


def test_split_training_same_seed_and_resume(tmp_path):
    import json

    a = _run(tmp_path / "a", 0, 2)
    b = _run(tmp_path / "b", 0, 2)
    assert a == b and (tmp_path / "a" / "log.jsonl").read_text().count("param_digest") == 2
    logs = [[json.loads(x)["param_digest"] for x in (tmp_path / n / "log.jsonl").read_text().splitlines()]
            for n in ("a", "b")]
    assert logs[0] == logs[1]

    straight = _run(tmp_path / "straight", 0, 3)
    _run(tmp_path / "resumed", 0, 2)
    assert _run(tmp_path / "resumed", 0, 3, "--resume") == straight

    config = json.loads((tmp_path / "a" / "config.json").read_text())
    comp = config["computation"]
    assert config["form"] == "split" and comp["controllers"] == N_LEGS
    assert comp["per_controller"]["layers"] == [LEG_OBS, 16, 16, LEG_ACT]
    assert comp["per_controller"]["params"] == sum(p.numel() for p in train.make_model(
        "split", 0, 0, 49, [16, 16], [8], 1.0).actor.parameters())

    res = evaluate.evaluate(str(tmp_path / "a" / "last.pt"), seconds=1.5)
    assert res["form"] == "split" and len(res["commands"]) == len(evaluate.COMMANDS)
    assert res["lin_vel_err"] is not None


def test_message_noise_starts_small_and_has_no_entropy_bonus():
    """#28: the message values' std starts at msg_init_noise_std and they are left out of the entropy bonus."""
    ac = train.make_model("split", 0, 0, 10, [8, 8], [8], 1.0, train.PPO["msg_init_noise_std"])
    std = ac.log_std.detach().exp()
    assert torch.allclose(std[:LEG_JOINTS], torch.ones(LEG_JOINTS))
    assert torch.allclose(std[LEG_JOINTS:], torch.full((LEG_ACT - LEG_JOINTS,), train.PPO["msg_init_noise_std"]))
    assert train.PPO["msg_in_entropy"] is False
