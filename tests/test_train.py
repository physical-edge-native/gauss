"""A few PPO updates on a handful of envs: the same seed gives bit-identical training, another seed does
not, and stopping and resuming from a checkpoint continues exactly. Learning itself is checked by longer
runs recorded outside CI."""
import json

import pytest

torch = pytest.importorskip("torch")

from gauss import evaluate, train  # noqa: E402

SMALL = ["--robot", "a2", "--num-envs", "4", "--workers", "2", "--steps-per-env", "8", "--threads", "1", "--quiet"]


def _run(out, seed, updates, *extra):
    return train.main(["--out", str(out), "--seed", str(seed), "--updates", str(updates), *SMALL, *extra])


def _log(out):
    keys = ("loss_pi", "loss_v", "entropy", "kl", "mean_step_reward", "param_digest")
    with open(out / "log.jsonl") as f:
        return [{k: rec[k] for k in keys} for rec in map(json.loads, f)]


def test_same_seed_same_training_and_resume(tmp_path):
    a = _run(tmp_path / "a", 0, 2)
    b = _run(tmp_path / "b", 0, 2)
    c = _run(tmp_path / "c", 1, 2)
    assert a == b and _log(tmp_path / "a") == _log(tmp_path / "b")
    assert a != c

    straight = _run(tmp_path / "straight", 0, 3)
    _run(tmp_path / "resumed", 0, 2)
    resumed = _run(tmp_path / "resumed", 0, 3, "--resume")
    assert straight == resumed
    assert _log(tmp_path / "straight") == _log(tmp_path / "resumed")

    config = json.loads((tmp_path / "a" / "config.json").read_text())
    assert config["machine"]["threads"] and config["torch"] == torch.__version__
    rec = json.loads((tmp_path / "a" / "log.jsonl").read_text().splitlines()[-1])
    assert rec["env_steps"] == 2 * 4 * 8 and rec["cpu_s"] > 0 and "reward_terms_per_step" in rec


def test_evaluate_runs(tmp_path):
    _run(tmp_path / "a", 0, 1)
    res = evaluate.evaluate(str(tmp_path / "a" / "last.pt"), seconds=2.0)
    assert len(res["commands"]) == len(evaluate.COMMANDS)
    assert res["falls_per_minute"] >= 0 and res["lin_vel_err"] is not None
