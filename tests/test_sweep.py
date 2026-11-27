"""The sweep runner and the quality tool: a tiny sweep of two runs finishes and gives the same result as train.py run
directly; running it again skips the finished runs; a run stopped between checkpoints continues to the budget and
equals a straight run bit for bit; the first point at which a quality is reached and held is found in a log."""
import json
import os
import shutil

import pytest

pytest.importorskip("torch")

import gauss  # noqa: E402
from gauss import quality, sweep, train  # noqa: E402

TINY = {"num_envs": 4, "steps_per_env": 8, "workers_per_run": 2, "threads_per_run": 1, "max_parallel_runs": 2}
PER_UPDATE = 4 * 8


@pytest.fixture(autouse=True)
def _child_imports_this_gauss(monkeypatch):
    # The runs are separate processes; make them import the gauss under test.
    root = os.path.dirname(os.path.dirname(os.path.abspath(gauss.__file__)))
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, [root, os.environ.get("PYTHONPATH")])))


def _digests(run_dir):
    with open(os.path.join(run_dir, "log.jsonl")) as f:
        return [json.loads(x)["param_digest"] for x in f]


def _direct(out, seed, updates):
    return train.main(["--out", str(out), "--robot", "a2", "--seed", str(seed), "--updates", str(updates),
                       "--num-envs", "4", "--workers", "2", "--steps-per-env", "8", "--threads", "1", "--quiet"])


def test_plan_names_and_parallel():
    runs = sweep.plan({"budget_env_steps": 100, **TINY,
                       "grid": {"robot": ["a2"], "form": ["central", "split"], "hidden": [None, [16, 16]],
                                "seed": [0, 1]}})
    assert len(runs) == 8 and runs[0]["name"] == "a2-central-256x128x64-s0" and runs[3]["name"] == "a2-central-16x16-s1"
    assert runs[0]["updates"] == 4  # 100 env steps, rounded up to whole updates of 32
    assert "--hidden" in runs[3]["train_args"] and "16,16" in runs[3]["train_args"]
    with pytest.raises(SystemExit):
        sweep.plan({"budget_env_steps": 1, "runs": [{"seed": 0}, {"seed": 0}]})
    assert sweep.default_parallel(4, 4, cores=16) == 4 and sweep.default_parallel(8, 2, cores=4) == 1


def test_sweep_skip_resume_and_same_as_train(tmp_path):
    spec = {"budget_env_steps": 2 * PER_UPDATE, **TINY, "grid": {"robot": ["a2"], "seed": [0, 1]}}
    out = tmp_path / "sweep"
    doc = sweep.run_sweep(spec, str(out), quiet=True, poll_s=0.1)
    assert [r["status"] for r in doc["runs"]] == ["finished", "finished"]
    assert all(r["env_steps"] == 2 * PER_UPDATE and r["cpu_s"] > 0 for r in doc["runs"])
    assert doc["machine"]["threads"] and json.loads((out / "sweep.json").read_text())["counts"] == {"finished": 2}
    s0, s1 = out / "a2-central-256x128x64-s0", out / "a2-central-256x128x64-s1"
    assert _digests(s0)[-1] == _direct(tmp_path / "direct", 0, 2) != _digests(s1)[-1]

    # Running it again starts nothing.
    sweep.run_sweep(spec, str(out), quiet=True, poll_s=0.1)
    starts = [line for line in (s0 / "train.out").read_text().splitlines() if line.startswith("# ")]
    assert len(starts) == 1, starts  # one header line per start of the run

    # Stop seed 0 between checkpoints: its last.pt is at update 1 while its log already has update 2. Raising the
    # budget to 3 updates continues it from update 1, and it equals a straight run of 3 updates.
    _direct(tmp_path / "one", 0, 1)
    shutil.copy(tmp_path / "one" / "last.pt", s0 / "last.pt")
    (s0 / "done.json").unlink()
    spec3 = dict(spec, budget_env_steps=3 * PER_UPDATE, grid={"robot": ["a2"], "seed": [0]})
    doc = sweep.run_sweep(spec3, str(out), quiet=True, poll_s=0.1)
    assert doc["runs"][0]["status"] == "finished" and doc["runs"][0]["updates"] == 3
    _direct(tmp_path / "straight", 0, 3)
    assert _digests(s0) == _digests(tmp_path / "straight")

    rows = [quality._one((str(d), {"max": {"lin_vel_err": 1e9}}, 2, False, 10.0, 123, False)) for d in (s0, s1)]
    assert [r["update"] for r in rows] == [1, 1]

    # Evaluation-based quality on the saved checkpoints (here the final one, under a checkpoint's name), cached.
    shutil.copy(s1 / "last.pt", s1 / "model_00002.pt")
    res = quality.eval_quality(str(s1), {"max": {"falls_per_minute": 1e9}}, 1, seconds=1.0)
    assert res["reached"] and res["update"] == 2 and res["env_steps"] == 2 * PER_UPDATE
    cached = json.loads((s1 / "model_00002.eval.json").read_text())
    assert cached["update"] == 2 and quality.evaluate_cached(str(s1 / "model_00002.pt"), 1.0) == cached


def _rec(u, lin, fall=0.0):
    return {"update": u, "env_steps": 100 * u, "wall_s": 2.0 * u, "cpu_s": 7.2 * u, "lin_vel_err": lin,
            "fall_rate": fall, "episodes_ended": 1}


def test_quality_first_held(tmp_path):
    lins = [0.5, 0.3, 0.1, 0.4, 0.1, 0.09, 0.08, 0.2]
    log = [_rec(i + 1, x) for i, x in enumerate(lins)]
    log[5]["fall_rate"] = None  # no episode ended in update 6: read as no falls
    b = quality.parse_bounds(["lin_vel_err=0.15", "fall_rate=0.05"])
    assert quality.first_held(log, b, 1)["update"] == 3
    assert quality.first_held(log, b, 3)["update"] == 5
    assert quality.first_held(log, b, 4) is None  # held for only 3 updates
    log[6]["fall_rate"] = 0.5
    assert quality.first_held(log, b, 2)["update"] == 5 and quality.first_held(log, b, 3) is None
    assert quality.first_held(log, quality.parse_bounds([], ["update=4"]), 2)["update"] == 4

    run = tmp_path / "a2-central-64x64-s0"
    run.mkdir()
    (run / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in log))
    res = quality.training_quality(str(run), quality.parse_bounds(["lin_vel_err=0.15"]), 3)
    assert res["reached"] and res["env_steps"] == 500 and res["cpu_h"] == round(36.0 / 3600, 4)
    rows = quality.main([str(tmp_path), "--max", "lin_vel_err=0.15", "--hold", "3"])
    assert rows[0]["update"] == 5
    assert quality.summarise(rows)[0] == {"group": "a2-central-64x64", "runs": 1, "reached": 1, "env_steps_mean": 500.0,
                                          "env_steps_sd": None, "cpu_h_mean": 0.01, "cpu_h_sd": None}
