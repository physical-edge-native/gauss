from gauss import models


def test_fetched_models_match_the_lock():
    for robot in models.SOURCES:
        assert models.verify(robot) == []


def test_license_files_are_kept():
    assert "BSD" in (models.robot_dir("a2") / "LICENSE").read_text(encoding="utf-8")
    assert "Redistribution" in (models.robot_dir("go2") / "unitree_go2" / "LICENSE").read_text(encoding="utf-8")
