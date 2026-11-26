"""Fetch the robot models at pinned commits and check them against recorded hashes.

The model files are not committed to gauss. They are fetched from their upstream repositories at
the commits below, keeping their BSD-3-Clause license files, and every file is checked against
the SHA-256 hashes in ``models.lock.json``.

    python -m gauss.models fetch    # fetch missing models and verify them
    python -m gauss.models verify   # verify only
    python -m gauss.models lock     # (maintainers) rewrite the hashes after changing a commit
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCK = Path(__file__).resolve().parent / "models.lock.json"
DEFAULT_DIR = ROOT / "third_party" / "models"


@dataclass(frozen=True)
class Source:
    robot: str
    repository: str
    commit: str
    files: tuple[str, ...]  # paths inside the upstream repository
    xml: str  # the robot's MJCF file, inside the upstream repository


_A2_MESHES = tuple(
    f"unitree_robots/a2/meshes/{name}.STL"
    for name in ["base_link"]
    + [f"{side}_{end}_Link{i}" for side in ("left", "right") for end in ("front", "hind") for i in (1, 2, 3, 4)]
)
_GO2_ASSETS = tuple(
    f"unitree_go2/assets/{name}.obj"
    for name in [f"base_{i}" for i in range(5)]
    + ["calf_0", "calf_1", "calf_mirror_0", "calf_mirror_1", "foot", "hip_0", "hip_1"]
    + ["thigh_0", "thigh_1", "thigh_mirror_0", "thigh_mirror_1"]
)

SOURCES = {
    "a2": Source(
        robot="a2",
        repository="https://github.com/unitreerobotics/unitree_mujoco",
        commit="1eb6642e3f3fdfb7fb13a9794fd6a2dd93ea0e7d",
        files=("LICENSE", "unitree_robots/a2/a2.xml") + _A2_MESHES,
        xml="unitree_robots/a2/a2.xml",
    ),
    "go2": Source(
        robot="go2",
        repository="https://github.com/google-deepmind/mujoco_menagerie",
        commit="eea4c7431b63da8278b0ff31638e0b0e941cec0e",
        files=("unitree_go2/LICENSE", "unitree_go2/README.md", "unitree_go2/go2.xml") + _GO2_ASSETS,
        xml="unitree_go2/go2.xml",
    ),
}


def models_dir() -> Path:
    return Path(os.environ.get("GAUSS_MODELS_DIR", DEFAULT_DIR))


def robot_dir(robot: str) -> Path:
    return models_dir() / robot


def xml_path(robot: str) -> Path:
    """Path of the robot's MJCF file; the model must have been fetched."""
    return robot_dir(robot) / SOURCES[robot].xml


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, stdout=subprocess.DEVNULL)


def _fetch_source(src: Source, dest: Path) -> None:
    """Fetch only ``src.files`` at ``src.commit`` into ``dest``."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        _git("init", "-q", cwd=work)
        _git("remote", "add", "origin", src.repository, cwd=work)
        _git("config", "core.sparseCheckout", "true", cwd=work)
        # Keep the bytes exactly as upstream on every OS, so the hashes match on Windows too.
        _git("config", "core.autocrlf", "false", cwd=work)
        (work / ".git" / "info" / "sparse-checkout").write_text(
            "".join(f"/{p}\n" for p in src.files), encoding="utf-8"
        )
        _git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", src.commit, cwd=work)
        _git("-c", "advice.detachedHead=false", "checkout", "-q", "FETCH_HEAD", cwd=work)
        if dest.exists():
            shutil.rmtree(dest)
        for rel in src.files:
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(work / rel, target)


def _load_lock() -> dict:
    return json.loads(LOCK.read_text(encoding="utf-8"))


def verify(robot: str) -> list[str]:
    """Return a list of problems with the fetched files of ``robot`` (empty when all is well)."""
    entry = _load_lock()[robot]
    src = SOURCES[robot]
    problems = []
    if entry["commit"] != src.commit or entry["repository"] != src.repository:
        problems.append(f"{robot}: models.lock.json does not match the source in gauss/models.py")
    for rel, digest in entry["sha256"].items():
        path = robot_dir(robot) / rel
        if not path.is_file():
            problems.append(f"{robot}: missing {rel}")
        elif _sha256(path) != digest:
            problems.append(f"{robot}: {rel} does not match its recorded SHA-256")
    return problems


def fetch(robots: list[str] | None = None) -> None:
    for robot in robots or list(SOURCES):
        if verify(robot):
            print(f"fetching {robot} from {SOURCES[robot].repository} at {SOURCES[robot].commit[:12]}")
            _fetch_source(SOURCES[robot], robot_dir(robot))
        problems = verify(robot)
        if problems:
            raise SystemExit("\n".join(problems))
        print(f"{robot}: {len(SOURCES[robot].files)} files verified")


def lock() -> None:
    data = {}
    for robot, src in SOURCES.items():
        _fetch_source(src, robot_dir(robot))
        data[robot] = {
            "repository": src.repository,
            "commit": src.commit,
            "sha256": {rel: _sha256(robot_dir(robot) / rel) for rel in src.files},
        }
    LOCK.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {LOCK}")


def main(argv: list[str]) -> None:
    command = argv[1] if len(argv) > 1 else "fetch"
    if command == "fetch":
        fetch(argv[2:] or None)
    elif command == "verify":
        problems = [p for robot in SOURCES for p in verify(robot)]
        if problems:
            raise SystemExit("\n".join(problems))
        print("all models verified")
    elif command == "lock":
        lock()
    else:
        raise SystemExit(f"unknown command {command!r}; use fetch, verify or lock")


if __name__ == "__main__":
    main(sys.argv)
