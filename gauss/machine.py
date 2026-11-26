"""Describe the machine a measurement or a training run was made on."""
from __future__ import annotations

import os
import platform
import subprocess
import sys

import mujoco
import numpy
import psutil


def cpu_model() -> str:
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        elif system == "Darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
            ).stdout.strip()
        elif system == "Windows":
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def describe() -> dict:
    return {
        "cpu": cpu_model(),
        "physical_cores": psutil.cpu_count(logical=False),
        "threads": os.cpu_count(),
        "memory_gib": round(psutil.virtual_memory().total / 2**30, 1),
        "os": f"{platform.system()} {platform.release()}",
        "python": sys.version.split()[0],
        "mujoco": mujoco.__version__,
        "numpy": numpy.__version__,
    }
