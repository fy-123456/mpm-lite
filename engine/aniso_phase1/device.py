"""Device selection helpers for reproducible phase-one probes."""

from __future__ import annotations

import csv
import io
import subprocess


def query_gpu_memory() -> dict[str, dict[str, int]]:
    """Return current ``used_mib`` and ``total_mib`` for visible GPUs."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
        memory: dict[str, dict[str, int]] = {}
        for row in csv.reader(io.StringIO(result.stdout), skipinitialspace=True):
            if len(row) < 3:
                continue
            index, used, total = (int(value.strip()) for value in row[:3])
            if total > 0:
                memory[f"cuda:{index}"] = {"used_mib": used, "total_mib": total}
        return memory
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def select_lowest_memory_device(requested: str = "auto") -> str:
    """Prefer the CUDA GPU with the least used memory, or fall back to CPU."""
    if requested != "auto":
        return requested
    memory = query_gpu_memory()
    if memory:
        device = min(memory, key=lambda name: (memory[name]["used_mib"], name))
        return device
    return "cpu"
