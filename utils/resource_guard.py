"""Storage preflight and cache relocation for phase-one experiments."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shutil

GIB = 1024 ** 3
SYSTEM_LOW_BYTES = 2 * GIB
TOTAL_PAUSE_BYTES = 5 * GIB


class StoragePaused(RuntimeError):
    """Raised when an experiment must stop because storage is too low."""


@dataclass(frozen=True)
class StorageStatus:
    system_free_bytes: int
    data_free_bytes: int | None
    total_free_bytes: int
    system_low: bool
    paused: bool


def _existing_path(path: Path) -> Path:
    path = path.expanduser()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def inspect_storage(data_root: str | os.PathLike[str] | None = None) -> StorageStatus:
    """Inspect root and an optional data-disk path without mutating anything."""
    system_path = Path("/")
    system = shutil.disk_usage(system_path)
    data_free: int | None = None
    if data_root is not None:
        data_path = _existing_path(Path(data_root))
        # Do not count the same filesystem twice if a caller passes a root path.
        if os.stat(data_path).st_dev != os.stat(system_path).st_dev:
            data_free = int(shutil.disk_usage(data_path).free)
    total = int(system.free) + (data_free or 0)
    return StorageStatus(
        system_free_bytes=int(system.free),
        data_free_bytes=data_free,
        total_free_bytes=total,
        system_low=int(system.free) < SYSTEM_LOW_BYTES,
        paused=total < TOTAL_PAUSE_BYTES,
    )


def prepare_warp_cache(
    cache_path: str | os.PathLike[str],
    data_root: str | os.PathLike[str] | None = None,
) -> str:
    """Check storage and relocate a Warp cache if the system disk is low.

    ``data_root`` should point to a writable directory on the data disk.  It
    can also be supplied through ``MPM_LITE_DATA_ROOT``.  The normal case
    returns ``cache_path`` unchanged.  Relocation only moves the cache, which
    is reproducible and safe to regenerate.
    """
    if data_root is None:
        data_root = os.environ.get("MPM_LITE_DATA_ROOT")
    status = inspect_storage(data_root)
    if status.paused:
        raise StoragePaused(
            f"storage paused: total free space is {status.total_free_bytes / GIB:.2f} GiB, below 5 GiB"
        )
    source = Path(cache_path).expanduser()
    if not status.system_low:
        return str(source)
    if data_root is None or status.data_free_bytes is None:
        raise StoragePaused(
            "storage paused: system free space is below 2 GiB and no separate data root was provided"
        )
    data_path = Path(data_root).expanduser()
    data_path.mkdir(parents=True, exist_ok=True)
    target = data_path / "mpm-lite-warp-cache"
    if source.resolve() == target.resolve():
        return str(target)
    if source.exists():
        target.mkdir(parents=True, exist_ok=True)
        for child in tuple(source.iterdir()):
            destination = target / child.name
            if not destination.exists():
                shutil.move(str(child), str(destination))
        if not any(source.iterdir()):
            source.rmdir()
    return str(target)
