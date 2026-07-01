"""Compatibility helpers for Warp runtime behavior."""

from __future__ import annotations

import builtins
import os
from typing import Any


_ORIGINAL_OPEN = builtins.open
_INSTALLED = False


def _is_text_write(mode: str) -> bool:
    return "b" not in mode and any(flag in mode for flag in ("w", "a", "x"))


def _is_cuda_source_path(file: Any) -> bool:
    try:
        path = os.fspath(file)
    except TypeError:
        return False
    return path.endswith(".cu")


def install_warp_cuda_utf8_open() -> None:
    """Make generated CUDA source writes UTF-8 even under an ASCII locale.

    Warp 1.10 writes generated ``.cu`` files with ``open(path, "w")``. If the
    process default encoding is ASCII, Unicode math symbols in Python comments
    copied into that generated source can raise ``UnicodeEncodeError``.
    """

    global _INSTALLED
    if _INSTALLED:
        return

    def open_with_cuda_utf8(
        file: Any,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        closefd: bool = True,
        opener: Any | None = None,
    ):
        if encoding is None and _is_text_write(mode) and _is_cuda_source_path(file):
            encoding = "utf-8"
        return _ORIGINAL_OPEN(file, mode, buffering, encoding, errors, newline, closefd, opener)

    builtins.open = open_with_cuda_utf8
    _INSTALLED = True
