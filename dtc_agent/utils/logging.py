"""Utilities for writing resilient console diagnostics."""

from __future__ import annotations

import os
import sys
from typing import TextIO

__all__ = ["_safe_console_write", "_emit_console_info", "_emit_console_warning"]


def _safe_console_write(message: str, stream: TextIO) -> None:
    """Write ``message`` to ``stream`` without propagating encoding errors."""

    data = (message + "\n").encode("utf-8", "replace")
    fileno = getattr(stream, "fileno", None)
    if callable(fileno):
        try:
            os.write(fileno(), data)
            return
        except OSError:
            pass
    try:
        stream.write(message + "\n")
    except Exception:
        pass


def _emit_console_info(message: str) -> None:
    _safe_console_write(message, sys.stdout)


def _emit_console_warning(message: str) -> None:
    _safe_console_write(message, sys.stderr)
