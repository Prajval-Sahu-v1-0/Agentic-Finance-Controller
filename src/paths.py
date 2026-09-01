"""
paths.py — Frozen-safe path resolution for the packaged .exe
================================================================

Every other module in this project resolves paths as
``Path(__file__).parent.parent`` — fine for running from source, wrong once
the app is packaged with PyInstaller, for two different reasons:

1. Bundled read-only assets (``static/``) live inside PyInstaller's onefile
   extraction — ``sys._MEIPASS``, a fresh temp directory PyInstaller creates
   and unpacks into on every launch. Reading from there is correct.
2. The user's actual data — uploaded account sheets, generated datasets,
   saved reports — must NOT go there. ``sys._MEIPASS`` is deleted and
   recreated on every run; anything written there is gone the moment the
   app closes. This is the mistake that would make "run the .exe and it's
   ready to use" quietly lose the user's data between sessions, so it
   matters for correctness, not just for tidiness.

This module gives every writable path a home next to the actual .exe
(or the project root, when running from source) — never inside the
temp extraction directory — while bundled read-only assets still resolve
into the temp directory when frozen. Only src/api.py (and whatever it
imports for path defaults) needs this; main.py/benchrec_map.py/etc. are
dev-only CLI entry points, never part of the frozen bundle, and keep their
existing Path(__file__)-relative resolution unchanged.

Usage
-----
    from src.paths import DATA_DIR, STATIC_DIR, is_frozen
"""

from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    """True when running as a PyInstaller-frozen executable."""
    return bool(getattr(sys, "frozen", False))


def app_root() -> Path:
    """
    Directory for writable, persistent app data.

    Frozen: the directory containing the .exe itself (NOT sys._MEIPASS —
    see module docstring for why that would silently lose user data).
    Source: the project root (two levels up from this file).
    """
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def bundle_root() -> Path:
    """
    Directory for bundled read-only assets (static/).

    Frozen: sys._MEIPASS, PyInstaller onefile's per-run temp extraction
    directory. Source: same as app_root().
    """
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", app_root()))
    return app_root()


PROJECT_ROOT  = app_root()
DATA_DIR      = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "external" / "processed"
UPLOADED_DIR  = DATA_DIR / "uploaded"
STATIC_DIR    = bundle_root() / "static"
