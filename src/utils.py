"""Shared helpers: logging, JSON cache files, running FFmpeg with a progress bar."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Callable

from tqdm import tqdm

log = logging.getLogger("montage")


class MontageError(Exception):
    """An error with a message meant for the user (no traceback needed)."""


# ---------------------------------------------------------------- logging

class _TqdmHandler(logging.Handler):
    """Console handler that doesn't break tqdm progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:  # pragma: no cover
            self.handleError(record)


def setup_logging(work_dir: Path, verbose: bool = False) -> Path:
    """Log to the console and append to work/run.log."""
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "run.log"

    log.setLevel(logging.DEBUG)
    log.handlers.clear()

    console = _TqdmHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(console)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
    log.addHandler(file_handler)

    log.debug("=" * 60)
    log.debug("Run started")
    return log_path


# ---------------------------------------------------------------- json

def read_json(path: Path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path: Path, data) -> None:
    """Write atomically so an interrupted run never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


# ---------------------------------------------------------------- formatting

def fmt_time(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.0f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"


# ---------------------------------------------------------------- subprocess

def run(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run a command, capture output, never raise on a non-zero exit."""
    log.debug("RUN %s", " ".join(cmd))
    kwargs = {}
    if os.name == "nt":  # don't flash console windows on Windows
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=timeout, **kwargs,
    )


def run_ffmpeg_progress(
    cmd: list[str],
    on_progress: Callable[[float], None],
) -> tuple[int, str]:
    """Run ffmpeg with `-progress pipe:1` and report output time (seconds) as it goes.

    Returns (exit code, stderr tail). stderr goes to a temp file so a chatty
    ffmpeg can never fill a pipe and stall.
    """
    import tempfile

    full = cmd[:1] + ["-hide_banner", "-nostats", "-progress", "pipe:1"] + cmd[1:]
    log.debug("RUN %s", " ".join(full))
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as err:
        proc = subprocess.Popen(
            full, stdout=subprocess.PIPE, stderr=err, stdin=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace", **kwargs,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            key, _, value = line.strip().partition("=")
            # out_time_us and (despite the name) out_time_ms are both microseconds
            if key in ("out_time_us", "out_time_ms") and value.isdigit():
                on_progress(int(value) / 1_000_000)
        code = proc.wait()
        err.seek(0)
        tail = err.read()[-3000:]
    return code, tail


class StageTimer:
    """Logs how long a stage took."""

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        self.start = time.monotonic()
        log.info("\n=== %s ===", self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            log.info("%s finished in %s", self.name, fmt_time(time.monotonic() - self.start))

