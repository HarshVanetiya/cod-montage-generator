"""Startup checks: FFmpeg present, Python packages installed, which encoder to use."""

from __future__ import annotations

import importlib.util
import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import Config
from .utils import MontageError, fmt_size, log, run

FFMPEG_INSTALL_HELP = """
FFmpeg was not found. It does all the video cutting, so it's required.

Install it on Windows (pick one):

  A) Easiest - open PowerShell and run:
         winget install --id Gyan.FFmpeg -e
     then CLOSE and reopen your terminal.

  B) Manual:
     1. Download "ffmpeg-release-essentials.zip" from https://www.gyan.dev/ffmpeg/builds/
     2. Extract it to C:\\ffmpeg  (so C:\\ffmpeg\\bin\\ffmpeg.exe exists)
     3. Add C:\\ffmpeg\\bin to PATH:
        Start menu -> "Edit the system environment variables" -> Environment Variables
        -> under "User variables" select Path -> Edit -> New -> C:\\ffmpeg\\bin -> OK
     4. Close and reopen your terminal.

  Or set hardware.ffmpeg_path in config.yaml to the full path of ffmpeg.exe.

Check it worked with:   ffmpeg -version
"""

# Encoder name -> ffmpeg codec, in the order "auto" tries them.
ENCODERS = {
    "nvenc": "h264_nvenc",
    "qsv": "h264_qsv",
    "amf": "h264_amf",
    "x264": "libx264",
}
ENCODER_LABELS = {
    "nvenc": "NVIDIA NVENC (GPU)",
    "qsv": "Intel Quick Sync (GPU)",
    "amf": "AMD AMF (GPU)",
    "x264": "libx264 (CPU)",
}

# Which Python packages each stage needs (import name, pip name).
STAGE_PACKAGES = {
    "ingest": [("numpy", "numpy"), ("yaml", "PyYAML"), ("tqdm", "tqdm")],
    "detect": [("cv2", "opencv-python-headless"), ("scenedetect", "scenedetect"),
               ("librosa", "librosa"), ("soundfile", "soundfile")],
    "review": [("cv2", "opencv-python-headless")],
    "music": [("librosa", "librosa"), ("soundfile", "soundfile")],
}


@dataclass
class System:
    ffmpeg: str
    ffprobe: str
    ffmpeg_version: str
    encoder: str          # key in ENCODERS
    hw_decode: bool

    @property
    def encoder_label(self) -> str:
        return ENCODER_LABELS[self.encoder]

    def decode_args(self) -> list[str]:
        """Input options for hardware decoding.

        `-hwaccel auto` picks D3D11/DXVA2 on Windows, which works on Intel,
        NVIDIA and AMD alike, and ffmpeg falls back to the CPU on its own when
        a codec isn't supported by the GPU (e.g. 10-bit HEVC on older Intel).
        """
        return ["-hwaccel", "auto"] if self.hw_decode else []

    def encode_args(self, quality: int, fast: bool = False) -> list[str]:
        """Output options for H.264 video at roughly comparable quality."""
        q = str(quality)
        if self.encoder == "nvenc":
            return ["-c:v", "h264_nvenc", "-preset", "p2" if fast else "p5",
                    "-rc", "vbr", "-cq", q, "-b:v", "0", "-pix_fmt", "yuv420p"]
        if self.encoder == "qsv":
            return ["-c:v", "h264_qsv", "-preset", "veryfast" if fast else "medium",
                    "-global_quality", q, "-pix_fmt", "nv12"]
        if self.encoder == "amf":
            return ["-c:v", "h264_amf", "-quality", "speed" if fast else "quality",
                    "-rc", "cqp", "-qp_i", q, "-qp_p", q, "-pix_fmt", "yuv420p"]
        return ["-c:v", "libx264", "-preset", "ultrafast" if fast else "veryfast",
                "-crf", q, "-pix_fmt", "yuv420p"]


def find_ffmpeg(cfg: Config) -> tuple[str, str]:
    custom = (cfg["hardware"].get("ffmpeg_path") or "").strip()
    if custom:
        ffmpeg = Path(custom)
        if ffmpeg.is_dir():
            ffmpeg = ffmpeg / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
        probe = ffmpeg.with_name(ffmpeg.name.replace("ffmpeg", "ffprobe"))
        if not ffmpeg.exists():
            raise MontageError(f"hardware.ffmpeg_path points to {ffmpeg}, which doesn't exist.\n"
                               + FFMPEG_INSTALL_HELP)
        if not probe.exists():
            raise MontageError(f"Found ffmpeg at {ffmpeg} but not ffprobe next to it ({probe}).")
        return str(ffmpeg), str(probe)

    ffmpeg, probe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not probe:
        raise MontageError(FFMPEG_INSTALL_HELP)
    return ffmpeg, probe


def _encoder_works(ffmpeg: str, codec: str) -> bool:
    """Encode half a second of black frames to check the GPU/driver really works."""
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", "color=c=black:s=640x360:r=30:d=0.5",
           "-c:v", codec]
    if codec == "h264_qsv":
        cmd += ["-pix_fmt", "nv12"]
    else:
        cmd += ["-pix_fmt", "yuv420p"]
    cmd += ["-f", "null", "-"]
    try:
        res = run(cmd, timeout=30)
    except Exception as e:  # timeout, driver hang
        log.debug("Encoder test %s failed: %s", codec, e)
        return False
    if res.returncode != 0:
        log.debug("Encoder test %s failed: %s", codec, res.stderr.strip()[-400:])
    return res.returncode == 0


def pick_encoder(ffmpeg: str, wanted: str) -> str:
    listing = run([ffmpeg, "-hide_banner", "-encoders"]).stdout
    available = [k for k, codec in ENCODERS.items() if f" {codec} " in listing]

    if wanted != "auto":
        if wanted not in available or not _encoder_works(ffmpeg, ENCODERS[wanted]):
            fallback = "x264"
            log.warning("Encoder '%s' was requested but doesn't work on this PC - using %s.",
                        wanted, ENCODER_LABELS[fallback])
            return fallback
        return wanted

    for key in ("nvenc", "qsv", "amf"):
        if key in available and _encoder_works(ffmpeg, ENCODERS[key]):
            return key
    if "x264" not in available:
        raise MontageError("Your FFmpeg build has no H.264 encoder (libx264). "
                           "Install the gyan.dev 'essentials' build - see the README.")
    return "x264"


def check_packages(stages: list[str]) -> None:
    missing = []
    for stage in stages:
        for module, pip_name in STAGE_PACKAGES.get(stage, []):
            if importlib.util.find_spec(module) is None and pip_name not in missing:
                missing.append(pip_name)
    if missing:
        raise MontageError(
            f"Missing Python packages: {', '.join(missing)}\n"
            "Install everything with:\n    pip install -r requirements.txt"
        )


def check_system(cfg: Config) -> System:
    ffmpeg, ffprobe = find_ffmpeg(cfg)
    first_line = run([ffmpeg, "-version"]).stdout.splitlines()[:1]
    version = first_line[0] if first_line else "unknown version"
    encoder = pick_encoder(ffmpeg, cfg["hardware"].get("encoder", "auto"))
    system = System(ffmpeg=ffmpeg, ffprobe=ffprobe, ffmpeg_version=version,
                    encoder=encoder, hw_decode=bool(cfg["hardware"].get("hw_decode", True)))

    log.info("FFmpeg:   %s", version)
    log.info("Encoder:  %s", system.encoder_label)
    log.info("Decoding: %s", "GPU when possible (auto)" if system.hw_decode else "CPU")
    log.debug("Python %s on %s", platform.python_version(), platform.platform())
    return system


def check_free_space(folder: Path, needed_bytes: float) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(folder).free
    log.debug("Disk: %s free, ~%s needed", fmt_size(free), fmt_size(needed_bytes))
    if free < needed_bytes:
        raise MontageError(
            f"Not enough disk space in {folder}: about {fmt_size(needed_bytes)} needed, "
            f"{fmt_size(free)} free.\nFree up space or move the project to a bigger drive."
        )
    if free < needed_bytes * 2:
        log.warning("Disk space is getting low (%s free).", fmt_size(free))
