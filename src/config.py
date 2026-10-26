"""Load config.yaml, apply env-variable and command-line overrides, validate."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .utils import MontageError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Keys every config.yaml must have. A clear error beats a KeyError deep in a stage.
REQUIRED = [
    "game", "platform", "style", "paths", "hardware", "ingest", "detection",
    "regions", "music", "montage", "platforms", "styles", "render",
]

DETECTION_KEYS = [
    "weight_audio", "weight_kills", "multikill_min_kills", "multikill_window", "multikill_bonus",
    "lead_in", "lead_out", "merge_gap", "max_clip_len", "max_candidates", "kill_z", "hitmarker_z",
    "event_debounce", "audio_peak", "static_motion", "static_min_seconds", "dark_level",
    "uniform_std", "desat_ratio", "scene_threshold",
]

ENV_OVERRIDES = {
    "MONTAGE_PLATFORM": "platform",
    "MONTAGE_STYLE": "style",
    "MONTAGE_GAME": "game",
    "MONTAGE_LENGTH": "length",
}


@dataclass
class Config:
    raw: dict[str, Any]
    root: Path
    # Run options (not stored in config.yaml)
    auto: bool = False
    preview: bool = False
    force: bool = False
    music_choice: str | None = None
    sources: dict[str, str] = field(default_factory=dict)  # where each override came from

    # ---- shortcuts
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    @property
    def game(self) -> str:
        return self.raw["game"]

    @property
    def platform_name(self) -> str:
        return self.raw["platform"]

    @property
    def style_name(self) -> str:
        return self.raw["style"]

    @property
    def platform(self) -> dict:
        return self.raw["platforms"][self.platform_name]

    @property
    def style(self) -> dict:
        return self.raw["styles"][self.style_name]

    @property
    def regions(self) -> dict:
        return self.raw["regions"][self.game]

    @property
    def length(self) -> float | str:
        """'auto' or a number of seconds."""
        return self.raw["montage"]["length"]

    def path(self, name: str) -> Path:
        return (self.root / self.raw["paths"][name]).resolve()

    @property
    def work(self) -> Path:
        return self.path("work")

    def output_size(self) -> tuple[int, int, int]:
        """(width, height, fps) of the final render, honoring --preview."""
        w, h = self.platform["width"], self.platform["height"]
        fps = self.raw["render"]["fps"]
        if self.preview:
            p = self.raw["render"]["preview"]
            w, h = _even(w * p["scale"]), _even(h * p["scale"])
            fps = p["fps"]
        return w, h, fps


def _even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def _parse_length(value: Any) -> float | str:
    if value is None or str(value).strip().lower() == "auto":
        return "auto"
    try:
        seconds = float(value)
    except ValueError:
        raise MontageError(f"Length must be 'auto' or a number of seconds, got '{value}'.")
    if not 10 <= seconds <= 1200:
        raise MontageError(f"Length {seconds:g}s is out of range (10-1200 seconds).")
    return seconds


def load_config(path: Path | None = None, cli: dict[str, Any] | None = None) -> Config:
    path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not path.exists():
        raise MontageError(
            f"Settings file not found: {path}\n"
            "Restore config.yaml from the project download (it must sit next to the src/ folder)."
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise MontageError(
            f"config.yaml has a formatting mistake:\n{e}\n"
            "Tip: YAML is picky about indentation - use spaces, not tabs."
        )

    missing = [k for k in REQUIRED if k not in raw]
    if missing:
        raise MontageError(
            f"config.yaml is missing these sections: {', '.join(missing)}.\n"
            "Copy them back from the original config.yaml."
        )

    cfg = Config(raw=raw, root=path.parent.resolve())
    cfg.sources = {k: "config.yaml" for k in ("game", "platform", "style", "length")}

    # env variables, then command line (highest priority)
    for env, key in ENV_OVERRIDES.items():
        if os.environ.get(env):
            _set(cfg, key, os.environ[env].strip(), f"env {env}")
    cli = cli or {}
    for key in ("game", "platform", "style", "length"):
        if cli.get(key) is not None:
            _set(cfg, key, str(cli[key]).strip(), "command line")

    raw["montage"]["length"] = _parse_length(raw["montage"].get("length"))
    cfg.auto = bool(cli.get("auto"))
    cfg.preview = bool(cli.get("preview"))
    cfg.force = bool(cli.get("force"))
    cfg.music_choice = cli.get("music")

    _validate(cfg)
    return cfg


def _set(cfg: Config, key: str, value: str, source: str) -> None:
    if key == "length":
        cfg.raw["montage"]["length"] = value
    else:
        cfg.raw[key] = value.lower()
    cfg.sources[key] = source


def _validate(cfg: Config) -> None:
    def check(value: str, options: dict, what: str, flag: str):
        if value not in options:
            raise MontageError(
                f"Unknown {what} '{value}' (from {cfg.sources.get(flag, 'config.yaml')}).\n"
                f"Options: {', '.join(options)}"
            )

    check(cfg.game, cfg.raw["regions"], "game", "game")
    check(cfg.platform_name, cfg.raw["platforms"], "platform", "platform")
    check(cfg.style_name, cfg.raw["styles"], "style", "style")

    for name, box in cfg.regions.items():
        if (not isinstance(box, list) or len(box) != 4
                or not all(isinstance(v, (int, float)) for v in box)):
            raise MontageError(f"Region '{name}' for game '{cfg.game}' must be [x, y, width, height].")
        x, y, w, h = box
        if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1
                and x + w <= 1.0001 and y + h <= 1.0001):
            raise MontageError(
                f"Region '{name}' for game '{cfg.game}' = {box} goes outside the screen.\n"
                "Values are fractions of the screen (0.0 to 1.0) and x+width, y+height must be <= 1."
            )

    missing = [k for k in DETECTION_KEYS if k not in cfg.raw["detection"]]
    if missing:
        raise MontageError(f"config.yaml 'detection' is missing: {', '.join(missing)}.\n"
                           "Copy them back from the original config.yaml.")

    enc = cfg.raw["hardware"].get("encoder", "auto")
    if enc not in ("auto", "nvenc", "qsv", "amf", "x264"):
        raise MontageError(f"hardware.encoder must be auto, nvenc, qsv, amf or x264 (got '{enc}').")
