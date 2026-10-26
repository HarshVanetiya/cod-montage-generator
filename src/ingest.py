"""Stage 1 - Ingest.

For every recording in assets/:
  * read duration / resolution / fps / codec with ffprobe
  * make a small constant-frame-rate proxy (e.g. 480p @ 10 fps) for analysis
  * extract mono WAV audio for analysis

Both come out of a single FFmpeg pass per file, streamed straight to disk, so
nothing is ever held in RAM. Files are processed one at a time. Results are
cached in work/videos/<id>/ and reused until the source file changes.
The original files are only ever read, never written.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from fractions import Fraction
from pathlib import Path

from tqdm import tqdm

from .config import Config
from .system_check import System, check_free_space
from .utils import (MontageError, fmt_size, fmt_time, log, read_json,
                    run, run_ffmpeg_progress, write_json)

INGEST_VERSION = 1          # bump to invalidate old caches when the format changes
PROXY_CRF = 28
# Rough bytes of cache per second of footage (proxy + 22 kHz mono WAV), padded.
CACHE_BYTES_PER_SEC = 130_000


# ---------------------------------------------------------------- discovery

def find_recordings(cfg: Config) -> list[Path]:
    assets = cfg.path("assets")
    if not assets.exists():
        assets.mkdir(parents=True)
        raise MontageError(
            f"The assets folder didn't exist, so I created it:\n    {assets}\n"
            "Put your gameplay recordings in there and run again."
        )

    exts = {e.lower() for e in cfg["ingest"]["extensions"]}
    videos, skipped = [], []
    for p in sorted(assets.rglob("*")):
        if not p.is_file() or p.name.startswith(".") or p.name == ".gitkeep":
            continue
        (videos if p.suffix.lower() in exts else skipped).append(p)

    for p in skipped:
        log.warning("Skipping unsupported file: %s (supported: %s)",
                    p.relative_to(assets), ", ".join(sorted(exts)))
    if not videos:
        raise MontageError(
            f"No recordings found in {assets}\n"
            f"Copy your gameplay videos ({', '.join(sorted(exts))}) into that folder and run again."
        )
    return videos


def video_id(path: Path) -> tuple[str, str]:
    """Stable id for the cache folder + a signature that changes if the file changes."""
    st = path.stat()
    signature = f"{path.name}|{st.st_size}|{st.st_mtime_ns}"
    digest = hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem)[:40].strip("_") or "video"
    return f"{stem}_{digest}", signature


# ---------------------------------------------------------------- probing

def _fraction(text: str | None) -> float:
    try:
        f = Fraction(text)  # "60000/1001"
        return float(f) if f > 0 else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def _rotation(stream: dict) -> int:
    rot = stream.get("tags", {}).get("rotate")
    for sd in stream.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = sd["rotation"]
    try:
        return int(float(rot or 0)) % 360
    except ValueError:
        return 0


def probe(system: System, path: Path) -> dict:
    res = run([system.ffprobe, "-v", "error", "-print_format", "json",
               "-show_format", "-show_streams", str(path)], timeout=120)
    try:
        data = json.loads(res.stdout or "{}")
    except json.JSONDecodeError:
        data = {}
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    if res.returncode != 0 or video is None:
        detail = (res.stderr or "").strip().splitlines()[-1:] or ["no video stream"]
        raise MontageError(f"can't read this file ({detail[0]}) - is it corrupt or still recording?")
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = (_fraction(data.get("format", {}).get("duration"))
                or _fraction(video.get("duration")))
    avg_fps = _fraction(video.get("avg_frame_rate"))
    max_fps = _fraction(video.get("r_frame_rate"))
    fps = avg_fps or max_fps
    rotation = _rotation(video)
    w, h = int(video.get("width", 0)), int(video.get("height", 0))
    if rotation in (90, 270):
        w, h = h, w

    return {
        "duration": round(duration, 3),
        "width": w,                       # as displayed (rotation applied)
        "height": h,
        "fps": round(fps, 3),
        # Phone screen recorders usually write variable frame rate.
        "vfr": bool(avg_fps and max_fps and abs(max_fps - avg_fps) / max_fps > 0.02),
        "rotation": rotation,
        "codec": video.get("codec_name", "?"),
        "pix_fmt": video.get("pix_fmt", "?"),
        "bit_depth": 10 if "10" in str(video.get("pix_fmt", "")) else 8,
        "start_time": _fraction(video.get("start_time")),   # negative -> 0
        "has_audio": audio is not None,
        "audio_codec": audio.get("codec_name") if audio else None,
        "size_bytes": path.stat().st_size,
    }


def proxy_size(width: int, height: int, short_side: int) -> tuple[int, int]:
    """Scale so the short side is `short_side` (never upscale), keep aspect, even numbers."""
    short = min(width, height)
    target = min(short_side, short)
    scale = target / short
    def even(v: float) -> int:
        return max(2, int(round(v / 2)) * 2)
    return even(width * scale), even(height * scale)


# ---------------------------------------------------------------- processing

def _is_cached(vdir: Path, signature: str) -> dict | None:
    meta = read_json(vdir / "meta.json")
    if not meta or meta.get("signature") != signature or meta.get("version") != INGEST_VERSION:
        return None
    if not meta.get("ingest_done") or not (vdir / "proxy.mp4").exists():
        return None
    if meta["info"]["has_audio"] and not (vdir / "audio.wav").exists():
        return None
    return meta


def _build_cmd(cfg: Config, system: System, src: Path, info: dict, vdir: Path,
               hw: bool) -> tuple[list[str], list[tuple[Path, Path]]]:
    ing = cfg["ingest"]
    pw, ph = proxy_size(info["width"], info["height"], ing["proxy_height"])
    proxy_tmp, audio_tmp = vdir / "proxy.tmp.mp4", vdir / "audio.tmp.wav"

    cmd = [system.ffmpeg, "-y", "-loglevel", "error"]
    if hw:
        cmd += system.decode_args()
    cmd += ["-i", str(src),
            # fps first so the scaler only touches frames we keep; fps= also
            # turns variable-frame-rate phone recordings into constant rate.
            "-map", "0:v:0", "-vf", f"fps={ing['proxy_fps']},scale={pw}:{ph}:flags=bilinear",
            # Proxies always use libx264 ultrafast: cheap at 480p and identical on every PC.
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", str(PROXY_CRF),
            "-g", str(ing["proxy_fps"]),        # keyframe every second -> fast seeking
            "-pix_fmt", "yuv420p", "-an", "-f", "mp4", str(proxy_tmp)]
    outputs = [(proxy_tmp, vdir / "proxy.mp4")]
    if info["has_audio"]:
        cmd += ["-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(ing["audio_sample_rate"]),
                "-c:a", "pcm_s16le", "-f", "wav", str(audio_tmp)]
        outputs.append((audio_tmp, vdir / "audio.wav"))
    return cmd, outputs


def _process(cfg: Config, system: System, src: Path, info: dict, vdir: Path,
             bar: tqdm) -> None:
    start_n = bar.n
    done = {"t": 0.0}

    def on_progress(t: float) -> None:
        t = min(t, info["duration"])
        if t > done["t"]:
            bar.update(t - done["t"])
            done["t"] = t

    attempts = [True, False] if system.hw_decode else [False]
    last_err = ""
    for hw in attempts:
        cmd, outputs = _build_cmd(cfg, system, src, info, vdir, hw)
        code, err = run_ffmpeg_progress(cmd, on_progress)
        if code == 0 and all(tmp.exists() and tmp.stat().st_size > 0 for tmp, _ in outputs):
            for tmp, final in outputs:
                os.replace(tmp, final)
            bar.update(info["duration"] - done["t"])
            return
        last_err = err.strip()
        for tmp, _ in outputs:
            tmp.unlink(missing_ok=True)
        # rewind the bar before retrying
        bar.n = start_n
        bar.refresh()
        done["t"] = 0.0
        if hw:
            log.debug("GPU decode failed for %s, retrying on CPU:\n%s", src.name, last_err)
    raise MontageError(f"FFmpeg couldn't process this file:\n{last_err[-800:]}")


def ingest(cfg: Config, system: System) -> list[dict]:
    """Run stage 1. Returns the meta dict of every usable recording."""
    assets = cfg.path("assets")
    videos_dir = cfg.work / "videos"
    recordings = find_recordings(cfg)
    log.info("Found %d recording(s) in %s", len(recordings), assets)

    # ---- probe everything first (fast) so we know the total work up front
    todo, ready, failed = [], [], []
    for src in recordings:
        vid, signature = video_id(src)
        vdir = videos_dir / vid
        rel = str(src.relative_to(assets))
        cached = None if cfg.force else _is_cached(vdir, signature)
        if cached:
            ready.append(cached)
            log.info("  [cached]  %s", rel)
            continue
        try:
            info = probe(system, src)
        except MontageError as e:
            failed.append(rel)
            log.error("  [skip]    %s: %s", rel, e)
            continue
        if info["duration"] < cfg["ingest"]["min_duration"]:
            log.warning("  [skip]    %s: only %.1fs long (min_duration is %ss)",
                        rel, info["duration"], cfg["ingest"]["min_duration"])
            continue
        notes = [f"{info['width']}x{info['height']}", f"{info['fps']:g}fps",
                 info["codec"], fmt_time(info["duration"])]
        if info["vfr"]:
            notes.append("variable fps")
        if not info["has_audio"]:
            notes.append("NO AUDIO")
        log.info("  [new]     %s  (%s)", rel, ", ".join(notes))
        if not info["has_audio"]:
            log.warning("            %s has no audio track - audio scoring will be skipped for it.", rel)
        todo.append((src, rel, vid, signature, vdir, info))

    # ---- process new files one at a time
    if todo:
        total = sum(t[5]["duration"] for t in todo)
        check_free_space(cfg.work, total * CACHE_BYTES_PER_SEC)
        log.info("Making analysis proxies for %s of footage...", fmt_time(total))
        with tqdm(total=round(total, 1), unit="s", unit_scale=False, dynamic_ncols=True,
                  bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}s footage [{elapsed}<{remaining}]") as bar:
            for src, rel, vid, signature, vdir, info in todo:
                bar.set_description(rel[:30])
                vdir.mkdir(parents=True, exist_ok=True)
                try:
                    _process(cfg, system, src, info, vdir, bar)
                except MontageError as e:
                    failed.append(rel)
                    log.error("  [failed]  %s: %s", rel, e)
                    continue
                meta = {
                    "id": vid, "version": INGEST_VERSION, "signature": signature,
                    "source": str(src), "name": rel, "info": info,
                    "proxy": {
                        "fps": cfg["ingest"]["proxy_fps"],
                        "size": proxy_size(info["width"], info["height"], cfg["ingest"]["proxy_height"]),
                    },
                    "audio_sample_rate": cfg["ingest"]["audio_sample_rate"] if info["has_audio"] else None,
                    "ingest_done": True,
                }
                write_json(vdir / "meta.json", meta)
                ready.append(meta)

    if not ready:
        raise MontageError("None of the recordings could be used - see the messages above.")

    ready.sort(key=lambda m: m["name"])
    write_json(cfg.work / "ingest.json", {"videos": [m["id"] for m in ready]})

    total = sum(m["info"]["duration"] for m in ready)
    cache = sum(f.stat().st_size for m in ready for f in (videos_dir / m["id"]).glob("*") if f.is_file())
    log.info("Ready: %d recording(s), %s of footage, cache %s%s",
             len(ready), fmt_time(total), fmt_size(cache),
             f" ({len(failed)} failed: {', '.join(failed)})" if failed else "")
    return ready


def check_music_folder(cfg: Config) -> None:
    """Early heads-up so the user doesn't find out at stage 4."""
    music = cfg.path("music")
    music.mkdir(parents=True, exist_ok=True)
    tracks = [p for p in music.iterdir() if p.suffix.lower() in (".mp3", ".wav", ".flac", ".ogg", ".m4a")]
    if not tracks:
        log.warning("Heads-up: the music folder is empty (%s). Add royalty-free tracks "
                    "before the music stage - see README for free sources.", music)
    else:
        log.info("Music folder: %d track(s)", len(tracks))


def load_ingested(cfg: Config) -> list[dict]:
    """Metadata of recordings from the last ingest (for running later stages alone)."""
    index = read_json(cfg.work / "ingest.json")
    if not index:
        raise MontageError("No ingested recordings yet - run  python -m src.main --stage ingest  first.")
    metas = []
    for vid in index["videos"]:
        meta = read_json(cfg.work / "videos" / vid / "meta.json")
        if meta and meta.get("ingest_done") and Path(meta["source"]).exists():
            metas.append(meta)
    if not metas:
        raise MontageError("The ingest cache is out of date - run  python -m src.main --stage ingest  again.")
    return metas
