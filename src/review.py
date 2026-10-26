"""Stage 3 - contact sheets and the review step.

For the top candidates (detection.max_candidates) this writes one contact sheet
per clip to review/<id>.jpg, plus review/index.md and review/index.json.

Then one of:
  * --auto: selections come straight from the computed scores.
  * Claude review: in a Claude Code session, Claude looks at the sheets and
    writes work/selections.json (format below). The run stops here until a
    review for the *current* candidate list exists, then continues on rerun.
  * Manual: edit work/selections.json yourself (same format).

work/selections.json
{
  "candidates_run_id": "<run_id from work/candidates.json>",
  "reviewer": "claude" | "auto" | "manual",
  "clips": [
    {"id": "c001", "keep": true, "rating": 9, "tags": ["multikill"],
     "note": "triple kill through the doorway",
     "start": 16.0, "end": 27.2, "key_time": 20.4}     # last three optional
  ]
}
Clips missing from "clips" count as rejected. rating is 1-10.
Tags: multikill, snipe, clutch, funny, streak, kill, action, filler.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime

import numpy as np

from .config import Config
from .system_check import System
from .utils import MontageError, fmt_time, log, read_json, write_json

TAGS = {"multikill", "snipe", "clutch", "funny", "streak", "kill", "action", "filler"}
HEADER_H = 56


# ------------------------------------------------------------------ contact sheets

def frame_times(start: float, end: float, key: float, n: int) -> list[float]:
    """n evenly spaced times inside the clip, with the closest one moved onto the key moment."""
    pad = min(0.15, (end - start) / (4 * n))
    times = list(np.linspace(start + pad, end - pad, n))
    closest = min(range(n), key=lambda i: abs(times[i] - key))
    times[closest] = key
    return [round(float(t), 2) for t in sorted(times)]


def grab_frames(system: System, c: dict, times: list[float], width: int,
                size: tuple[int, int]) -> list[np.ndarray | None]:
    """Frames at `times` from one short decode of the clip (much faster than seeking per frame).

    The clip is decoded once at 10 fps and low resolution and streamed through a
    pipe; only the frames closest to the wanted times are kept.
    """
    import subprocess

    w, h = size
    height = max(2, int(round(width * h / w / 2)) * 2)
    start = max(0.0, min(times) - 0.05)
    dur = max(times) - start + 0.2
    cmd = [system.ffmpeg, "-hide_banner", "-loglevel", "error", *system.decode_args(),
           "-ss", f"{start:.3f}", "-i", c["source"], "-t", f"{dur:.3f}",
           "-vf", f"fps=10,scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    want = {i: round((t - start) * 10) for i, t in enumerate(times)}
    out: list[np.ndarray | None] = [None] * len(times)
    nbytes = width * height * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdout is not None
    k, last = 0, None
    while True:
        buf = proc.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        hits = [i for i, f in want.items() if f == k]
        if hits:
            frame = np.frombuffer(buf, np.uint8).reshape(height, width, 3).copy()
            for i in hits:
                out[i] = frame
        last = buf
        k += 1
    proc.wait()
    # times past the decoded end (rounding at the clip's tail) reuse the last frame
    if last is not None:
        tail = np.frombuffer(last, np.uint8).reshape(height, width, 3).copy()
        out = [f if f is not None else tail for f in out]
    return out


def _label(img, text, org, color=(255, 255, 255), scale=0.5):
    import cv2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(img, (x - 3, y - th - 4), (x + tw + 3, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def make_sheet(cfg: Config, system: System, c: dict) -> np.ndarray:
    import cv2

    rv = cfg["review"]
    n, cols, tw = int(rv["frames"]), int(rv["columns"]), int(rv["tile_width"])
    rows = -(-n // cols)
    times = frame_times(c["start"], c["end"], c["key_time"], n)

    meta = read_json(cfg.work / "videos" / c["video"] / "meta.json", {})
    info = meta.get("info", {})
    frames = grab_frames(system, c, times, tw, (info.get("width", 16), info.get("height", 9)))
    ref = next((f for f in frames if f is not None), None)
    th = int(round(tw * ref.shape[0] / ref.shape[1])) if ref is not None else int(tw * 9 / 16)

    sheet = np.zeros((HEADER_H + rows * th, cols * tw, 3), np.uint8)
    for i, (t, f) in enumerate(zip(times, frames)):
        x, y = (i % cols) * tw, HEADER_H + (i // cols) * th
        tile = np.full((th, tw, 3), 40, np.uint8)
        if f is not None:
            tile = cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
        else:
            _label(tile, "frame unavailable", (10, th // 2))
        near_kill = any(abs(t - k) <= 0.6 for k in c["kill_times"])
        is_key = abs(t - c["key_time"]) < 0.01
        if near_kill:
            cv2.rectangle(tile, (0, 0), (tw - 1, th - 1), (0, 0, 255), 4)
        if is_key:
            cv2.rectangle(tile, (4, 4), (tw - 5, th - 5), (0, 215, 255), 3)
        _label(tile, f"{fmt_time(int(t))}.{int((t % 1) * 10)}  (+{t - c['start']:.1f}s)", (8, 20))
        tags = (["KEY"] if is_key else []) + (["KILL"] if near_kill else [])
        if tags:
            _label(tile, " ".join(tags), (8, th - 10), (0, 215, 255) if is_key else (80, 80, 255))
        sheet[y:y + th, x:x + tw] = tile

    multi = f"  multi x{c['multikill']}" if c["multikill"] >= 2 else ""
    line1 = (f"{c['id']}   score {c['score']:.0f}   rank {c['rank']}   "
             f"kills {c['kill_count']}{multi}   hitmarkers {c['hitmarkers']}   audio {c['audio']:.2f}")
    line2 = (f"{c['name']}   {fmt_time(c['start'])} - {fmt_time(c['end'])}  ({c['duration']:.1f}s)"
             f"   red = near detected kill, yellow = key moment")
    cv2.putText(sheet, line1, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(sheet, line2, (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (190, 190, 190), 1, cv2.LINE_AA)
    return sheet


def _sheet_key(cfg: Config, c: dict) -> str:
    rv = cfg["review"]
    parts = [c["video"], c["start"], c["end"], c["key_time"], c["kill_times"], c["score"],
             c["id"], rv["frames"], rv["columns"], rv["tile_width"]]
    return hashlib.sha1(json.dumps(parts).encode()).hexdigest()[:12]


def load_candidates(cfg: Config) -> dict:
    data = read_json(cfg.work / "candidates.json")
    if not data or "candidates" not in data:
        raise MontageError("No candidates yet - run  python -m src.main --stage detect  first.")
    return data


def export_sheets(cfg: Config, system: System, cands: list[dict]) -> None:
    import cv2
    from tqdm import tqdm

    review_dir, cache_dir = cfg.path("review"), cfg.work / "sheets"
    review_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for old in review_dir.glob("c*.jpg"):
        old.unlink()

    made = 0
    for c in tqdm(cands, desc="  contact sheets", unit="clip", leave=False, dynamic_ncols=True):
        cached = cache_dir / f"{_sheet_key(cfg, c)}.jpg"
        if cfg.force or not cached.exists():
            cv2.imwrite(str(cached), make_sheet(cfg, system, c), [cv2.IMWRITE_JPEG_QUALITY, 85])
            made += 1
        shutil.copyfile(cached, review_dir / f"{c['id']}.jpg")
    log.info("Contact sheets: %d clips (%d new) in %s", len(cands), made, review_dir)


def write_index(cfg: Config, data: dict, cands: list[dict]) -> None:
    review_dir = cfg.path("review")
    write_json(review_dir / "index.json", {
        "candidates_run_id": data["run_id"],
        "game": data.get("game"),
        "clips": [{k: c[k] for k in ("id", "score", "name", "start", "end", "duration",
                                     "key_time", "kill_times", "kill_count", "multikill",
                                     "hitmarkers", "audio")} | {"sheet": f"{c['id']}.jpg"}
                  for c in cands],
    })
    lines = [
        "# Clip review",
        "",
        f"Candidate list `{data['run_id']}` - {len(cands)} clips, game preset `{data.get('game')}`.",
        "",
        "**Claude:** follow *Reviewing clips* in CLAUDE.md. Look at every sheet, then write",
        "`work/selections.json` (format in `src/review.py`) with",
        f'`"candidates_run_id": "{data["run_id"]}"` and `"reviewer": "claude"`.',
        "",
        "| id | score | recording | time | len | kills | multi | sheet |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in cands:
        multi = f"x{c['multikill']}" if c["multikill"] >= 2 else ""
        lines.append(f"| {c['id']} | {c['score']:.0f} | {c['name']} | {fmt_time(c['start'])}-"
                     f"{fmt_time(c['end'])} | {c['duration']:.1f}s | {c['kill_count']} | {multi} | "
                     f"[{c['id']}.jpg]({c['id']}.jpg) |")
    (review_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ selections

def auto_selections(cfg: Config, data: dict, cands: list[dict]) -> dict:
    min_score = cfg["review"]["auto_min_score"]
    clips = []
    for c in cands:
        if c["score"] < min_score:
            continue
        tags = (["multikill"] if c["multikill"] >= 2 else []) or (["kill"] if c["kill_count"] else ["action"])
        clips.append({"id": c["id"], "keep": True,
                      "rating": int(round(1 + 9 * c["score"] / 100)), "tags": tags,
                      "note": "auto (computed score)"})
    return {"candidates_run_id": data["run_id"], "reviewer": "auto",
            "created": datetime.now().isoformat(timespec="seconds"), "clips": clips}


def load_selections(cfg: Config) -> list[dict]:
    """Kept clips merged with candidate data, best first. Validates hand/Claude edits."""
    data = load_candidates(cfg)
    path = cfg.work / "selections.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            sel = json.load(f)
    except FileNotFoundError:
        raise MontageError("No clip selections yet - run the review stage (or use --auto).")
    except json.JSONDecodeError as e:
        raise MontageError(f"work/selections.json isn't valid JSON: {e}")

    if sel.get("candidates_run_id") != data["run_id"]:
        raise MontageError("work/selections.json is from an older candidate list. "
                           "Review again, or run with --auto.")
    by_id = {c["id"]: c for c in data["candidates"]}
    durations: dict[str, float] = {}
    kept, problems = [], []
    for entry in sel.get("clips", []):
        cid = entry.get("id")
        if cid not in by_id:
            problems.append(f"unknown clip id {cid!r}")
            continue
        if not entry.get("keep", True):
            continue
        c = dict(by_id[cid])
        if c["video"] not in durations:
            meta = read_json(cfg.work / "videos" / c["video"] / "meta.json", {})
            durations[c["video"]] = meta.get("info", {}).get("duration", c["end"])
        dur = durations[c["video"]]

        rating = entry.get("rating", round(1 + 9 * c["score"] / 100))
        if not isinstance(rating, (int, float)) or not 1 <= rating <= 10:
            problems.append(f"{cid}: rating must be 1-10")
            continue
        start = float(entry.get("start", c["start"]))
        end = float(entry.get("end", c["end"]))
        start, end = max(0.0, start), min(dur, end)
        if end - start < 1.0:
            problems.append(f"{cid}: start/end leave less than 1 second")
            continue
        key = float(entry.get("key_time", c["key_time"]))
        tags = [t for t in entry.get("tags", []) if t in TAGS]
        c.update({"start": round(start, 2), "end": round(end, 2), "duration": round(end - start, 2),
                  "key_time": round(min(max(key, start), end), 2), "rating": float(rating),
                  "tags": tags, "note": entry.get("note", "")})
        kept.append(c)

    for p in problems:
        log.warning("selections.json: %s (skipped)", p)
    if not kept:
        raise MontageError("work/selections.json keeps no clips - nothing to edit with.")
    kept.sort(key=lambda c: (-c["rating"], -c["score"]))
    return kept


# ------------------------------------------------------------------ stage entry

def review(cfg: Config, system: System) -> bool:
    """Returns True when selections are ready and the pipeline may continue."""
    data = load_candidates(cfg)
    cands = [c for c in data["candidates"] if c.get("top")]
    sel_path = cfg.work / "selections.json"

    if cfg.auto:
        sel = auto_selections(cfg, data, cands)
        write_json(sel_path, sel)
        log.info("--auto: kept %d of %d top clips by score (no visual review).",
                 len(sel["clips"]), len(cands))
        return True

    export_sheets(cfg, system, cands)
    write_index(cfg, data, cands)

    existing = read_json(sel_path)
    if (existing and existing.get("candidates_run_id") == data["run_id"]
            and existing.get("reviewer") in ("claude", "manual")):
        kept = load_selections(cfg)
        log.info("Using the %s review: %d clips kept.", existing["reviewer"], len(kept))
        return True

    log.info(
        "\nReview needed before editing. The contact sheets are in:\n    %s\n\n"
        "Option 1 - Claude review: open Claude Code in this project folder and say\n"
        "    \"review the montage clips\"\n"
        "  then run the same command again; it continues from here.\n"
        "Option 2 - skip the review:  python -m src.main --auto\n"
        "Option 3 - edit work/selections.json by hand (format in src/review.py).",
        cfg.path("review"),
    )
    return False
