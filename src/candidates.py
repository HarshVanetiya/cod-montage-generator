"""Stage 2 - find candidate highlight clips in every recording.

Per recording (one at a time):
  1. audio features      -> work/videos/<id>/audio.npz   (cached)
  2. visual features     -> work/videos/<id>/visual.npz  (cached, keyed by screen regions)
  3. scene cuts          -> work/videos/<id>/analysis.json (cached)
  4. events + per-second score (fast, always recomputed so threshold tweaks apply)
     -> work/videos/<id>/signals.csv for tuning in Excel

Then all recordings' clips are ranked together into work/candidates.json.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from . import audio_score, visual_score
from .config import Config
from .utils import MontageError, fmt_time, log, read_json, write_json

ANALYSIS_VERSION = 2
MAX_STORED = 200        # candidates kept in candidates.json (top N are flagged for review)


# ------------------------------------------------------------------ caching helpers

def _key(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:12]


def _load_npz(path: Path, key: str) -> dict | None:
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            if str(data["_key"]) != key:
                return None
            return {k: data[k] for k in data.files if k != "_key"}
    except Exception:
        return None


def _save_npz(path: Path, key: str, arrays: dict) -> None:
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, _key=np.array(key), **arrays)
    tmp.replace(path)


def analyse_video(cfg: Config, meta: dict) -> dict:
    """Features for one recording, from cache when possible."""
    vdir = cfg.work / "videos" / meta["id"]
    proxy = vdir / "proxy.mp4"
    proxy_stamp = proxy.stat().st_mtime_ns
    label = meta["name"]
    force = cfg.force

    # audio
    akey = _key(ANALYSIS_VERSION, proxy_stamp, "audio")
    audio = None if force else _load_npz(vdir / "audio.npz", akey)
    if audio is None:
        if meta["info"]["has_audio"]:
            log.info("  %s: analysing audio...", label)
            audio = audio_score.audio_features(vdir / "audio.wav", meta["audio_sample_rate"])
        else:
            audio = {"rms_db": np.zeros(0, np.float32), "onset": np.zeros(0, np.float32),
                     "frame_rate": np.float32(1)}
        _save_npz(vdir / "audio.npz", akey, audio)

    # visual (depends on the screen regions of the selected game)
    vkey = _key(ANALYSIS_VERSION, proxy_stamp, cfg.regions)
    visual = None if force else _load_npz(vdir / "visual.npz", vkey)
    if visual is None:
        log.info("  %s: scanning video for kill indicators...", label)
        visual = visual_score.frame_features(proxy, cfg.regions, label)
        _save_npz(vdir / "visual.npz", vkey, visual)

    # scene cuts
    det = cfg["detection"]
    ckey = _key(ANALYSIS_VERSION, proxy_stamp, det["scene_threshold"])
    analysis = read_json(vdir / "analysis.json", {})
    if force or analysis.get("cuts_key") != ckey:
        log.info("  %s: detecting scene changes...", label)
        analysis = {"cuts_key": ckey, "cuts": visual_score.scene_cuts(proxy, det["scene_threshold"])}
        write_json(vdir / "analysis.json", analysis)

    return {"audio": audio, "visual": visual, "cuts": analysis["cuts"]}


# ------------------------------------------------------------------ scoring

def score_video(cfg: Config, meta: dict, feats: dict) -> dict:
    det = cfg["detection"]
    duration = meta["info"]["duration"]
    n_sec = max(1, int(math.ceil(duration)))

    audio = audio_score.per_second(feats["audio"], n_sec)
    events = visual_score.detect_events(feats["visual"], det)
    fps = events["fps"]

    # per-second exclusion: mostly-excluded seconds are out
    ex_frames = events["excluded"]
    sec_idx = np.minimum((np.arange(len(ex_frames)) / fps).astype(int), n_sec - 1)
    ex_count = np.bincount(sec_idx, weights=ex_frames.astype(float), minlength=n_sec)[:n_sec]
    fr_count = np.maximum(np.bincount(sec_idx, minlength=n_sec)[:n_sec], 1)
    excluded = (ex_count / fr_count) > 0.5

    kills = events["kills"]
    kill_sec = np.zeros(n_sec, np.float32)
    for k in kills:
        kill_sec[min(int(k["t"]), n_sec - 1)] += k["strength"]
    hit_sec = np.zeros(n_sec, np.float32)
    for hm in events["hits"]:
        hit_sec[min(int(hm["t"]), n_sec - 1)] += hm["strength"]

    # multi-kill bonus over the seconds the streak spans
    bonus = np.zeros(n_sec, np.float32)
    times = [k["t"] for k in kills]
    for i, t in enumerate(times):
        in_win = [u for u in times[i:] if u - t <= det["multikill_window"]]
        if len(in_win) >= det["multikill_min_kills"]:
            level = len(in_win) - det["multikill_min_kills"] + 1
            a, b = int(t), min(int(in_win[-1]), n_sec - 1)
            bonus[a:b + 1] = np.maximum(bonus[a:b + 1], det["multikill_bonus"] * level)

    score = (det["weight_audio"] * audio["intensity"]
             + det["weight_kills"] * (kill_sec + 0.3 * np.minimum(hit_sec, 3))
             + bonus)
    score[excluded] = 0

    # A kill popup can look like a hard cut to PySceneDetect; real cuts (to a
    # killcam, menu or respawn) practically never land on a kill.
    cuts = [c for c in feats["cuts"] if all(abs(c - t) > 0.6 for t in times)]

    _write_signals_csv(cfg.work / "videos" / meta["id"] / "signals.csv",
                       audio, kill_sec, hit_sec, bonus, excluded, score)
    return {"n_sec": n_sec, "audio": audio, "kills": kills, "hits": events["hits"],
            "kill_sec": kill_sec, "hit_sec": hit_sec, "bonus": bonus,
            "excluded": excluded, "score": score, "cuts": cuts}


def _write_signals_csv(path, audio, kill_sec, hit_sec, bonus, excluded, score) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["second", "time", "audio", "loudness", "gunfire_density",
                    "kills", "hitmarkers", "multikill_bonus", "excluded", "score"])
        for s in range(len(score)):
            w.writerow([s, fmt_time(s), f"{audio['intensity'][s]:.2f}", f"{audio['loudness'][s]:.2f}",
                        f"{audio['density'][s]:.2f}", f"{kill_sec[s]:.2f}", f"{hit_sec[s]:.2f}",
                        f"{bonus[s]:.1f}", int(excluded[s]), f"{score[s]:.2f}"])


# ------------------------------------------------------------------ clip building

def _segment_bounds(t: float, cuts: list[float], excluded: np.ndarray, duration: float):
    """The stretch of usable gameplay around time t (between cuts / excluded seconds)."""
    n = len(excluded)
    s = min(int(t), n - 1)
    lo = s
    while lo > 0 and not excluded[lo - 1]:
        lo -= 1
    hi = s
    while hi < n - 1 and not excluded[hi + 1]:
        hi += 1
    start, end = float(lo), min(float(hi + 1), duration)
    for c in cuts:
        if c <= t:
            start = max(start, c)
        elif c > t:
            end = min(end, c)
            break
    return start, end


def build_clips(cfg: Config, meta: dict, sv: dict) -> list[dict]:
    det = cfg["detection"]
    duration = meta["info"]["duration"]
    excluded, score, cuts = sv["excluded"], sv["score"], sv["cuts"]
    intensity = sv["audio"]["intensity"]

    # moments: every kill, plus strong audio-only action (fights without a detected kill)
    moments = [(k["t"], "kill") for k in sv["kills"]]
    kill_secs = {int(k["t"]) for k in sv["kills"]}
    for s in np.flatnonzero((intensity >= det["audio_peak"]) & ~excluded):
        if int(s) not in kill_secs:
            moments.append((float(s) + 0.5, "audio"))
    moments.sort()
    if not moments:
        return []

    def blocked(a: float, b: float) -> bool:
        if any(a < c <= b for c in cuts):
            return True
        return bool(excluded[int(a):int(b) + 1].any())

    groups, cur = [], [moments[0]]
    for m in moments[1:]:
        if m[0] - cur[-1][0] <= det["merge_gap"] and not blocked(cur[-1][0], m[0]):
            cur.append(m)
        else:
            groups.append(cur)
            cur = [m]
    groups.append(cur)

    clips = []
    max_len = det["max_clip_len"]
    for g in groups:
        kill_ts = [t for t, kind in g if kind == "kill"]
        anchor = kill_ts[0] if kill_ts else g[0][0]
        seg_start, seg_end = _segment_bounds(anchor, cuts, excluded, duration)
        start = max(g[0][0] - det["lead_in"], seg_start, 0.0)
        end = min(g[-1][0] + det["lead_out"], seg_end, duration)

        if end - start > max_len:  # keep the best-scoring window
            a, b = int(start), int(math.ceil(end))
            csum = np.concatenate([[0], np.cumsum(score[a:b])])
            w = int(max_len)
            best = int(np.argmax(csum[w:] - csum[:-w])) if len(csum) > w else 0
            start, end = float(a + best), float(min(a + best + w, end))
        if end - start < 2.0:
            continue

        in_clip = [k for k in sv["kills"] if start <= k["t"] <= end]
        times = [k["t"] for k in in_clip]
        multikill = max((sum(1 for u in times if 0 <= u - t <= det["multikill_window"]) for t in times),
                        default=0)
        hits = [hm for hm in sv["hits"] if start <= hm["t"] <= end]
        a, b = int(start), max(int(start) + 1, int(math.ceil(end)))
        audio_mean = float(np.mean(intensity[a:b]))
        if in_clip:
            key_time = max(in_clip, key=lambda k: k["strength"])["t"]
        else:
            key_time = a + float(np.argmax(intensity[a:b])) + 0.5

        kill_value = sum(k["strength"] for k in in_clip)
        multi_value = (det["multikill_bonus"] * (multikill - det["multikill_min_kills"] + 1)
                       if multikill >= det["multikill_min_kills"] else 0.0)
        value = (6 * det["weight_kills"] * kill_value + 6 * multi_value
                 + 10 * det["weight_audio"] * audio_mean + 2 * min(len(hits), 10))

        clips.append({
            "video": meta["id"], "name": meta["name"], "source": meta["source"],
            "start": round(start, 2), "end": round(end, 2), "duration": round(end - start, 2),
            "key_time": round(min(max(key_time, start), end), 2),
            "kill_times": times, "kill_count": len(times), "multikill": multikill,
            "hitmarkers": len(hits), "audio": round(audio_mean, 3), "value": round(value, 3),
        })
    return clips


# ------------------------------------------------------------------ stage entry

def detect(cfg: Config, videos: list[dict]) -> list[dict]:
    det = cfg["detection"]
    log.info("Game preset: %s (screen regions from config.yaml)", cfg.game)

    all_clips, summaries = [], []
    for i, meta in enumerate(videos, 1):
        log.info("[%d/%d] %s (%s)", i, len(videos), meta["name"], fmt_time(meta["info"]["duration"]))
        feats = analyse_video(cfg, meta)
        sv = score_video(cfg, meta, feats)
        clips = build_clips(cfg, meta, sv)
        all_clips += clips
        ex_pct = 100 * float(sv["excluded"].mean())
        multis = sum(1 for c in clips if c["multikill"] >= det["multikill_min_kills"])
        log.info("        %d kill events, %d hitmarkers, %d multi-kills, %d scene cuts, "
                 "%.0f%% excluded (menus/loading/death) -> %d clips",
                 len(sv["kills"]), len(sv["hits"]), multis, len(sv["cuts"]), ex_pct, len(clips))
        summaries.append({"video": meta["id"], "name": meta["name"], "kills": len(sv["kills"]),
                          "hitmarkers": len(sv["hits"]), "cuts": len(sv["cuts"]),
                          "excluded_pct": round(ex_pct, 1), "clips": len(clips)})

    if not all_clips:
        raise MontageError(
            "No highlight candidates were found.\n"
            "Likely causes: the screen regions don't match your game's HUD, or the thresholds "
            "are too strict.\nRun  python -m src.main --stage regions  to check the boxes, and see "
            "work/videos/<name>/signals.csv for the raw scores."
        )

    all_clips.sort(key=lambda c: -c["value"])
    all_clips = all_clips[:MAX_STORED]
    top_value = all_clips[0]["value"] or 1.0
    for rank, c in enumerate(all_clips, 1):
        c["id"] = f"c{rank:03d}"
        c["rank"] = rank
        c["score"] = round(100 * c["value"] / top_value, 1)
        c["top"] = rank <= det["max_candidates"]

    # Identifies this exact candidate list, so a review of an older list is never reused.
    run_id = _key([(c["id"], c["video"], c["start"], c["end"]) for c in all_clips])
    write_json(cfg.work / "candidates.json", {
        "run_id": run_id,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "game": cfg.game,
        "detection": det,
        "videos": summaries,
        "candidates": [{k: c[k] for k in ("id", "rank", "score", "top", "video", "name", "source",
                                          "start", "end", "duration", "key_time", "kill_times",
                                          "kill_count", "multikill", "hitmarkers", "audio")}
                       for c in all_clips],
    })

    log.info("\nTop candidates (of %d):", len(all_clips))
    log.info("  %-5s %5s  %-28s %-15s %5s %6s", "id", "score", "recording", "time", "kills", "multi")
    for c in all_clips[:10]:
        log.info("  %-5s %5.0f  %-28s %-15s %5d %6s", c["id"], c["score"], c["name"][:28],
                 f"{fmt_time(c['start'])}-{fmt_time(c['end'])}", c["kill_count"],
                 f"x{c['multikill']}" if c["multikill"] >= 2 else "")
    log.info("Saved %s", cfg.work / "candidates.json")
    return all_clips
