"""Stage 4a - music: analyse tracks, pick one (or chain several), build the beat map.

Per track (cached in work/music/<id>.json, decoded once to a small mono WAV):
  * tempo and beat times (librosa), downbeats (bar phase from kick energy)
  * per-beat energy (loudness + bass), high/low sections, drops, build-ups
  * where the music really ends (trailing silence trimmed)

The soundtrack is then planned on the montage timeline:
  * starts a few bars before the first big drop, so the hook lands inside the
    platform's hook_seconds
  * ends on a phrase boundary near the target length (with a fade-out)
  * chains extra tracks with a crossfade on downbeats if one is too short

Output: work/beatmap.json - every time in it is montage-timeline seconds.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from .config import Config
from .system_check import System
from .utils import MontageError, fmt_time, log, read_json, run, write_json

MUSIC_EXTS = (".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac")
SR = 22050
HOP = 512
ANALYSIS_VERSION = 1
NO_MUSIC_HELP = (
    "Add royalty-free tracks (mp3/wav) to that folder. Free sources:\n"
    "  YouTube Audio Library  https://studio.youtube.com -> Audio library\n"
    "  Pixabay Music          https://pixabay.com/music/\n"
    "  NoCopyrightSounds      https://ncs.io"
)


# ------------------------------------------------------------------ track discovery

def list_tracks(cfg: Config) -> list[Path]:
    folder = cfg.path("music")
    folder.mkdir(parents=True, exist_ok=True)
    tracks = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in MUSIC_EXTS)
    if not tracks:
        raise MontageError(f"No music found in {folder}\n{NO_MUSIC_HELP}")
    return tracks


def _find_track(tracks: list[Path], name: str) -> Path:
    exact = [t for t in tracks if t.name.lower() == name.lower()]
    if exact:
        return exact[0]
    partial = [t for t in tracks if name.lower() in t.name.lower()]
    if len(partial) == 1:
        return partial[0]
    options = "\n  ".join(t.name for t in tracks)
    if partial:
        raise MontageError(f"'{name}' matches several tracks - be more specific:\n  "
                           + "\n  ".join(t.name for t in partial))
    raise MontageError(f"Track '{name}' isn't in the music folder. Available:\n  {options}")


def _track_id(path: Path) -> str:
    st = path.stat()
    digest = hashlib.sha1(f"{path.name}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:8]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", path.stem)[:40].strip("_") or "track"
    return f"{stem}_{digest}"


# ------------------------------------------------------------------ analysis

def _decode(system: System, src: Path, wav: Path) -> None:
    """Any format -> small mono WAV via FFmpeg (works for mp3/m4a on every PC)."""
    tmp = wav.with_name(wav.stem + ".tmp.wav")
    res = run([system.ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(src),
               "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(tmp)], timeout=300)
    if res.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise MontageError(f"Couldn't read the music file {src.name}: {res.stderr.strip()[-300:]}")
    tmp.replace(wav)


def _norm(x: np.ndarray, lo: float = 5, hi: float = 95) -> np.ndarray:
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / max(b - a, 1e-9), 0, 1)


def analyse_track(cfg: Config, system: System, path: Path) -> dict:
    import librosa
    import soundfile as sf

    tid = _track_id(path)
    mdir = cfg.work / "music"
    mdir.mkdir(parents=True, exist_ok=True)
    cache = mdir / f"{tid}.json"
    cached = read_json(cache)
    if cached and cached.get("version") == ANALYSIS_VERSION and not cfg.force:
        return cached

    log.info("  analysing %s ...", path.name)
    wav = mdir / f"{tid}.wav"
    if not wav.exists():
        _decode(system, path, wav)
    y, sr = sf.read(str(wav), dtype="float32")
    duration = len(y) / sr
    if duration < 15:
        raise MontageError(f"{path.name} is only {duration:.0f}s long - too short for a montage track.")

    # beats
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=HOP)
    tempo = float(np.atleast_1d(tempo)[0])
    beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP)
    if len(beats) < 16:
        raise MontageError(f"Couldn't find a steady beat in {path.name} - pick a track with clear drums.")
    # The tracker drops beats in quiet intros/outros; extend the grid at the same tempo
    # so cuts (and drop detection) work across the whole track.
    idx = np.arange(len(beats))
    period = float(np.polyfit(idx, beats, 1)[0])    # regression: no drift when extending
    head = np.arange(beats[0] - period, 0, -period)[::-1]
    tail = np.arange(beats[-1] + period, duration - period / 2, period)
    beats = np.concatenate([head, beats, tail])
    beat_frames = librosa.time_to_frames(beats, sr=sr, hop_length=HOP)

    # loudness and bass per beat
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    mel = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=HOP, n_mels=64, fmax=8000)
    mel_f = librosa.mel_frequencies(n_mels=64, fmax=8000)
    bass = mel[mel_f < 200].sum(axis=0)
    low_onset = librosa.onset.onset_strength(S=librosa.power_to_db(mel[mel_f < 200]), sr=sr,
                                             hop_length=HOP)
    edges = np.append(beat_frames, len(rms))
    rms_b = np.array([rms[a:max(a + 1, b)].mean() for a, b in zip(edges[:-1], edges[1:])])
    bass_b = np.array([bass[a:max(a + 1, b)].mean() for a, b in zip(edges[:-1], edges[1:])])
    energy = 0.5 * _norm(20 * np.log10(rms_b + 1e-6)) + 0.5 * _norm(10 * np.log10(bass_b + 1e-9))
    bar_energy = np.convolve(energy, np.ones(4) / 4, mode="same")      # smoothed over a bar

    # how hard each beat hits (for landing kills / zooms on strong beats)
    strength = _norm(onset_env[np.minimum(beat_frames, len(onset_env) - 1)], 0, 99)

    # drops: beats where the next 2 bars are much louder than the previous 2,
    # refined to the single beat with the sharpest rise
    min_jump = cfg["music"]["drop_min_jump"]
    n = len(beats)
    jumps = []
    for i in range(8, n - 8):
        before, after = energy[i - 8:i].mean(), energy[i:i + 8].mean()
        if after - before >= min_jump and after >= 0.55:
            jumps.append((i, float(after - before)))
    drop_idx = []
    for i, j in jumps:  # one drop per build-up
        if all(j >= j2 for i2, j2 in jumps if abs(i2 - i) <= 8):
            k = max(range(max(1, i - 2), min(n, i + 3)), key=lambda b: energy[b] - energy[b - 1])
            if not drop_idx or k - drop_idx[-1][0] > 8:
                drop_idx.append((k, j))

    # downbeats (assumes 4/4). Drops start on bar 1, so they fix the bar phase.
    # Without drops: the phase with the biggest energy changes plus the strongest kick.
    if drop_idx:
        votes = np.zeros(4)
        for k, j in drop_idx:
            votes[k % 4] += j
        phase = int(np.argmax(votes))
    else:
        lo_at_beats = _norm(low_onset[np.minimum(beat_frames, len(low_onset) - 1)], 0, 99)
        prev = np.array([energy[max(0, i - 4):i].mean() if i else energy[0] for i in range(n)])
        novelty = np.maximum(0, energy - prev)
        novelty[novelty < np.percentile(novelty, 95)] = 0
        phase = int(np.argmax([2 * novelty[p::4].sum() + 0.5 * lo_at_beats[p::4].mean() * n / 4
                               for p in range(4)]))
    downbeat_idx = list(range(phase, n, 4))
    drops = [{"beat": k, "t": round(float(beats[k]), 3), "strength": round(j, 3)} for k, j in drop_idx]

    # high / low sections with hysteresis, at least 2 bars long
    level, sections, start = "low", [], 0
    for i, e in enumerate(bar_energy):
        new = "high" if (level == "low" and e >= 0.62) else "low" if (level == "high" and e < 0.48) else level
        if new != level:
            if i - start >= 8:
                sections.append({"start": round(float(beats[start]), 3),
                                 "end": round(float(beats[i]), 3), "level": level})
                start = i
            level = new
    sections.append({"start": round(float(beats[start]), 3), "end": round(duration, 3), "level": level})

    # where the music really ends (ignore trailing silence)
    loud = np.flatnonzero(rms > 0.05 * rms.max())
    music_end = float(librosa.frames_to_time(loud[-1], sr=sr, hop_length=HOP)) if len(loud) else duration

    info = {
        "version": ANALYSIS_VERSION, "id": tid, "file": str(path), "name": path.name,
        "duration": round(duration, 3), "music_end": round(min(music_end + 0.3, duration), 3),
        "bpm": round(tempo, 1),
        "beats": [round(float(b), 3) for b in beats],
        "downbeat_idx": downbeat_idx,
        "strength": [round(float(s), 3) for s in strength],
        "energy": [round(float(e), 3) for e in bar_energy],
        "drops": drops, "sections": sections,
        "mean_energy": round(float(energy.mean()), 3),
        "dynamics": round(float(np.percentile(bar_energy, 90) - np.percentile(bar_energy, 10)), 3),
    }
    write_json(cache, info)
    return info


# ------------------------------------------------------------------ picking

def track_score(cfg: Config, info: dict, target: float, history: list[str]) -> float:
    lo, hi = cfg["music"]["bpm_range"]
    bpm = info["bpm"]
    tempo_ok = any(lo <= b <= hi for b in (bpm, bpm * 2, bpm / 2))
    usable = info["music_end"]
    drop = max((d["strength"] for d in info["drops"]), default=0.0)

    style = cfg.style_name
    if style == "hype":
        character = 0.5 * info["mean_energy"] + 0.3 * drop + 0.2 * min(bpm / 170, 1)
    elif style == "cinematic":
        character = 0.4 * info["dynamics"] + 0.4 * drop + 0.2 * info["mean_energy"]
    else:
        character = 0.4 * info["mean_energy"] + 0.3 * drop + 0.3 * info["dynamics"]

    fit = 1.0 if usable >= target else 0.55 + 0.4 * usable / target
    score = 0.45 * character + 0.35 * fit + 0.2 * (1.0 if tempo_ok else 0.4)
    if not info["drops"]:
        score -= 0.1
    # rotation: recently used tracks are pushed down (most recent hardest)
    for age, tid in enumerate(history[:3]):
        if tid == info["id"]:
            score -= 0.3 - 0.1 * age
    return round(score, 4)


def history_path(cfg: Config) -> Path:
    return cfg.work / "music" / "history.json"


def remember_used(cfg: Config, track_ids: list[str]) -> None:
    """Called after a successful render so the next montage rotates to other music."""
    hist = read_json(history_path(cfg), {"used": []})["used"]
    hist = track_ids + [t for t in hist if t not in track_ids]
    write_json(history_path(cfg), {"used": hist[:10]})


# ------------------------------------------------------------------ soundtrack plan

def plan_soundtrack(cfg: Config, chosen: list[dict], target: float) -> dict:
    """Lay tracks on the montage timeline and build the combined beat map."""
    xf = cfg["music"]["crossfade"]
    hook = cfg.platform["hook_seconds"]
    pieces, t_cursor, complete = [], 0.0, False

    for n, info in enumerate(chosen):
        beats = info["beats"]
        downs = [beats[i] for i in info["downbeat_idx"]]
        if n == 0:
            # start a few bars before the first big drop so the drop lands within hook_seconds
            big = [d for d in info["drops"]
                   if d["strength"] >= 0.6 * max(x["strength"] for x in info["drops"])]
            first_drop = big[0]["t"] if big else (downs[0] if downs else 0.0)
            starts = [d for d in downs if 0 <= first_drop - d <= hook]
            t_in = starts[0] if starts else max(0.0, first_drop - hook)
            if first_drop - t_in < 0.5:
                t_in = first_drop  # drop right at the start (short-form)
        else:
            t_in = downs[0] if downs else 0.0
        remaining = target - t_cursor
        t_end = info["music_end"]
        if t_in + remaining <= t_end:  # this track finishes the montage
            # end on a downbeat near the target, preferably a phrase start (every 4 bars)
            want = t_in + remaining
            phrase = [d for k, d in enumerate(downs) if k % 4 == 0 and t_in < d <= t_end]
            cands = phrase or [d for d in downs if t_in < d <= t_end] or [t_end]
            t_out = min(cands, key=lambda d: abs(d - want))
            if abs(t_out - want) > 8:
                t_out = min(want, t_end)
            pieces.append({"info": info, "in": t_in, "out": t_out, "at": t_cursor})
            t_cursor += t_out - t_in
            complete = True
            break
        # too short: use it to a late downbeat and crossfade into the next track
        late = [d for d in downs if d <= t_end - xf] or [t_end - xf]
        t_out = late[-1] + xf
        pieces.append({"info": info, "in": t_in, "out": min(t_out, info["duration"]), "at": t_cursor})
        t_cursor += (min(t_out, info["duration"]) - t_in) - xf
    else:
        t_cursor += xf  # the last piece ran out without a crossfade partner

    total = t_cursor

    # combined beat map on the montage timeline
    beats, strength, energy, downbeat = [], [], [], []
    drops, sections = [], []
    for k, p in enumerate(pieces):
        info, shift = p["info"], p["at"] - p["in"]
        lo = p["in"] if k == 0 else p["in"] + xf / 2         # hand over mid-crossfade
        hi = p["out"] - (xf / 2 if k < len(pieces) - 1 else 0)
        down_set = set(info["downbeat_idx"])
        for i, b in enumerate(info["beats"]):
            if lo - 1e-6 <= b < hi:
                beats.append(round(b + shift, 3))
                strength.append(info["strength"][i])
                energy.append(info["energy"][i])
                downbeat.append(i in down_set)
        drops += [{"t": round(d["t"] + shift, 3), "strength": d["strength"]}
                  for d in info["drops"] if lo <= d["t"] < hi]
        sections += [{"start": round(max(s["start"], lo) + shift, 3),
                      "end": round(min(s["end"], hi) + shift, 3), "level": s["level"]}
                     for s in info["sections"] if s["end"] > lo and s["start"] < hi]

    if not drops:  # no detected drop: treat the start of the first high section as one
        high = [s for s in sections if s["level"] == "high"]
        drops = [{"t": high[0]["start"] if high else 0.0, "strength": 0.0}]

    return {
        "duration": round(total, 3),
        "complete": complete,     # False = ran out of music before the target
        "tracks": [{"file": p["info"]["file"], "name": p["info"]["name"], "id": p["info"]["id"],
                    "bpm": p["info"]["bpm"], "in": round(p["in"], 3), "out": round(p["out"], 3),
                    "at": round(p["at"], 3)} for p in pieces],
        "crossfade": xf, "fade_out": cfg["music"]["fade_out"],
        "bpm": pieces[0]["info"]["bpm"],
        "beats": beats, "downbeat": downbeat, "strength": strength, "energy": energy,
        "drops": drops, "sections": sections,
    }


# ------------------------------------------------------------------ stage entry

def build_beatmap(cfg: Config, system: System, target: float) -> dict:
    tracks = list_tracks(cfg)
    log.info("Music folder: %d track(s)", len(tracks))
    infos = {}
    for t in tracks:
        try:
            infos[t] = analyse_track(cfg, system, t)
        except MontageError as e:
            log.warning("  skipping %s: %s", t.name, e)
    if not infos:
        raise MontageError("None of the music tracks could be used - see the messages above.")

    history = read_json(history_path(cfg), {"used": []})["used"]
    choice = cfg.music_choice or (cfg["music"]["pick"] if cfg["music"]["pick"] != "auto" else None)
    if choice:
        first = infos.get(_find_track(list(infos), choice))
        if first is None:
            raise MontageError(f"'{choice}' couldn't be analysed - see the messages above.")
        ranked = [first] + sorted((i for i in infos.values() if i is not first),
                                  key=lambda i: -track_score(cfg, i, target, history))
        log.info("Track: %s (chosen by you)", first["name"])
    else:
        ranked = sorted(infos.values(), key=lambda i: -track_score(cfg, i, target, history))
        for i in ranked:
            log.debug("  score %.3f  %s", track_score(cfg, i, target, history), i["name"])
        log.info("Track: %s (auto-picked: %.0f BPM, %d drop(s), %s long)", ranked[0]["name"],
                 ranked[0]["bpm"], len(ranked[0]["drops"]), fmt_time(ranked[0]["music_end"]))

    # chain more tracks only while the planned soundtrack is too short (max 3)
    chosen = [ranked[0]]
    bm = plan_soundtrack(cfg, chosen, target)
    for extra in ranked[1:3]:
        if bm["complete"]:
            break
        chosen.append(extra)
        bm = plan_soundtrack(cfg, chosen, target)
    if len(bm["tracks"]) > 1:
        log.info("Chaining %d tracks to reach %s: %s", len(bm["tracks"]), fmt_time(target),
                 " -> ".join(t["name"] for t in bm["tracks"]))

    if bm["duration"] < target - 10:
        log.warning("All music together is only %s - the montage will be that long.",
                    fmt_time(bm["duration"]))
    log.info("Soundtrack: %s, %d beats, first drop at %s, %d drop(s)",
             fmt_time(bm["duration"]), len(bm["beats"]), f"{bm['drops'][0]['t']:.1f}s", len(bm["drops"]))
    write_json(cfg.work / "beatmap.json", bm)
    return bm
