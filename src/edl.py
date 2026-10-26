"""Stage 4b - the edit decision list (work/edl.json).

Walks the soundtrack beat by beat and fills it with the reviewed clips:
  * hook:   the best clip; its kill lands on the first big drop
  * finale: the second-best clip fills the last slot
  * drops:  every later drop starts a new clip with the best clip left
  * pacing: short clips where the music is loud, longer where it's calm
            (montage.calm_clip / busy_clip, times platform and style pace)
  * every cut is on a beat (downbeats preferred) and each clip is shifted so
    its kill moment hits a strong beat
  * no more than montage.max_same_source_in_row clips in a row from one recording

Effects are chosen in stage 5; this file only records the timing and hints.
"""

from __future__ import annotations

import bisect

import numpy as np

from .config import Config
from .utils import MontageError, fmt_time, log, write_json

# ------------------------------------------------------------------ length

def pace(cfg: Config) -> float:
    return float(cfg.platform["pace"]) * float(cfg.style["pace"])


def decide_length(cfg: Config, clips: list[dict]) -> float:
    """Target montage length: fixed by the user, or from how much good material exists."""
    lo, hi = cfg.platform["length_min"], cfg.platform["length_max"]
    if cfg.length != "auto":
        return float(cfg.length)
    m = cfg["montage"]
    avg_slot = (m["calm_clip"] + m["busy_clip"]) / 2 / pace(cfg)
    material = 0.0
    for c in clips:
        used = min(c["duration"], avg_slot * 1.2, m["max_clip"])
        material += used if c["rating"] >= 4 else used * 0.5   # weak clips count half
    target = min(max(material, lo), hi)
    if material < lo:
        target = max(material, 15.0)
        log.warning("Only about %s of good clips - the montage will be %s, shorter than the "
                    "usual %s-%s for %s. Add more recordings for a longer one.",
                    fmt_time(material), fmt_time(target), fmt_time(lo), fmt_time(hi),
                    cfg.platform_name)
    log.info("Length: %s (auto, from %d clips / ~%s of usable action)",
             fmt_time(target), len(clips), fmt_time(material))
    return round(target, 1)


# ------------------------------------------------------------------ beat helpers

class Grid:
    def __init__(self, bm: dict):
        self.t = np.asarray(bm["beats"], float)
        self.down = np.asarray(bm["downbeat"], bool)
        self.strength = np.asarray(bm["strength"], float)
        self.energy = np.asarray(bm["energy"], float)
        self.drops = [d["t"] for d in bm["drops"]]

    def snap(self, t: float, lo: float, hi: float) -> float:
        """Beat closest to t inside [lo, hi]; a downbeat wins if it's within one beat."""
        idx = np.flatnonzero((self.t >= lo - 1e-6) & (self.t <= hi + 1e-6))
        if len(idx) == 0:
            return t
        beat_len = float(np.median(np.diff(self.t))) if len(self.t) > 1 else 0.5
        near = idx[np.argmin(np.abs(self.t[idx] - t))]
        downs = [i for i in idx if self.down[i] and abs(self.t[i] - t) <= beat_len * 1.05]
        if downs:
            near = min(downs, key=lambda i: abs(self.t[i] - t))
        return float(self.t[near])

    def energy_at(self, t: float, span: float = 4.0) -> float:
        idx = np.flatnonzero((self.t >= t) & (self.t < t + span))
        if len(idx) == 0:
            idx = [min(bisect.bisect_left(list(self.t), t), len(self.t) - 1)]
        return float(np.mean(self.energy[idx]))

    def beats_in(self, a: float, b: float) -> np.ndarray:
        return np.flatnonzero((self.t >= a - 1e-6) & (self.t < b - 1e-6))


# ------------------------------------------------------------------ clip choice

def _usable(c: dict, max_clip: float) -> float:
    return min(c["end"] - c["start"], max_clip)


def _pick(pool: list[dict], energy: float, on_drop: bool, prev: list[str], max_row: int,
          want_len: float, max_clip: float, must_reach: bool = False) -> dict | None:
    if not pool:
        return None
    tail = 0
    for v in reversed(prev):
        if prev and v == prev[-1]:
            tail += 1
        else:
            break
    allowed = [c for c in pool if not (prev and c["video"] == prev[-1] and tail >= max_row)]
    cands = allowed or pool  # if every clip is from one recording, there's no choice

    if must_reach:  # the slot has to reach the drop: prefer clips long enough
        cands = [c for c in cands if _usable(c, max_clip) >= want_len] or cands

    def score(c: dict) -> float:
        r = (c["rating"] - 1) / 9
        # match clip intensity to the music; strong clips are saved for loud parts
        s = 0.5 * r - 0.9 * abs(r - energy)
        if on_drop:
            s += 0.8 * r                              # drops get the best clip left
        if _usable(c, max_clip) >= want_len:
            s += 0.15
        return s

    return max(cands, key=score)


def _kill_span(c: dict) -> float:
    kills = c.get("kill_times") or []
    return (max(kills) - min(kills)) if len(kills) > 1 else 0.0


def _place_key(grid: Grid, c: dict, t0: float, t1: float, force_beat: float | None) -> tuple[float, float]:
    """Choose src_in so the clip's key moment lands on a strong beat in [t0, t1],
    keeping every kill of the clip on screen when the slot is long enough."""
    slot = t1 - t0
    lo_in, hi_in = c["start"], c["end"] - slot
    kills = c.get("kill_times") or []
    if kills:  # narrow the range so the first kill has lead-up and the last one is shown
        need_lo, need_hi = min(kills) - 0.8, max(kills) + 0.6
        a, b = max(lo_in, need_hi - slot), min(hi_in, need_lo)
        if a <= b:
            lo_in, hi_in = a, b
    key = c["key_time"]
    options = []
    if force_beat is not None and t0 <= force_beat <= t1:
        options.append(force_beat)
    idx = grid.beats_in(t0 + min(0.6, slot / 4), t1 - min(0.4, slot / 4))
    ranked = sorted(idx, key=lambda i: -(grid.strength[i] + (0.3 if grid.down[i] else 0)))
    options += [float(grid.t[i]) for i in ranked]
    # any kill can be the one on the beat; the strongest (key) is tried first
    anchors = [key] + [k for k in kills if abs(k - key) > 0.05]
    for b in options:
        for a in anchors:
            src_in = a - (b - t0)
            if lo_in - 1e-6 <= src_in <= hi_in + 1e-6:
                return round(src_in, 3), round(t0 + (key - src_in), 3)
    # no beat fits inside the clip's bounds: keep the key as close to the best beat as possible
    b = options[0] if options else (t0 + t1) / 2
    src_in = float(np.clip(key - (b - t0), lo_in, max(lo_in, hi_in)))
    return round(src_in, 3), round(t0 + (key - src_in), 3)


# ------------------------------------------------------------------ build

def build_edl(cfg: Config, clips: list[dict], bm: dict) -> dict:
    if not clips:
        raise MontageError("No clips to edit with.")
    m = cfg["montage"]
    min_clip, max_clip = m["min_clip"], m["max_clip"]
    p = pace(cfg)
    grid = Grid(bm)
    T = float(bm["duration"])

    pool = sorted(clips, key=lambda c: (-c["rating"], -c["score"]))
    hook_clip = pool.pop(0)
    finale_clip = pool.pop(0) if pool else None
    finale_len = min(_usable(finale_clip, max_clip), max(min_clip, 7.0)) if finale_clip else 0.0
    first_drop = grid.drops[0] if grid.drops else 0.0
    later_drops = [d for d in grid.drops[1:]]

    slots: list[dict] = []
    t, hook_done = 0.0, False
    while T - t > 0.5:
        rem = T - t
        e = grid.energy_at(t)
        on_drop = any(abs(t - d) < 0.05 for d in later_drops)
        force_beat = None

        if not hook_done and (t >= first_drop - 0.05 or first_drop - t < min_clip * 0.8):
            role, clip = "hook", hook_clip
            hook_done = True
            # intro shorter than a clip: run the hook through the drop with its kill on it
            force_beat = first_drop if first_drop > t + 0.5 else None
        elif finale_clip and rem <= finale_len + min_clip * 0.75:
            role, clip = "finale", finale_clip
        else:
            want = (m["calm_clip"] - (m["calm_clip"] - m["busy_clip"]) * e) / p
            must_reach = not hook_done and first_drop - t <= max_clip
            if must_reach:
                want = first_drop - t
            clip = _pick(pool, e, on_drop, [s["video"] for s in slots], m["max_same_source_in_row"],
                         want, max_clip, must_reach)
            if clip is None:
                if finale_clip:
                    role, clip = "finale", finale_clip
                else:
                    break
            else:
                role = "drop" if on_drop else "peak" if e >= 0.62 else "build" if t < first_drop else "breather"

        usable = _usable(clip, max_clip)
        if role == "finale":
            end = T if usable >= rem - 1e-6 else grid.snap(t + usable, t + 1.0, t + usable)
            finale_clip = None
        else:
            want = (m["calm_clip"] - (m["calm_clip"] - m["busy_clip"]) * e) / p
            streak = role in ("hook", "drop", "peak") and _kill_span(clip) + 1.6 > want
            if streak:
                want = _kill_span(clip) + 1.6               # show the whole kill streak
            want = min(max(want, min_clip), max_clip, usable)
            end_t = t + want
            # the intro before the first drop must end exactly on it
            if not hook_done and t < first_drop <= t + usable + 1e-6:
                end_t = first_drop if first_drop - t >= min_clip * 0.5 else end_t
                nxt = first_drop
            else:
                nxt = next((d for d in later_drops if t + min_clip * 0.6 < d <= end_t + 1.5), None)
                if nxt is not None and nxt - t <= usable:
                    end_t = nxt
            # leave room for the finale
            if finale_clip and T - end_t < finale_len:
                end_t = max(t + min_clip * 0.6, T - finale_len)
            if nxt is not None and abs(end_t - nxt) < 1e-6:
                end = nxt
            elif streak:  # round up, so the last kill isn't cut off
                end = grid.snap(end_t, end_t, min(t + usable, T))
            else:
                end = grid.snap(end_t, t + min(min_clip, usable) * 0.6, min(t + usable, T))
            end = min(end, t + usable, T)
            if end - t < 0.8:
                end = min(t + usable, T)

        src_in, key_tl = _place_key(grid, clip, t, end, force_beat)
        dur = end - t
        idx = grid.beats_in(t, end)
        slots.append({
            "id": clip["id"], "role": role, "video": clip["video"], "source": clip["source"],
            "name": clip["name"], "rating": clip["rating"], "tags": clip.get("tags", []),
            "multikill": clip.get("multikill", 0),
            "tl_start": round(t, 3), "tl_end": round(end, 3),
            "src_in": src_in, "src_out": round(src_in + dur, 3),
            "key_src": clip["key_time"], "key_tl": round(key_tl, 3),
            "kills_tl": [round(k - src_in + t, 3) for k in clip.get("kill_times", [])
                         if src_in <= k <= src_in + dur],
            "energy": round(e, 3), "on_drop": role in ("hook", "drop") and (
                any(abs(t - d) < 0.05 for d in grid.drops) or force_beat is not None),
            "beats_tl": [round(float(grid.t[i]), 3) for i in idx],
            "strong_beats_tl": [round(float(grid.t[i]), 3) for i in idx
                                if grid.strength[i] >= 0.6 or grid.down[i]],
        })
        if clip in pool:
            pool.remove(clip)
        t = end
        if role == "finale":
            break

    total = slots[-1]["tl_end"] if slots else 0.0
    if total < T - 0.5:
        log.info("Ran out of clips at %s - the montage ends there (music fades out).", fmt_time(total))
    w, h, fps = cfg.output_size()
    return {
        "duration": round(total, 3), "width": w, "height": h, "fps": fps,
        "platform": cfg.platform_name, "aspect": cfg.platform["aspect"], "style": cfg.style_name,
        "preview": cfg.preview,
        "music": {k: bm[k] for k in ("tracks", "crossfade", "fade_out", "bpm")},
        "drops": bm["drops"], "sections": bm["sections"],
        "clips": [{"i": i, **s} for i, s in enumerate(slots)],
    }


def summarize(edl: dict) -> None:
    log.info("\nTimeline (%s, %d clips, %.0f BPM):", fmt_time(edl["duration"]), len(edl["clips"]),
             edl["music"]["bpm"])
    log.info("  %-4s %-11s %-9s %-5s %-24s %-13s %s", "#", "time", "role", "clip", "recording",
             "source", "kill on beat")
    for c in edl["clips"]:
        on_beat = "yes" if any(abs(k - b) < 0.01 for k in c["kills_tl"] for b in c["beats_tl"]) else "~"
        log.info("  %-4d %-11s %-9s %-5s %-24s %-13s %s", c["i"],
                 f"{c['tl_start']:6.1f}-{c['tl_end']:.1f}", c["role"], c["id"], c["name"][:24],
                 f"{fmt_time(c['src_in'])}-{fmt_time(c['src_out'])}", on_beat)


def run_edl(cfg: Config, clips: list[dict], bm: dict) -> dict:
    edl = build_edl(cfg, clips, bm)
    write_json(cfg.work / "edl.json", edl)
    summarize(edl)
    log.info("Saved %s", cfg.work / "edl.json")
    return edl
