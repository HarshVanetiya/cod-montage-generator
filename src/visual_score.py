"""Stage 2b - on-screen signals from the low-res proxy.

One streaming pass over the proxy (one frame in memory at a time) records small
per-frame features:

  * per HUD region (kill feed, kill banner, medals, hitmarker):
      - stable-edge fraction: edges that stayed put for the last few frames.
        HUD overlays are static while the game world moves, so a jump in
        stable edges means "a new HUD element just appeared" (a kill-feed row,
        a skull, a medal) - without needing game-specific templates.
      - red / gold / white pixel fractions (skulls, medals, "+100" text,
        hitmarkers)
  * whole-frame motion, brightness, contrast and saturation, used to find
    menus, loading screens and death screens.

Events (kills, hitmarkers) are then derived from these features with
thresholds from config.yaml, which is fast - so changing thresholds does
not require re-reading any video.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .utils import MontageError

REGION_NAMES = ["kill_feed", "kill_banner", "medals", "hitmarker"]
# Per-region features, stored as f"{region}_{feature}"
REGION_FEATURES = ["stable", "red", "gold", "white"]
GLOBAL_FEATURES = ["motion", "bright", "contrast", "sat", "gstable"]
STABLE_FRAMES = 3          # an edge must persist this many frames to count as HUD
SMALL = (160, 90)          # whole-frame analysis size (landscape)


def _region_px(box: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    x, y, bw, bh = box
    x0, y0 = int(x * w), int(y * h)
    x1, y1 = max(x0 + 2, int((x + bw) * w)), max(y0 + 2, int((y + bh) * h))
    return x0, y0, min(x1, w), min(y1, h)


def _color_fracs(hsv: np.ndarray) -> tuple[float, float, float]:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    vivid = (s > 120) & (v > 120)
    red = vivid & ((h < 8) | (h > 172))
    gold = vivid & (h >= 15) & (h <= 35) & (v > 150)
    white = (s < 45) & (v > 215)
    n = h.size
    return red.sum() / n, gold.sum() / n, white.sum() / n


def frame_features(proxy: Path, regions: dict[str, list[float]], label: str = "") -> dict:
    """Single pass over the proxy. Returns {feature_name: float32 array per frame}."""
    import cv2

    cap = cv2.VideoCapture(str(proxy))
    if not cap.isOpened():
        raise MontageError(f"OpenCV can't open the proxy {proxy}. Delete work/ and run again.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    boxes = {name: _region_px(regions[name], w, h) for name in REGION_NAMES if name in regions}
    small_size = SMALL if w >= h else (SMALL[1], SMALL[0])

    cols: dict[str, list[float]] = {f"{r}_{f}": [] for r in boxes for f in REGION_FEATURES}
    cols.update({g: [] for g in GLOBAL_FEATURES})
    edge_hist = {r: deque(maxlen=STABLE_FRAMES) for r in list(boxes) + ["_global"]}
    prev_small = None

    with tqdm(total=total, unit="fr", desc=f"  video {label}"[:34], leave=False,
              dynamic_ncols=True) as bar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # ---- whole frame
            small = cv2.resize(gray, small_size, interpolation=cv2.INTER_AREA)
            cols["motion"].append(0.0 if prev_small is None
                                  else float(cv2.absdiff(small, prev_small).mean()))
            prev_small = small
            cols["bright"].append(float(small.mean()))
            cols["contrast"].append(float(small.std()))
            small_hsv = cv2.cvtColor(cv2.resize(frame, small_size, interpolation=cv2.INTER_AREA),
                                     cv2.COLOR_BGR2HSV)
            cols["sat"].append(float(small_hsv[..., 1].mean()))
            g_edges = cv2.Canny(small, 80, 160) > 0
            edge_hist["_global"].append(g_edges)
            cols["gstable"].append(_stable_frac(edge_hist["_global"]))

            # ---- HUD regions
            for name, (x0, y0, x1, y1) in boxes.items():
                crop = frame[y0:y1, x0:x1]
                edges = cv2.Canny(gray[y0:y1, x0:x1], 80, 160) > 0
                edge_hist[name].append(edges)
                red, gold, white = _color_fracs(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV))
                cols[f"{name}_stable"].append(_stable_frac(edge_hist[name]))
                cols[f"{name}_red"].append(red)
                cols[f"{name}_gold"].append(gold)
                cols[f"{name}_white"].append(white)
            bar.update(1)
    cap.release()

    out = {k: np.asarray(v, np.float32) for k, v in cols.items()}
    out["fps"] = np.float32(fps)
    return out


def _stable_frac(hist: deque) -> float:
    if len(hist) < hist.maxlen:
        return 0.0
    stable = hist[0].copy()
    for e in list(hist)[1:]:
        stable &= e
    return float(stable.mean())


# ------------------------------------------------------------------ events

def _trailing(x: np.ndarray, k: int, fn) -> np.ndarray:
    """fn over the previous k values (not including the current one)."""
    if len(x) == 0:
        return x
    padded = np.concatenate([np.full(k, x[0]), x])
    win = np.lib.stride_tricks.sliding_window_view(padded, k)[:-1]
    return fn(win, axis=1)


def _robust_z(x: np.ndarray, floor: float) -> np.ndarray:
    med = np.median(x)
    scale = max(np.median(np.abs(x - med)) * 1.4826, floor)
    return (x - med) / scale


def _peaks(signal: np.ndarray, thr: float, min_gap: int) -> list[int]:
    """Local maxima above thr, at least min_gap frames apart."""
    from scipy.ndimage import maximum_filter1d

    if len(signal) == 0:
        return []
    local = signal == maximum_filter1d(signal, size=2 * min_gap + 1, mode="nearest")
    picked: list[int] = []
    for i in np.flatnonzero(local & (signal >= thr)):
        if not picked or i - picked[-1] >= min_gap:
            picked.append(int(i))
    return picked


def detect_events(feats: dict, det: dict) -> dict:
    """Kill / hitmarker events and per-frame exclusion flags from frame features."""
    fps = float(feats["fps"])
    n = len(feats["motion"])
    half = max(1, int(round(0.5 * fps)))
    one = max(1, int(round(fps)))

    # How much the whole picture "froze" - subtracted so standing still
    # doesn't look like HUD popping up everywhere.
    g_rise = np.maximum(0, feats["gstable"] - _trailing(feats["gstable"], half, np.min))

    kill = np.zeros(n, np.float32)
    per_region = {}
    for r in ("kill_feed", "kill_banner", "medals"):
        if f"{r}_stable" not in feats:
            continue
        st = feats[f"{r}_stable"]
        appear = np.maximum(0, st - _trailing(st, half, np.min) - g_rise)
        color = feats[f"{r}_red"] + feats[f"{r}_gold"]
        color_pop = np.maximum(0, color - _trailing(color, one, np.median))
        z = np.maximum(_robust_z(appear, 0.01), _robust_z(color_pop, 0.004))
        per_region[r] = z
        kill = np.maximum(kill, z)

    hit = np.zeros(n, np.float32)
    if "hitmarker_white" in feats:
        wh = feats["hitmarker_white"] + feats["hitmarker_red"]
        hit = _robust_z(np.maximum(0, wh - _trailing(wh, half, np.median)), 0.003)

    excluded = exclusion_flags(feats, det)
    kill[excluded] = 0
    hit[excluded] = 0

    gap = max(1, int(round(det["event_debounce"] * fps)))
    kill_idx = _peaks(kill, det["kill_z"], gap)
    hit_idx = _peaks(hit, det["hitmarker_z"], max(1, int(0.25 * fps)))

    def strength(z: float, thr: float) -> float:
        return round(float(min(z / thr, 3.0)), 3)

    return {
        "fps": fps,
        "kills": [{"t": round(i / fps, 2), "strength": strength(kill[i], det["kill_z"]),
                   "region": max(per_region, key=lambda r: per_region[r][i]) if per_region else ""}
                  for i in kill_idx],
        "hits": [{"t": round(i / fps, 2), "strength": strength(hit[i], det["hitmarker_z"])}
                 for i in hit_idx],
        "excluded": excluded,
    }


def exclusion_flags(feats: dict, det: dict) -> np.ndarray:
    """Per-frame True where the footage is a menu / loading / frozen / death screen."""
    fps = float(feats["fps"])
    n = len(feats["motion"])
    dark = feats["bright"] < det["dark_level"]
    flat = feats["contrast"] < det["uniform_std"]

    # Frozen screens (menus, lobbies, pause, scoreboards): very little motion
    # for a sustained stretch. Gameplay almost never holds perfectly still.
    still = feats["motion"] < det["static_motion"]
    static = _sustained(still, int(det["static_min_seconds"] * fps))

    # Death / spectate screens in CoD are usually washed-out grey.
    sat = feats["sat"]
    med_sat = np.median(sat) if n else 0
    desat = _sustained(sat < det["desat_ratio"] * med_sat, int(1.0 * fps))

    return dark | flat | static | desat


def _sustained(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Keep only True-runs at least min_len long."""
    out = np.zeros_like(mask)
    if min_len <= 1:
        return mask.copy()
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def scene_cuts(proxy: Path, threshold: float) -> list[float]:
    """Hard cuts (gameplay -> killcam, -> menu, respawn) via PySceneDetect."""
    from scenedetect import ContentDetector, detect

    scenes = detect(str(proxy), ContentDetector(threshold=threshold, min_scene_len=5),
                    show_progress=False)

    def secs(tc) -> float:
        return float(tc.seconds) if hasattr(tc, "seconds") else float(tc.get_seconds())

    return [round(secs(start), 2) for start, _ in scenes[1:]]
