"""Region helper: draw the configured HUD boxes on real frames so you can tune them.

    python -m src.main --stage regions

Writes images to review/regions/. Each image shows a 10% grid (so you can read
off coordinates) and the kill_feed / kill_banner / medals / hitmarker boxes from
config.yaml. Frames come from:
  * your recordings - at detected kill moments if stage 2 has run, otherwise
    spread across each recording
  * any screenshots you put in samples/ (png/jpg)
"""

from __future__ import annotations

import subprocess

import numpy as np

from .config import Config
from .system_check import System
from .utils import fmt_time, log, read_json

COLORS = {  # BGR
    "kill_feed": (0, 200, 255),
    "kill_banner": (0, 0, 255),
    "medals": (0, 215, 255),
    "hitmarker": (255, 255, 255),
}
WIDTH = 1280
FRAMES_PER_VIDEO = 4


def grab_frame(system: System, src: str, t: float, width: int = WIDTH) -> np.ndarray | None:
    """One frame from the original recording at time t, without decoding the whole file."""
    import cv2

    cmd = [system.ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", src,
           "-frames:v", "1", "-vf", f"scale='min({width},iw)':-2", "-f", "image2pipe",
           "-vcodec", "png", "-"]
    res = subprocess.run(cmd, capture_output=True, timeout=60)
    if res.returncode != 0 or not res.stdout:
        return None
    return cv2.imdecode(np.frombuffer(res.stdout, np.uint8), cv2.IMREAD_COLOR)


def draw_regions(img: np.ndarray, regions: dict, title: str) -> np.ndarray:
    import cv2

    out = img.copy()
    h, w = out.shape[:2]
    overlay = out.copy()
    for i in range(1, 10):  # 10% grid
        x, y = int(w * i / 10), int(h * i / 10)
        cv2.line(overlay, (x, 0), (x, h), (200, 200, 200), 1)
        cv2.line(overlay, (0, y), (w, y), (200, 200, 200), 1)
        cv2.putText(overlay, f"{i/10:.1f}", (x + 2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(overlay, f"{i/10:.1f}", (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)

    for name, (rx, ry, rw, rh) in regions.items():
        color = COLORS.get(name, (0, 255, 0))
        p0 = (int(rx * w), int(ry * h))
        p1 = (int((rx + rw) * w) - 1, int((ry + rh) * h) - 1)
        cv2.rectangle(out, p0, p1, color, 2)
        label = f"{name} [{rx:.2f}, {ry:.2f}, {rw:.2f}, {rh:.2f}]"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ty = p0[1] - 6 if p0[1] > th + 10 else p1[1] + th + 8
        tx = min(p0[0] + 2, w - tw - 6)
        cv2.rectangle(out, (tx - 3, ty - th - 4), (tx + tw + 3, ty + 4), (0, 0, 0), -1)
        cv2.putText(out, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    cv2.rectangle(out, (0, h - 24), (w, h), (0, 0, 0), -1)
    cv2.putText(out, title, (8, h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def _pick_times(cfg: Config, meta: dict) -> list[tuple[float, str]]:
    cands = read_json(cfg.work / "candidates.json", {}).get("candidates", [])
    kill_times = []
    for c in cands:
        if c["video"] == meta["id"]:
            kill_times += c["kill_times"]
    picked = [(t + 0.3, "detected kill") for t in kill_times[:FRAMES_PER_VIDEO - 1]]
    dur = meta["info"]["duration"]
    spread = [dur * f for f in (0.15, 0.4, 0.65, 0.9)]
    for t in spread:
        if len(picked) >= FRAMES_PER_VIDEO:
            break
        picked.append((t, "sample"))
    return picked


def regions_helper(cfg: Config, system: System, videos: list[dict]) -> None:
    import cv2

    out_dir = cfg.path("review") / "regions"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.jpg"):
        old.unlink()

    regions = cfg.regions
    written = 0
    for meta in videos:
        for t, why in _pick_times(cfg, meta):
            img = grab_frame(system, meta["source"], t)
            if img is None:
                log.warning("  couldn't grab a frame from %s at %s", meta["name"], fmt_time(t))
                continue
            title = f"{meta['name']}  @ {fmt_time(t)}  ({why})  game preset: {cfg.game}"
            name = f"{meta['id']}_{int(t):05d}.jpg"
            cv2.imwrite(str(out_dir / name), draw_regions(img, regions, title),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            written += 1

    samples = cfg.root / "samples"
    if samples.is_dir():
        for p in sorted(samples.iterdir()):
            if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                continue
            img = cv2.imread(str(p))
            if img is None:
                continue
            if img.shape[1] > WIDTH:
                img = cv2.resize(img, (WIDTH, int(img.shape[0] * WIDTH / img.shape[1])),
                                 interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / f"sample_{p.stem}.jpg"),
                        draw_regions(img, regions, f"samples/{p.name}  game preset: {cfg.game}"),
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            written += 1

    log.info("Wrote %d image(s) to %s", written, out_dir)
    log.info(
        "Open them and check each box covers its HUD element:\n"
        "  kill_feed   - where 'You killed X' / the kill list appears\n"
        "  kill_banner - the skull / '+100' popup after a kill\n"
        "  medals      - multi-kill / streak medals\n"
        "  hitmarker   - the X that flashes around the crosshair\n"
        "Read new values off the 0.1 grid and edit  regions: %s:  in config.yaml as\n"
        "[x, y, width, height]. Re-run this command to check, then  --stage detect.",
        cfg.game,
    )
