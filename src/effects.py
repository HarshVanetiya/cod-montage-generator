"""Stage 5a - decide the effects for every clip (work/effects.json).

Rules (from CLAUDE.md, tuned per style in config.yaml):
  * slow motion only on the kill moment of top clips (hook, finale, drops first),
    ramping in and out; the kill stays exactly on its beat
  * a riser slow-mo on the last moment before the first drop, snapping back on the drop
  * zoom punch on a big beat (the kill beat when it is on one), small amount
  * shake only when the style has it (hype), on drops and multi-kills
  * flash / whip / blur transitions only on drops; hard cuts everywhere else
  * the same effect never on two clips in a row
  * text popups only for real multi-kills in styles that allow them
  * music dropout right before up to `max_dropouts` big kills
Everything here is timing data; render.py turns it into FFmpeg filters.
"""

from __future__ import annotations

from .config import Config
from .utils import log, write_json

POPUP_TEXT = {2: "DOUBLE KILL", 3: "TRIPLE KILL", 4: "QUAD KILL"}
RAMP = 0.22                 # seconds of half-speed ramp on each side of a slow-mo


def _speed_pieces(dur: float, ks: float, style: dict, riser: bool) -> list[list[float]] | None:
    """Timeline pieces [[duration, speed], ...] covering 0..dur, or None for normal speed."""
    s = float(style["slowmo_speed"])
    mid = (1 + s) / 2
    if riser:  # last stretch before the drop slows down, the cut snaps back to normal
        slow = min(1.2, dur / 3)
        if slow < 0.5:
            return None
        return [[dur - slow - RAMP, 1.0], [RAMP, mid], [slow, s]] if dur - slow - RAMP > 0.3 \
            else [[dur - slow, 1.0], [slow, s]]
    length = float(style["slowmo_length"])
    a = ks - 0.35 * length                     # kill sits a third of the way into the slow part
    b = a + length
    a, b = max(a, RAMP + 0.3), min(b, dur - RAMP - 0.2)
    if b - a < 0.6:
        return None
    pieces = [[a - RAMP, 1.0], [RAMP, mid], [b - a, s], [RAMP, mid], [dur - b - RAMP, 1.0]]
    return [[round(d, 3), v] for d, v in pieces if d > 1e-3]


def _zoom_beat(c: dict) -> float | None:
    for k in c["kills_tl"]:  # a kill that's on a beat is the best place to punch in
        if any(abs(k - b) < 0.01 for b in c["beats_tl"]):
            return k
    inner = [b for b in c["strong_beats_tl"] if c["tl_start"] + 0.3 < b < c["tl_end"] - 0.4]
    return inner[0] if inner else None


def plan_effects(cfg: Config, edl: dict) -> dict:
    style = cfg.style
    rnd = cfg["render"]
    clips = edl["clips"]
    n = len(clips)
    landscape = edl["aspect"] == "landscape"
    first_drop = edl["drops"][0]["t"] if edl["drops"] else None

    plan = [{"i": c["i"], "pieces": None, "zoom": [], "shake": [], "transition": "cut",
             "popup": None, "riser": False} for c in clips]

    # ---- slow motion on kills: best roles and ratings first
    budget = round(style["slowmo_share"] * n) if style["slowmo_share"] > 0 else 0
    role_rank = {"hook": 0, "finale": 1, "drop": 2, "peak": 3}
    cands = sorted((c for c in clips if c["kills_tl"] and c["role"] in role_rank and c["rating"] >= 5),
                   key=lambda c: (role_rank[c["role"]], -c["rating"]))
    slowed: set[int] = set()
    for c in cands:
        if len(slowed) >= budget:
            break
        if c["i"] - 1 in slowed or c["i"] + 1 in slowed:
            continue                                  # never on consecutive clips
        dur = c["tl_end"] - c["tl_start"]
        pieces = _speed_pieces(dur, c["key_tl"] - c["tl_start"], style, riser=False)
        if pieces:
            plan[c["i"]]["pieces"] = pieces
            slowed.add(c["i"])

    # ---- riser: the clip that ends on the first drop slows into it
    if style["slowmo_share"] > 0 and first_drop is not None:
        for c in clips:
            if abs(c["tl_end"] - first_drop) < 0.05 and c["i"] not in slowed and c["role"] == "build":
                pieces = _speed_pieces(c["tl_end"] - c["tl_start"], 0, style, riser=True)
                if pieces:
                    plan[c["i"]]["pieces"] = pieces
                    plan[c["i"]]["riser"] = True

    # ---- zoom punches (not on slowed clips, never twice in a row)
    if style["zoom_punch"] > 1.0 and style["zoom_share"] > 0:
        budget = max(1, round(style["zoom_share"] * n))
        order = sorted(clips, key=lambda c: (not c["on_drop"], -c["rating"]))
        zoomed: set[int] = set()
        for c in order:
            if len(zoomed) >= budget:
                break
            if plan[c["i"]]["pieces"] or c["i"] - 1 in zoomed or c["i"] + 1 in zoomed:
                continue
            b = _zoom_beat(c)
            if b is not None:
                plan[c["i"]]["zoom"] = [round(b - c["tl_start"], 3)]
                zoomed.add(c["i"])

    # ---- shake (hype): first two strong beats of drop / multi-kill clips
    if style["shake"] > 0:
        for c in clips:
            if (c["on_drop"] or c["multikill"] >= 2) and not plan[c["i"]]["pieces"]:
                beats = [b - c["tl_start"] for b in c["strong_beats_tl"]
                         if c["tl_start"] + 0.1 < b < c["tl_end"] - 0.3][:2]
                plan[c["i"]]["shake"] = [round(b, 3) for b in beats]

    # ---- transitions only on drops, cycling through the style's list
    kinds = [t for t in style["transitions"] if t in ("flash", "whip", "blur")] or ["flash"]
    k = 0
    for c in clips:
        if c["on_drop"] and c["i"] > 0:
            plan[c["i"]]["transition"] = kinds[k % len(kinds)]
            k += 1

    # ---- text popups on real multi-kills
    if style["text_popups"]:
        for c in clips:
            kills = c["kills_tl"]
            if c["multikill"] >= 2 and len(kills) >= 2:
                text = POPUP_TEXT.get(min(len(kills), 4), "MULTI KILL")
                plan[c["i"]]["popup"] = {"text": text, "t": round(kills[-1] - c["tl_start"], 3)}

    # ---- music dropouts right before the biggest kills
    dropouts = []
    if style["music_dropout"]:
        ranked = sorted((c for c in clips if c["kills_tl"] and c["role"] in ("hook", "finale", "drop")),
                        key=lambda c: (-c["multikill"], role_rank.get(c["role"], 9)))
        for c in ranked[: int(rnd["max_dropouts"])]:
            t = c["key_tl"]
            a = t - float(rnd["dropout_seconds"])
            if a > 0.5:
                dropouts.append([round(a, 3), round(t, 3)])
        dropouts.sort()

    title = None
    name = str(cfg["gamer_name"] or "").strip()
    if rnd["title_card"] is True or (rnd["title_card"] == "auto" and name and name.lower() != "player"):
        title = name

    effects = {
        "style": cfg.style_name,
        "grade": {"contrast": style["contrast"], "saturation": style["saturation"],
                  "vignette": bool(style["vignette"])},
        "letterbox": bool(style["letterbox"]) and landscape,
        "title": title,
        "dropouts": dropouts,
        "clips": plan,
    }
    used = {
        "slow-mo": sum(1 for p in plan if p["pieces"] and not p["riser"]),
        "riser": sum(1 for p in plan if p["riser"]),
        "zoom": sum(1 for p in plan if p["zoom"]),
        "shake": sum(1 for p in plan if p["shake"]),
        "transitions": sum(1 for p in plan if p["transition"] != "cut"),
        "popups": sum(1 for p in plan if p["popup"]),
        "dropouts": len(dropouts),
    }
    log.info("Effects (%s): %s", cfg.style_name, ", ".join(f"{v} {k}" for k, v in used.items() if v) or "none")
    write_json(cfg.work / "effects.json", effects)
    return effects
