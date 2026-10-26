"""Stage 5b - render the montage with FFmpeg.

Low-memory design (8 GB PCs):
  1. Each timeline clip is rendered on its own into work/render/seg_<hash>.mkv
     (video encoded once, game audio as PCM). Only that short stretch of the
     recording is decoded; frames stream through the filters one at a time.
     Speed ramps use one piecewise setpts expression instead of split+trim, so
     FFmpeg never buffers a clip's frames. Segments are cached by content,
     so changing one clip only re-renders that clip.
  2. The audio mix (game audio ducked under the music, music dropouts, fades)
     is built as a WAV, measured, then loudness-normalised (two-pass loudnorm).
  3. Segments are joined with the concat demuxer (video stream copied, no
     second encode) and muxed with the normalised AAC audio.
Output: output/montage_YYYYMMDD_HHMM[_platform][_preview].mp4
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from .config import Config
from .music import remember_used
from .system_check import System
from .utils import MontageError, fmt_size, fmt_time, log, read_json, run, run_ffmpeg_progress

RENDER_VERSION = 1
SR = 48000
FONT_CANDIDATES = [
    "C:/Windows/Fonts/impact.ttf", "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/segoeuib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
]
OVERLAY_VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm")


# ------------------------------------------------------------------ small helpers

def ff_path(p: Path | str) -> str:
    """A file path as a quoted filter option value (handles Windows drive colons)."""
    s = str(p).replace("\\", "/").replace(":", r"\:").replace("'", r"\'")
    return f"'{s}'"


def concat_line(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/").replace("'", r"'\''")
    return f"file '{s}'"


def find_font(cfg: Config) -> str | None:
    custom = str(cfg["render"].get("font") or "").strip()
    if custom:
        if Path(custom).exists():
            return custom
        log.warning("render.font %s doesn't exist - looking for a system font instead.", custom)
    return next((f for f in FONT_CANDIDATES if Path(f).exists()), None)


def find_overlay(cfg: Config, stem: str, exts: tuple[str, ...]) -> Path | None:
    folder = cfg.path("overlays")
    if not folder.is_dir():
        return None
    return next((p for p in sorted(folder.iterdir()) if p.stem.lower() == stem and p.suffix.lower() in exts),
                None)


def _font_size(text: str, W: int, H: int, frac: float) -> int:
    """frac of the height, shrunk so the text fits 90% of the width (bold glyphs ~0.62 em)."""
    return int(min(H * frac, 0.9 * W / (0.62 * max(len(text), 1))))


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:14]


def _text_file(cfg: Config, text: str) -> Path:
    """drawtext reads text from a file so no character in a name can break the filter."""
    d = cfg.work / "render"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"text_{_hash(text)}.txt"
    if not p.exists():
        p.write_text(text, encoding="utf-8")
    return p


# ------------------------------------------------------------------ time mapping

def source_plan(clip: dict, pieces: list[list[float]] | None, dur: float) -> tuple[float, list]:
    """Where to start reading the recording, and pieces as [src_len, speed]
    so the clip's key moment still lands at key_tl on the timeline."""
    if not pieces:
        return max(0.0, clip["src_in"]), [[dur, 1.0]]
    ks = clip["key_tl"] - clip["tl_start"]
    before, acc = 0.0, 0.0
    for d, v in pieces:  # source seconds consumed before the key moment
        if acc + d <= ks:
            before += d * v
        elif acc < ks:
            before += (ks - acc) * v
        acc += d
    # stretch the last piece so rounding never leaves the timeline short
    pieces = [list(p) for p in pieces]
    pieces[-1][0] += max(0.0, dur - sum(d for d, _ in pieces))
    start = clip["key_src"] - before
    return max(0.0, start), [[d * v, v] for d, v in pieces]


def setpts_expr(src_pieces: list) -> str:
    """Piecewise source-time -> output-time mapping as one setpts expression."""
    if len(src_pieces) == 1 and src_pieces[0][1] == 1.0:
        return "PTS-STARTPTS"
    T = "((PTS-STARTPTS)*TB)"
    expr, s_acc, o_acc, bounds = None, 0.0, 0.0, []
    for src_len, v in src_pieces:
        bounds.append((s_acc, o_acc, v))
        s_acc += src_len
        o_acc += src_len / v
    # build from the last piece backwards: if(lt(T,S_i), <earlier>, O_i+(T-S_i)/v_i)
    for k in range(len(bounds) - 1, -1, -1):
        s, o, v = bounds[k]
        piece = f"({o:.5f}+({T}-{s:.5f})/{v:.5f})"
        expr = piece if expr is None else f"if(lt({T},{bounds[k + 1][0]:.5f}),{piece},{expr})"
    return f"({expr})/TB"


def _atempo(v: float) -> str:
    parts = []
    while v < 0.5:
        parts.append("atempo=0.5")
        v /= 0.5
    parts.append(f"atempo={v:.4f}")
    return ",".join(parts)


# ------------------------------------------------------------------ video filters

def _fit(W: int, H: int, aspect: str, fit: str) -> str:
    if aspect == "center_crop":  # vertical / 4:5 - CoD's crosshair is centred
        return (f"crop='min(iw,ih*{W}/{H})':'min(ih,iw*{H}/{W})',"
                f"scale={W}:{H}:flags=bicubic,setsar=1")
    if fit == "pad":
        return (f"scale={W}:{H}:force_original_aspect_ratio=decrease:flags=bicubic,"
                f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1")
    return f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=bicubic,crop={W}:{H},setsar=1"


def _motion(W: int, H: int, zoom: list[float], amount: float, shake: list[float], px: float) -> str:
    """Zoom punches and shake as one per-frame scale + fixed-size crop."""
    terms = []
    for b in zoom:
        u = f"(t-{b:.3f})"
        terms.append(f"{amount - 1:.4f}*if(lt({u},-0.06),0,if(lt({u},0),({u}+0.06)/0.06,max(0,1-{u}/0.35)))")
    margin = 1 + (2.5 * px / W if shake else 0)
    z = f"{margin:.4f}*(1+{'+'.join(terms) if terms else '0'})"
    sx = "+".join(f"{px:.1f}*sin(2*PI*22*(t-{b:.3f}))*max(0,1-(t-{b:.3f})/0.25)*gte(t,{b:.3f})"
                  for b in shake) or "0"
    sy = "+".join(f"{px * 0.6:.1f}*cos(2*PI*17*(t-{b:.3f}))*max(0,1-(t-{b:.3f})/0.25)*gte(t,{b:.3f})"
                  for b in shake) or "0"
    return (f"scale=w='trunc({W}*{z}/2)*2':h='trunc({H}*{z}/2)*2':eval=frame:flags=bicubic,"
            f"crop={W}:{H}:x='(in_w-{W})/2+{sx}':y='(in_h-{H})/2+{sy}'")


def video_chain(cfg: Config, eff: dict, plan: dict, src_pieces: list, W: int, H: int, fps: int,
                dur: float, aspect: str, first: bool, last: bool, font: str | None,
                logo: bool) -> str:
    style, rnd = cfg.style, cfg["render"]
    # tpad: if the recording ends a few frames early, hold the last frame
    f = [f"setpts='{setpts_expr(src_pieces)}'", f"fps={fps}", "tpad=stop_mode=clone:stop_duration=2",
         _fit(W, H, aspect, rnd["landscape_fit"])]
    g = eff["grade"]
    f.append(f"eq=contrast={g['contrast']}:saturation={g['saturation']}")
    if g["vignette"]:
        f.append("vignette=angle=PI/5")
    if plan["zoom"] or plan["shake"]:
        f.append(_motion(W, H, plan["zoom"], float(style["zoom_punch"]), plan["shake"],
                         float(style["shake"]) * W / 1920))
    tr = plan["transition"]
    if tr == "flash":
        f.append("fade=t=in:st=0:d=0.25:color=white")
    elif tr == "blur":
        f.append("gblur=sigma=18:steps=2:enable='lt(t,0.14)'")
    elif tr == "whip":
        f.append("avgblur=sizeX=60:sizeY=1:enable='lt(t,0.14)'")
    if font and plan.get("popup"):
        p = plan["popup"]
        a = p["t"]
        f.append(f"drawtext=fontfile={ff_path(font)}:textfile={ff_path(_text_file(cfg, p['text']))}:"
                 f"fontsize={_font_size(p['text'], W, H, 0.085)}:fontcolor=0xFFD34D:borderw={max(2, H // 200)}:bordercolor=black:"
                 f"x=(w-text_w)/2:y=h*0.2:enable='between(t,{a:.3f},{a + 1.1:.3f})':"
                 f"alpha='if(lt(t-{a:.3f},0.08),(t-{a:.3f})/0.08,if(gt(t-{a:.3f},0.85),max(0,(1.1-(t-{a:.3f}))/0.25),1))'")
    if font and first and eff["title"]:
        end = min(3.0, dur - 0.3)
        f.append(f"drawtext=fontfile={ff_path(font)}:textfile={ff_path(_text_file(cfg, eff['title']))}:"
                 f"fontsize={_font_size(eff['title'], W, H, 0.075)}:fontcolor=white:borderw={max(2, H // 300)}:bordercolor=black@0.6:"
                 f"x=(w-text_w)/2:y=(h-text_h)/2:enable='lt(t,{end:.3f})':"
                 f"alpha='min(1,min(t/0.4,max(0,({end:.3f}-t)/0.5)))'")
    if eff["letterbox"]:
        bar = int((H - W / float(rnd["letterbox_ratio"])) / 2)
        if bar > 2:
            f.append(f"drawbox=x=0:y=0:w=iw:h={bar}:color=black:t=fill,"
                     f"drawbox=x=0:y=ih-{bar}:w=iw:h={bar}:color=black:t=fill")
    if first:
        f.append(f"fade=t=in:st=0:d={float(rnd['fade_in']):.2f}")
    if last:
        fo = min(float(rnd["fade_out"]), dur / 2)
        f.append(f"fade=t=out:st={dur - fo:.3f}:d={fo:.3f}")
    chain = "[0:v]" + ",".join(f)
    if logo:
        size, op = float(rnd["logo_size"]), float(rnd["logo_opacity"])
        chain += (f"[base];[1:v]scale={int(W * size)}:-1,format=rgba,colorchannelmixer=aa={op}[lg];"
                  f"[base][lg]overlay=W-w-{int(W * 0.02)}:{int(W * 0.02)}:shortest=1")
    return chain + ",format=yuv420p[v]"


def audio_chain(src_pieces: list, dur: float, has_audio: bool) -> str:
    tail = f"aresample={SR},aformat=sample_fmts=fltp:channel_layouts=stereo,apad,atrim=0:{dur:.4f}[a]"
    if not has_audio:
        return f"anullsrc=r={SR}:cl=stereo,atrim=0:{dur:.4f}[a]"
    if len(src_pieces) == 1:
        return f"[0:a]atrim=0:{src_pieces[0][0]:.4f},asetpts=PTS-STARTPTS,{tail}"
    k = len(src_pieces)
    parts = [f"[0:a]asplit={k}" + "".join(f"[s{i}]" for i in range(k))]
    s = 0.0
    for i, (src_len, v) in enumerate(src_pieces):
        chain = f"[s{i}]atrim={s:.4f}:{s + src_len:.4f},asetpts=PTS-STARTPTS"
        if v != 1.0:
            chain += f",{_atempo(v)},volume=0.7"   # slowed game audio sits a bit lower
        parts.append(chain + f"[p{i}]")
        s += src_len
    parts.append("".join(f"[p{i}]" for i in range(k)) + f"concat=n={k}:v=0:a=1,{tail}")
    return ";".join(parts)


# ------------------------------------------------------------------ segment rendering

def _encode(system: System, cfg: Config, fps: int) -> list[str]:
    q = int(cfg["render"]["quality"]) + (3 if cfg.preview else 0)
    return system.encode_args(q, fast=cfg.preview) + ["-g", str(fps * 2), "-r", str(fps)]


def _run_segment(system: System, cmd_for_hw, out: Path, on_progress) -> None:
    tmp = out.with_name(out.stem + ".tmp.mkv")
    last = ""
    for hw in ([True, False] if system.hw_decode else [False]):
        code, err = run_ffmpeg_progress(cmd_for_hw(hw, tmp), on_progress)
        if code == 0 and tmp.exists() and tmp.stat().st_size > 0:
            os.replace(tmp, out)
            return
        last = err.strip()
        tmp.unlink(missing_ok=True)
        log.debug("segment failed (hw=%s):\n%s", hw, last)
    raise MontageError(f"FFmpeg failed while rendering a clip:\n{last[-1200:]}")


def render_segments(cfg: Config, system: System, edl: dict, eff: dict, W: int, H: int, fps: int,
                    intro: Path | None, outro: Path | None) -> tuple[list[Path], float]:
    seg_dir = cfg.work / "render"
    seg_dir.mkdir(parents=True, exist_ok=True)
    font = find_font(cfg)
    if not font and (eff["title"] or any(p["popup"] for p in eff["clips"])):
        log.warning("No font found for titles/popups - they'll be skipped (set render.font).")
    logo = find_overlay(cfg, "logo", (".png",))
    enc = _encode(system, cfg, fps)
    clips, plans = edl["clips"], eff["clips"]

    jobs = []  # (label, seconds, out_path, cmd_builder)
    intro_dur = 0.0
    for kind, path in (("intro", intro), ("outro", outro)):
        if not path:
            continue
        info = json.loads(run([system.ffprobe, "-v", "error", "-print_format", "json", "-show_format",
                               "-show_streams", str(path)]).stdout or "{}")
        d = float(info.get("format", {}).get("duration", 0) or 0)
        if d <= 0:
            log.warning("Can't read overlays/%s - skipping it.", path.name)
            continue
        has_a = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
        st = path.stat()
        out = seg_dir / f"{kind}_{_hash([RENDER_VERSION, path.name, st.st_size, st.st_mtime_ns, W, H, fps, enc])}.mkv"
        n_frames = max(1, round(d * fps))
        vf = (f"[0:v]fps={fps},{_fit(W, H, 'landscape', 'pad')},format=yuv420p[v];"
              + audio_chain([[d, 1.0]], n_frames / fps, has_a))

        def build(hw, tmp, path=path, vf=vf, n_frames=n_frames):
            return ([system.ffmpeg, "-y", "-loglevel", "error"] + (system.decode_args() if hw else [])
                    + ["-i", str(path), "-filter_complex", vf, "-map", "[v]", "-map", "[a]",
                       "-frames:v", str(n_frames)] + enc + ["-c:a", "pcm_s16le", str(tmp)])
        jobs.append((kind, n_frames / fps, out, build))
        if kind == "intro":
            intro_dur = n_frames / fps

    main_jobs = []
    for idx, (c, plan) in enumerate(zip(clips, plans)):
        f0, f1 = round(c["tl_start"] * fps), round(c["tl_end"] * fps)
        n_frames = max(1, f1 - f0)
        dur = n_frames / fps
        start, src_pieces = source_plan(c, plan["pieces"], dur)
        src_len = sum(p[0] for p in src_pieces)
        meta = read_json(cfg.work / "videos" / c["video"] / "meta.json", {})
        has_audio = meta.get("info", {}).get("has_audio", True)
        first, last = idx == 0, idx == len(clips) - 1
        vf = video_chain(cfg, eff, plan, src_pieces, W, H, fps, dur, edl["aspect"],
                         first and not intro, last and not outro, font, bool(logo))
        graph = vf + ";" + audio_chain(src_pieces, dur, has_audio)
        key = _hash([RENDER_VERSION, c["source"], start, src_len, graph, n_frames, enc,
                     Path(c["source"]).stat().st_mtime_ns])
        out = seg_dir / f"seg_{key}.mkv"

        def build(hw, tmp, c=c, start=start, src_len=src_len, graph=graph, n_frames=n_frames):
            cmd = [system.ffmpeg, "-y", "-loglevel", "error"] + (system.decode_args() if hw else [])
            cmd += ["-ss", f"{start:.3f}", "-t", f"{src_len + 0.5:.3f}", "-i", c["source"]]
            if logo:
                cmd += ["-loop", "1", "-i", str(logo)]
            return cmd + ["-filter_complex", graph, "-map", "[v]", "-map", "[a]",
                          "-frames:v", str(n_frames)] + enc + ["-c:a", "pcm_s16le", str(tmp)]
        main_jobs.append((f"clip {idx + 1}", dur, out, build))

    ordered = [j for j in jobs if j[0] == "intro"] + main_jobs + [j for j in jobs if j[0] == "outro"]
    todo = [j for j in ordered if cfg.force or not j[2].exists()]
    if len(todo) < len(ordered):
        log.info("Reusing %d already-rendered clip(s) from work/render/.", len(ordered) - len(todo))
    if todo:
        total = sum(j[1] for j in todo)
        with tqdm(total=round(total, 1), unit="s", dynamic_ncols=True,
                  bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}s video [{elapsed}<{remaining}]") as bar:
            for label, d, out, build in todo:
                bar.set_description(f"  {label}")
                base = bar.n
                done = {"t": 0.0}

                def prog(t, base=base, d=d, done=done):
                    t = min(t, d, bar.total - base)
                    if t > done["t"]:
                        bar.update(t - done["t"])
                        done["t"] = t
                _run_segment(system, build, out, prog)
                bar.n = min(base + d, bar.total)
                bar.refresh()
    return [j[2] for j in ordered], intro_dur


# ------------------------------------------------------------------ audio mix

def mix_audio(cfg: Config, system: System, edl: dict, eff: dict, concat_list: Path,
              total: float, intro_dur: float) -> Path:
    rnd = cfg["render"]
    music = edl["music"]
    xf = float(music["crossfade"])
    tracks = music["tracks"]
    montage_end = intro_dur + edl["duration"]
    out = cfg.work / "render" / "mix.wav"

    cmd = [system.ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat_list)]
    for t in tracks:
        if not Path(t["file"]).exists():
            raise MontageError(f"Music file {t['file']} is gone - run the music stage again.")
        cmd += ["-ss", f"{t['in']:.3f}", "-t", f"{t['out'] - t['in']:.3f}", "-i", t["file"]]

    g = [f"[0:a]volume={float(rnd['game_audio_level']):.3f},asplit=2[gk][gm]"]
    names = []
    for k, t in enumerate(tracks):
        length = t["out"] - t["in"]
        chain = f"[{k + 1}:a]aresample={SR},aformat=sample_fmts=fltp:channel_layouts=stereo"
        if k > 0:
            chain += f",afade=t=in:st=0:d={xf:.3f}"
        if k < len(tracks) - 1:
            chain += f",afade=t=out:st={max(0.0, length - xf):.3f}:d={xf:.3f}"
        delay = int(round((t["at"] + intro_dur) * 1000))
        chain += f",adelay={delay}:all=1[m{k}]"
        g.append(chain)
        names.append(f"[m{k}]")
    mus = "[mus0]"
    g.append("".join(names) + (f"amix=inputs={len(names)}:normalize=0" if len(names) > 1 else "anull") + mus)
    post = []
    for a, b in eff["dropouts"]:
        post.append(f"volume=volume=0.03:enable='between(t,{a + intro_dur:.3f},{b + intro_dur:.3f})'")
    fade = float(music.get("fade_out", 2.0))
    post.append(f"afade=t=out:st={max(0.0, montage_end - fade):.3f}:d={fade:.3f}")
    post.append(f"atrim=0:{montage_end:.3f},apad")
    g.append(f"{mus}{','.join(post)}[mus]")
    ratio = 1 + 6 * float(rnd["game_audio_peaks"])
    g.append(f"[mus][gk]sidechaincompress=threshold=0.03:ratio={ratio:.2f}:attack=5:release=180[duck]")
    g.append(f"[duck][gm]amix=inputs=2:normalize=0,atrim=0:{total:.4f},alimiter=limit=0.95:level=false[out]")
    cmd += ["-filter_complex", ";".join(g), "-map", "[out]", "-ar", str(SR), "-c:a", "pcm_s16le", str(out)]
    res = run(cmd, timeout=1800)
    if res.returncode != 0:
        raise MontageError(f"FFmpeg failed while mixing the audio:\n{res.stderr.strip()[-1200:]}")
    return out


def loudnorm_filter(cfg: Config, system: System, wav: Path) -> str:
    """Two-pass loudness normalisation to render.loudness_lufs (YouTube ~ -14 LUFS)."""
    target = float(cfg["render"]["loudness_lufs"])
    base = f"loudnorm=I={target}:TP=-1.5:LRA=11"
    res = run([system.ffmpeg, "-hide_banner", "-i", str(wav), "-af", base + ":print_format=json",
               "-f", "null", "-"], timeout=1800)
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", res.stderr or "", re.S)
    if res.returncode != 0 or not m:
        log.warning("Loudness measurement failed - using single-pass normalisation.")
        return base
    d = json.loads(m.group(0))
    log.debug("loudness measured: %s", d)
    return (f"{base}:measured_I={d['input_i']}:measured_TP={d['input_tp']}:"
            f"measured_LRA={d['input_lra']}:measured_thresh={d['input_thresh']}:"
            f"offset={d['target_offset']}:linear=true")


# ------------------------------------------------------------------ stage entry

def render(cfg: Config, system: System, edl: dict) -> Path:
    from .effects import plan_effects

    if edl["platform"] != cfg.platform_name:
        raise MontageError(f"The timeline was built for '{edl['platform']}', not '{cfg.platform_name}'. "
                           "Run the music and edl stages again (or the full command).")
    W, H, fps = cfg.output_size()
    started = time.monotonic()
    eff = plan_effects(cfg, edl)
    intro = find_overlay(cfg, "intro", OVERLAY_VIDEO_EXTS)
    outro = find_overlay(cfg, "outro", OVERLAY_VIDEO_EXTS)
    for name, p in (("intro", intro), ("outro", outro)):
        if p:
            log.info("Using overlays/%s as the %s.", p.name, name)

    log.info("Rendering %d clips at %dx%d %dfps with %s%s...", len(edl["clips"]), W, H, fps,
             system.encoder_label, " (preview)" if cfg.preview else "")
    segs, intro_dur = render_segments(cfg, system, edl, eff, W, H, fps, intro, outro)

    concat_list = cfg.work / "render" / "concat.txt"
    concat_list.write_text("ffconcat version 1.0\n" + "\n".join(concat_line(s) for s in segs) + "\n",
                           encoding="utf-8")
    total = 0.0
    for s in segs:
        info = run([system.ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                    str(s)])
        total += float(info.stdout.strip() or 0)

    log.info("Mixing audio (music + game audio, -%s LUFS)...", abs(cfg["render"]["loudness_lufs"]))
    mix = mix_audio(cfg, system, edl, eff, concat_list, total, intro_dur)
    norm = loudnorm_filter(cfg, system, mix)

    out_dir = cfg.path("output")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ("" if cfg.platform_name == "youtube" else f"_{cfg.platform_name}") + ("_preview" if cfg.preview else "")
    stem = f"montage_{datetime.now():%Y%m%d_%H%M}{suffix}"
    out, n = out_dir / f"{stem}.mp4", 2
    while out.exists():  # never overwrite an earlier montage
        out, n = out_dir / f"{stem}_{n}.mp4", n + 1
    tmp = out.with_name(out.stem + ".tmp.mp4")
    base = [system.ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-i", str(mix), "-map", "0:v", "-map", "1:a", "-af", f"{norm},aresample={SR}",
            "-c:a", "aac", "-b:a", str(cfg["render"]["audio_bitrate"]), "-movflags", "+faststart",
            "-t", f"{total:.3f}"]
    res = run(base + ["-c:v", "copy", str(tmp)], timeout=3600)
    if res.returncode != 0:  # very rare: segment streams don't join cleanly - re-encode once
        log.warning("Joining clips without re-encoding failed; re-encoding the video once.")
        log.debug(res.stderr)
        res = run(base + _encode(system, cfg, fps) + [str(tmp)], timeout=7200)
        if res.returncode != 0:
            raise MontageError(f"FFmpeg failed while writing the final video:\n{res.stderr.strip()[-1200:]}")
    os.replace(tmp, out)

    remember_used(cfg, [t["id"] for t in edl["music"]["tracks"]])
    log.info("\nDone: %s", out)
    log.info("  %s, %dx%d %dfps, %s, rendered in %s", fmt_time(total), W, H, fps,
             fmt_size(out.stat().st_size), fmt_time(time.monotonic() - started))
    return out
