# CLAUDE.md — CoD Montage Generator

Read this before changing the code or reviewing clips. It has two parts: the
**montage style guide** (what a good edit looks like; the user edits this over
time) and **project rules** (how the code is built).

---

## Part 1 — Montage style guide

### The goal
A montage should feel *edited by a person who loves the game*: every big moment
hits with the music, nothing drags, and the viewer is hooked in the first
seconds. Better a great 2-minute video than a padded 5-minute one.

### Structure (story arc)
1. **Hook** — the single best moment (biggest multi-kill / cleanest play) lands
   within `hook_seconds` of the platform. On YouTube it lands on the first drop;
   on Shorts/Reels/TikTok it is the very first thing on screen.
2. **Build-up** — setup clips and solid single kills while the music builds.
   Slow-mo during risers, speed snapping back on the drop.
3. **Peaks** — the strongest clips on drops and high-energy sections, separated
   by short breathers in calmer parts of the song.
4. **Finale** — the second-best clip, saved for the last high-energy section.
5. **Outro** — short, clean, ends on a beat (with outro overlay if provided).

### Music sync (most important)
- **Every cut lands on a beat.** Prefer downbeats/bar starts for section changes.
- **The kill moment lands on a beat**, not just the cut. Shift the clip in/out
  points so the kill frame hits a kick or snare.
- Match clip intensity to the music's energy curve: calm music → setup and
  movement, drops → multi-kills.
- **Music dropout** (style option): cut the music for ~0.3–0.6 s just before a
  big kill, then slam back in on the kill.
- Cut rate follows the song: faster in loud sections, longer in calm ones.

### Clip length and pacing
- Clips usually **3–8 s**, never longer than **10 s** (`montage.min_clip/max_clip`).
- Platform `pace` and style `pace` multiply together: >1 = faster cuts.
- Never more than `max_same_source_in_row` consecutive clips from one recording.
- Don't repeat the same effect on consecutive clips.

### What counts as a highlight
Ranked roughly best first:
1. Multi-kills (2+ kills in ~6 s; triple+ is top tier), team wipes, clutches.
2. Long-range snipes, quick-scopes, trick shots, knife/melee kills.
3. Scorestreak payoffs with visible kills, big explosions with kills.
4. Close calls / low-health survivals that end in a kill.
5. Genuinely funny moments (only if clear from the footage).
6. Solid single kills with good aim — filler, used to vary pacing.

### Always exclude
- Menus, lobbies, loadout screens, loading screens, pause screens, scoreboards.
- **The player's own deaths, death cams and killcams** (unless the killcam *is*
  the player's highlight and clearly shows it).
- Spectating, waiting to respawn, parachuting/spawning with no action.
- Long stretches of walking with nothing happening.
- Footage with notifications, recording overlays, or personal info on screen.

### Effects — tasteful, never overdone
| Effect | Rule |
|---|---|
| Slow-mo | Only on the kill moment of top clips (`slowmo_share`), ramp in and out, never on filler. |
| Zoom punch | On big beats only (`zoom_share`), small amount (`zoom_punch`). |
| Shake | Only in `hype` style, on big kicks. |
| Transitions | Flash/whip/blur **only on drops**; plain hard cuts everywhere else. |
| Colour | Light grade (slight contrast + saturation). Never crush blacks or change the game's look. |
| Letterbox | `cinematic` style on landscape platforms only. |
| Text popups | Only `hype` style, only real multi-kills. |
| Audio | Game audio ducked under music; kill/hitmarker/gunshot peaks poke through. Final mix ≈ −14 LUFS. |

### Styles
- **cinematic** (default) — more slow-mo, breathers, letterbox, vignette, music dropouts, slightly slower pacing.
- **hype** — fast cuts, more zooms, shake, whips, text popups.
- **clean** — minimal effects; the gameplay carries it.

### Platforms
- **youtube** (default) — 16:9, 3–5 min, full story arc.
- **shorts / reels / tiktok** — 9:16 centre-crop (CoD's crosshair is centred), 20–60 s, hook within 1.5–2 s, no slow intro.
- **instagram_feed** — 4:5, 30–60 s.
- When cropping vertically, important HUD moments (kill feed, medals) may be shown as a brief inset.

### Reviewing clips (Claude review step)
When asked to review, open the contact sheets in `review/` and, for each clip:
- Is it real action? Any excluded content (menu, death, killcam, spectating)?
- Rate 1–10 for impressiveness; tag it: `multikill`, `snipe`, `clutch`,
  `funny`, `streak`, `filler`, `reject`.
- Write picks to `work/selections.json` (format defined in `src/review.py`).
- Be strict: rejecting a weak clip makes the montage better.

---

## Part 2 — Project rules (for code changes)

- **Never modify or delete anything in `assets/`.** Only read from it.
- Music comes only from `music/`. Never download music or video.
- Heavy video work is done with **FFmpeg** (filter graphs / per-clip jobs), not MoviePy.
- **Low memory** (target PC: i5 6th gen, 8 GB RAM): one recording at a time, analyse
  proxies streamed through FFmpeg pipes, never load a whole video into RAM, render
  per-clip segments then concat.
- Encoders: auto picks nvenc → qsv → amf → x264, each verified by a test encode;
  hardware decode with `-hwaccel auto` and a CPU retry per file.
- Every stage caches its output in `work/` and skips finished work; `--force` redoes it.
  Write cache files atomically (write `.tmp` then rename).
- All tunables live in `config.yaml`. Screen regions are fractions `[x, y, w, h]`.
- User-facing errors raise `MontageError` with a plain-English fix; no tracebacks for
  expected problems. Log everything to `work/run.log`.
- Each stage is its own module in `src/`, runnable with `--stage <name>`.

### Layout
```
src/main.py          CLI and stage runner
src/config.py        config.yaml loading, env/CLI overrides, validation
src/system_check.py  FFmpeg detection, encoder selection, package checks
src/ingest.py        stage 1: probe, proxies, audio
src/audio_score.py   stage 2: loudness + gunfire density per second (chunked WAV reads)
src/visual_score.py  stage 2: per-frame HUD features, kill/hitmarker events, menu/death exclusion
src/candidates.py    stage 2: per-second score, clip building, ranking -> candidates.json
src/regions.py       helper: draws HUD regions on real frames (review/regions/)
src/utils.py         logging, json cache, ffmpeg progress runner
```
Cache layout: `work/videos/<id>/{meta.json, proxy.mp4, audio.wav, audio.npz, visual.npz,
analysis.json, signals.csv}`, `work/ingest.json`, `work/candidates.json`.

Detection design: kills are found without game-specific templates. HUD overlays stay
still while the world moves, so a jump in *stable edges* inside a HUD region (minus any
whole-frame freeze) means a new HUD element appeared; red/gold colour pops add evidence.
Thresholds are robust z-scores per recording. Raw per-frame features are cached and keyed
by the regions, so tuning thresholds never re-reads video.
