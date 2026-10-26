# CoD Montage Generator

Drop your Call of Duty recordings into a folder, add some royalty‑free music, run one
command and get a finished montage synced to the beat. Works with **CoD Mobile**,
Modern Warfare, Black Ops and Warzone recordings, and it all runs on your own PC.

> **Status:** all 5 stages are built. Kill detection uses starting-guess screen regions;
> check them once on your own footage (see *Tuning kill detection* below).

| Stage | What it does | Status |
|---|---|---|
| 1. Ingest | Reads your recordings, makes small analysis copies | ✅ done |
| 2. Detect | Finds kills, multi-kills and intense fights | ✅ done |
| 3. Review | Contact sheets of the best clips (optional Claude review) | ✅ done |
| 4. Music + edit | Beat detection, picks the song, builds the timeline | ✅ done |
| 5. Effects + render | Slow‑mo, zooms, transitions, colour, final video | ✅ done |

---

## 1. One-time setup (Windows)

### a) Install Python
1. Download Python **3.11 or newer** from <https://www.python.org/downloads/>.
2. Run the installer and **tick "Add python.exe to PATH"** at the bottom of the first screen.

### b) Install FFmpeg (does the video work)
Open **PowerShell** and run:
```powershell
winget install --id Gyan.FFmpeg -e
```
Close PowerShell and open it again. Check it works:
```powershell
ffmpeg -version
```
<details><summary>No winget? Manual install</summary>

1. Download `ffmpeg-release-essentials.zip` from <https://www.gyan.dev/ffmpeg/builds/>.
2. Extract it to `C:\ffmpeg` (so `C:\ffmpeg\bin\ffmpeg.exe` exists).
3. Start menu → *Edit the system environment variables* → *Environment Variables* →
   under *User variables* pick **Path** → *Edit* → *New* → `C:\ffmpeg\bin` → OK.
4. Reopen PowerShell.
</details>

### c) Get the tool and install its packages
Download this project (green **Code** button → *Download ZIP*) and extract it, then in
PowerShell:
```powershell
cd path\to\cod-montage-generator
pip install -r requirements.txt
```

---

## 2. Making a montage

1. **Copy your recordings** into `assets\` (MP4, MKV, MOV, …). Long recordings are fine.
   Your originals are **never changed or deleted**.
2. **Copy some music** into `music\` (MP3 or WAV). Only use music you're allowed to use;
   see the free sources below.
3. Run:
   ```powershell
   python -m src.main
   ```
4. Your video appears in `output\`.

### Options
```powershell
python -m src.main --platform shorts       # vertical video for YouTube Shorts
python -m src.main --platform reels        # Instagram Reels
python -m src.main --style hype            # fast cuts, more effects
python -m src.main --length 240            # exactly 4 minutes (default: auto)
python -m src.main --music "mytrack.mp3"   # pick the song yourself
python -m src.main --preview               # quick low-quality test render
python -m src.main --auto                  # skip the Claude review step
python -m src.main --game mw               # recordings from Modern Warfare
```

| Option | Choices | Default |
|---|---|---|
| `--platform` | `youtube` (16:9, 3–5 min), `shorts`, `reels`, `tiktok` (9:16, under a minute), `instagram_feed` (4:5) | `youtube` |
| `--style` | `cinematic` (slow‑mo, letterbox), `hype` (fast, flashy), `clean` (minimal) | `cinematic` |
| `--game` | `codm`, `mw`, `bo`, `warzone`, `generic` | `codm` |
| `--length` | seconds, or `auto` (decides from how many good clips you have) | `auto` |

You can also set these permanently in `config.yaml`, or with environment variables:
```powershell
$env:MONTAGE_PLATFORM = "reels"
$env:MONTAGE_STYLE = "hype"
```
Priority: command line → environment variable → `config.yaml`.

### The review step
After finding highlights, the tool makes a **contact sheet** (8 frames in a grid) for each
of the top 40 clips in `review\`, and pauses. You then choose one of these:

- **Let Claude review them:** open Claude Code in the project folder and say
  *"review the montage clips"*. Claude looks at every sheet, rejects menus, deaths and
  boring bits, rates the rest, and saves its picks. Then run your command again and it
  carries on from there.
- **Skip it:** add `--auto` and the computed scores decide.
- **Do it yourself:** edit `work\selections.json` (the format is at the top of `src\review.py`).

On the sheets, a red border means a kill was detected near that frame, and yellow marks
the key moment.

### Music choice
With no `--music`, the tool listens to every track in `music\` and picks the one that
fits best (energy, drops, tempo, length; `cinematic` prefers tracks with big build-ups
and drops, `hype` prefers loud fast ones), rotating so you don't get the same song every
time. It never downloads music. `--music` also accepts part of a name: `--music chill`.

How the edit follows the music:
- The song starts a few bars before its first big **drop**, and your **best clip** lands on
  that drop (within 8 s on YouTube, about 2 s on Shorts/Reels/TikTok).
- **Every cut is on a beat**, and each clip is shifted so a **kill hits a beat**.
- Loud parts of the song get quicker cuts, calm parts longer clips. Later drops get the
  best clips that are left, and your **second-best clip closes** the video.
- Multi-kill clips are kept long enough to show the whole streak.
- If one song is too short, a second one is mixed in with a crossfade on a downbeat.
- With `--length auto`, the length depends on how much good footage you have. If there isn't
  enough for the platform's usual length, the video comes out shorter rather than padded.

### Running one stage at a time (for testing)
```powershell
python -m src.main --stage check    # just check FFmpeg / GPU
python -m src.main --stage ingest   # just read recordings and make proxies
python -m src.main --stage detect   # just find highlight candidates
python -m src.main --stage review   # just make the contact sheets
python -m src.main --stage music    # just pick the song and find beats/drops
python -m src.main --stage edl      # just build the timeline (work\edl.json)
python -m src.main --stage render   # just render the timeline to output\
python -m src.main --stage regions  # draw the kill-detection boxes on your footage
python -m src.main --force          # redo everything, ignoring the cache
python -m src.main -v               # show extra detail
```

### Tuning kill detection for your game
The tool spots kills by watching a few areas of the screen (kill feed, kill popup,
medals, hitmarker). Each game's HUD is different, so check the boxes once:

1. Run `python -m src.main --stage regions`.
2. Open the images in `review\regions\`. Each one shows a 0.1 grid and the coloured boxes.
   You can also drop your own screenshots into a `samples\` folder and they get boxes too.
3. If a box misses its HUD element, read the new position off the grid and edit
   `regions:` → your game in `config.yaml` as `[x, y, width, height]`
   (fractions of the screen, from the top-left corner).
4. Run `python -m src.main --stage detect` again. Only the changed part is redone.

Finding too many or too few kills? Change `kill_z` in `config.yaml` (higher = stricter).
Every recording gets a `work\videos\<name>\signals.csv` you can open in Excel to see
the score for every second.

---

## 3. Free, royalty-free music

- **YouTube Audio Library**: <https://studio.youtube.com> → *Audio library*
- **Pixabay Music**: <https://pixabay.com/music/>
- **NoCopyrightSounds (NCS)**: <https://ncs.io> (follow their credit rules in your description)

Don't use songs you don't have rights to; YouTube and Instagram will mute or block them.

---

### Effects and the final video
The finished video is saved as `output\montage_YYYYMMDD_HHMM.mp4` (Shorts/Reels get the
platform in the name, previews end in `_preview`). Earlier videos are never overwritten.

What the styles do:

| | cinematic (default) | hype | clean |
|---|---|---|---|
| Slow motion on top kills (kill stays on the beat) | most | some | rare |
| Slow-mo into the first drop | ✓ | ✓ | ✓ |
| Zoom punch on a big beat | a few | often | – |
| Camera shake on big kicks | – | ✓ | – |
| Transitions on drops (flash / whip / blur) | flash, blur | all three | flash |
| Letterbox bars and vignette | ✓ | – | – |
| "DOUBLE / TRIPLE KILL" text | – | ✓ | – |
| Music cuts out right before a big kill | ✓ | ✓ | – |

Always: a light colour grade, game audio under the music with gunshots poking through,
fade in and out, and loudness evened out to about −14 LUFS (what YouTube expects).
Every amount is in `config.yaml` under `styles:` and `render:`.

**Optional extras in `overlays\`:**
- `intro.mp4` / `outro.mp4` (or .mov/.mkv/.webm): played before/after the montage.
- `logo.png`: shown small in the top-right corner (size and opacity in `config.yaml`).
- Set `gamer_name` in `config.yaml` to show your name as a title over the first clip
  (skipped when you have an intro video).

**Quick test first:** `python -m src.main --preview` renders a 720p version fast. Clips are
cached, so after a preview or a small change only the changed clips are re-rendered.

## 4. Good to know

- **Speed:** the tool uses your graphics hardware when it can (Intel Quick Sync, NVIDIA
  or AMD), otherwise the CPU. `--stage check` shows which one it picked.
- **Memory:** recordings are processed one at a time from small 480p copies, and the final
  video is rendered one clip at a time, so 8 GB RAM is enough even for hour‑long files.
- **Vertical videos** (Shorts/Reels/TikTok) are a centre crop around the crosshair, so the
  kill feed at the screen edge is cut off. (Showing it as an inset isn't built yet.)
- **Disk:** the `work\` folder holds about 400–500 MB of cache per hour of footage. It's
  safe to delete any time; it just makes the next run slower.
- **Phone recordings** (CoD Mobile) with variable frame rate or rotation are handled
  automatically.
- **Log file:** every run is logged to `work\run.log`. Attach it if you report a problem.

## 5. Troubleshooting

| Message | Fix |
|---|---|
| `FFmpeg was not found` | Do step 1b, then **reopen** PowerShell. |
| `Missing Python packages` | `pip install -r requirements.txt` |
| `No recordings found` | Put videos directly in the `assets\` folder. |
| `can't read this file` | The file is corrupt or still being recorded; re-export it. |
| `config.yaml has a formatting mistake` | Use spaces, not tabs, and keep the indentation. |
| `Not enough disk space` | Free some space or delete the `work\` folder. |

## Project layout
```
assets/    your recordings (never modified)
music/     your royalty-free tracks
overlays/  optional intro / outro / logo
work/      cache (safe to delete)
review/    clip contact sheets + region check images
samples/   optional screenshots for checking the regions
output/    finished montages
src/       the code
config.yaml  all settings
CLAUDE.md    montage style guide
```
