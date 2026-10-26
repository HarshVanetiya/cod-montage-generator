"""CoD Montage Generator - command line entry point.

    python -m src.main                      # full run
    python -m src.main --auto               # skip the Claude review step
    python -m src.main --preview            # fast low-quality test render
    python -m src.main --platform shorts --style hype
    python -m src.main --music "track.mp3" --length 240
    python -m src.main --stage ingest       # run one stage (for debugging)
"""

from __future__ import annotations

import argparse
import sys

from .config import load_config
from .utils import MontageError, StageTimer, log, setup_logging

STAGES = ["check", "ingest", "detect", "review", "music", "edl", "render"]
# Helpers that can be run with --stage but aren't part of a full run.
HELPERS = ["regions"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m src.main",
        description="Turn Call of Duty recordings in assets/ into a music-synced montage.",
    )
    p.add_argument("--stage", choices=STAGES + HELPERS + ["all"], default="all",
                   help="run only this stage (default: all); 'regions' draws the HUD "
                        "boxes on your footage for tuning")
    p.add_argument("--auto", action="store_true",
                   help="skip the Claude review step and use computed scores only")
    p.add_argument("--preview", action="store_true", help="fast, lower-resolution test render")
    p.add_argument("--music", metavar="FILE", help="use this track from music/ instead of auto-pick")
    p.add_argument("--length", metavar="SECONDS",
                   help="montage length in seconds, or 'auto' (default: from config)")
    p.add_argument("--platform", help="youtube, shorts, reels, tiktok, instagram_feed")
    p.add_argument("--style", help="cinematic, hype, clean")
    p.add_argument("--game", help="codm, mw, bo, warzone, generic")
    p.add_argument("--force", action="store_true", help="redo stages even if cached results exist")
    p.add_argument("--config", metavar="PATH", help="use a different config.yaml")
    p.add_argument("-v", "--verbose", action="store_true", help="show debug messages")
    return p.parse_args(argv)


def run_pipeline(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, vars(args))
    setup_logging(cfg.work, args.verbose)
    log.debug("Args: %s", vars(args))

    w, h, fps = cfg.output_size()
    length = cfg.length if cfg.length == "auto" else f"{cfg.length:g}s"
    log.info("Game: %s | Platform: %s (%dx%d @ %dfps) | Style: %s | Length: %s%s",
             cfg.game, cfg.platform_name, w, h, fps, cfg.style_name, length,
             " | PREVIEW" if cfg.preview else "")

    from .system_check import check_packages, check_system

    stages = STAGES[1:] if args.stage == "all" else [args.stage]
    check_packages(["ingest"] + stages)
    videos = None

    with StageTimer("System check"):
        system = check_system(cfg)
    if args.stage == "check":
        return 0

    for stage in stages:
        if stage == "ingest":
            from .ingest import check_music_folder, ingest
            with StageTimer("Stage 1: Ingest"):
                videos = ingest(cfg, system)
                check_music_folder(cfg)
        elif stage == "detect":
            from .candidates import detect
            from .ingest import load_ingested
            with StageTimer("Stage 2: Candidate detection"):
                detect(cfg, videos or load_ingested(cfg))
        elif stage == "review":
            from .review import review
            with StageTimer("Stage 3: Review"):
                ready = review(cfg, system)
            if not ready:
                break
        elif stage == "music":
            from .edl import decide_length
            from .music import build_beatmap
            from .review import load_selections
            with StageTimer("Stage 4a: Music and beat map"):
                build_beatmap(cfg, system, decide_length(cfg, load_selections(cfg)))
        elif stage == "edl":
            from .edl import run_edl
            from .review import load_selections
            from .utils import read_json
            bm = read_json(cfg.work / "beatmap.json")
            if not bm:
                raise MontageError("No beat map yet - run  python -m src.main --stage music  first.")
            with StageTimer("Stage 4b: Edit decision list"):
                run_edl(cfg, load_selections(cfg), bm)
        elif stage == "render":
            from .render import render
            from .utils import read_json
            edl = read_json(cfg.work / "edl.json")
            if not edl:
                raise MontageError("No timeline yet - run  python -m src.main --stage edl  first.")
            with StageTimer("Stage 5: Effects and render"):
                render(cfg, system, edl)
        elif stage == "regions":
            from .ingest import load_ingested
            from .regions import regions_helper
            with StageTimer("Region helper"):
                regions_helper(cfg, system, load_ingested(cfg))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run_pipeline(args)
    except MontageError as e:
        log.error("\nERROR: %s", e) if log.handlers else print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopped. Finished work is cached - run the same command to continue.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
