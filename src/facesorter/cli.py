from __future__ import annotations
import argparse, logging
from pathlib import Path
from .config import SorterConfig
from .logging_conf import setup_logging
from .core.sorter import FaceSorterCore
from .io.images import list_images

def main():
    setup_logging()
    p = argparse.ArgumentParser(prog="facesorter", description="Sort photos by person using InsightFace")
    p.add_argument("--people", type=Path, default=Path.cwd()/"people")
    p.add_argument("--input", type=Path, default=Path.cwd()/"input_photos")
    p.add_argument("--output", type=Path, default=Path.cwd()/"output_photos")
    p.add_argument("--move", action="store_true", help="Move instead of copy")
    p.add_argument("--ctx", type=int, default=-1, help="-1 CPU, 0 GPU#0")
    p.add_argument("--threshold", type=float, default=0.45)
    p.add_argument("--ambig", type=float, default=0.05)
    p.add_argument("--min-face", type=int, default=24)
    args = p.parse_args()

    cfg = SorterConfig(
        people_dir=args.people, input_dir=args.input, output_dir=args.output,
        move_instead_copy=args.move, ctx_id=args.ctx,
        match_threshold=args.threshold, ambiguous_margin=args.ambig, min_face_size=args.min_face,
    )

    core = FaceSorterCore(cfg)
    core.load_service()
    n = core.rebuild_gallery()
    logging.info("Gallery ready: %d person(s)", n)

    files = list_images(cfg.input_dir)
    def log_cb(s): logging.info(s)
    def prog_cb(i, n): pass
    moved = core.process_files(files, log_cb, prog_cb)
    logging.info("Done. Moved/Copied: %d", moved)

if __name__ == "__main__":
    main()
