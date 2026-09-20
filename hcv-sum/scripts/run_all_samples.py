"""Run the full pipeline on every sample document, printing and saving all stage outputs.

Usage:  python scripts/run_all_samples.py [--config path] [--set section.key=value ...] [--tag name]
Writes outputs/<tag>/<document>.txt (rendered report) and .json (full run).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from hcv_sum.cli import _quiet_third_party

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    parser.add_argument("--tag", default="default", help="subfolder of outputs/ for this run")
    parser.add_argument("--only", help="substring filter on sample file names")
    parser.add_argument("--no-diagnostic", action="store_true", help="skip the source-level Stage 3 diagnostic")
    args = parser.parse_args()

    _quiet_third_party()
    sys.stdout.reconfigure(encoding="utf-8")
    from hcv_sum.config import load_config
    from hcv_sum.models import ModelRegistry
    from hcv_sum.pipeline import HCVSumPipeline
    from hcv_sum.report import render_result, save_json

    cfg = load_config(args.config, args.overrides)
    pipeline = HCVSumPipeline(cfg, ModelRegistry(cfg))   # models loaded once, shared across documents
    out_dir = ROOT / "outputs" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.overrides:
        print(f"config overrides: {args.overrides}")

    samples = sorted(p for p in (ROOT / "data" / "samples").iterdir()
                     if p.suffix in {".txt", ".md"} and p.name != "GROUND_TRUTH.md")
    for path in samples:
        if args.only and args.only not in path.name:
            continue
        t = time.perf_counter()
        result = pipeline.run(path.read_text(encoding="utf-8-sig"), path.name,
                              diagnose_sources=not args.no_diagnostic,
                              progress=lambda msg: print(msg, file=sys.stderr, flush=True))
        rendered = render_result(result)
        print(rendered)
        print(f"\n[{path.name}] total {time.perf_counter() - t:.1f}s")
        (out_dir / f"{path.stem}.txt").write_text(rendered, encoding="utf-8")
        save_json(result, out_dir / f"{path.stem}.json")
    print(f"\nreports written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
