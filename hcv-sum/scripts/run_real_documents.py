"""Run the full default pipeline on the four real evaluation documents and count the flags.

Usage:  python scripts/run_real_documents.py [--tag NAME] [--set section.key=value ...]

The pair set (scripts/eval_pairs.py) measures a per-pair false-flag rate. This measures what a user
actually sees: how many contradiction flags the pipeline raises on real, presumably consistent
documents, where Stage 3 compares hundreds or thousands of pairs. Outputs: outputs/real_docs/ (or
outputs/real_docs_<tag>/ with --tag, so configurations can be compared side by side).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["research_refnli.md", "financial_fomc_minutes.md", "contract_imageware.txt", "transcript_fomc_presconf.txt"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="")
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from hcv_sum.cli import _quiet_third_party
    _quiet_third_party()
    sys.stdout.reconfigure(encoding="utf-8")
    from hcv_sum.config import load_config
    from hcv_sum.models import ModelRegistry
    from hcv_sum.pipeline import HCVSumPipeline
    from hcv_sum.report import render_result, save_json

    cfg = load_config(overrides=args.overrides)
    print(f"overrides: {args.overrides or 'none (config/default.yaml)'}")
    pipeline = HCVSumPipeline(cfg, ModelRegistry(cfg))
    out = ROOT / "outputs" / (f"real_docs_{args.tag}" if args.tag else "real_docs")
    out.mkdir(parents=True, exist_ok=True)
    print(f"{'document':<30}{'pages':>6}{'sections':>9}{'claims':>7}{'pairs 3a':>9}{'pairs 3b':>9}"
          f"{'flags 3a':>9}{'flags 3b':>9}{'minutes':>8}", flush=True)
    for name in DOCS:
        text = (ROOT / "data" / "external" / "eval_docs" / name).read_text(encoding="utf-8")
        t = time.perf_counter()
        result = pipeline.run(text, name)
        minutes = (time.perf_counter() - t) / 60
        s = result.stats
        stem = Path(name).stem
        save_json(result, out / f"{stem}.json")
        (out / f"{stem}.txt").write_text(render_result(result), encoding="utf-8")
        print(f"{name:<30}{len(text.split()) / 450:>6.0f}{s['sections']:>9}{s['summary_claims']:>7}"
              f"{s['stage3a_pairs']['compared']:>9}{s['stage3b_pairs']['compared']:>9}"
              f"{len(result.contradictions.contradictions):>9}{len(result.contradictions.one_sided):>9}"
              f"{minutes:>8.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
