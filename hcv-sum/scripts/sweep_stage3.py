"""Time Stage 3 on ONE saved run under several Stage 3 settings, reusing the saved Stages 1-2.

Usage:
  python scripts/sweep_stage3.py --run outputs/gao_tune/baseline.json            # all configurations below
  python scripts/sweep_stage3.py --run ... --only default,tight_a --repeat 2
  python scripts/sweep_stage3.py --run ... --only diagnostic                     # the opt-in source diagnostic

Stages 1-2 do not depend on Stage 3 settings, so their saved output is reused and only
check_section_summaries() is re-run and timed. Model loading and the DocumentIndex rebuild happen once,
before any timing, so the reported seconds are Stage 3 alone (3a + 3b + resolution) and comparable
across configurations.

This measures SPEED and COVERAGE (pairs compared, flags found) only. It says nothing about precision or
recall: a flag count is not a correctness check, and this script has no ground truth. Use it to see what
tightening the pre-filters costs in comparisons, not to decide whether the flags are right.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Progressively tighter pre-filtering. Each is the current default set with the named keys changed.
CONFIGS: dict[str, list[str]] = {
    "default": [],
    "tight_a": ["contradiction.pair_min_similarity=0.35", "contradiction.candidate_top_k=15"],
    "tight_b": ["contradiction.pair_min_similarity=0.45", "contradiction.candidate_top_k=8",
                "contradiction.max_pairs=8000"],
    "tight_c": ["contradiction.pair_min_similarity=0.55", "contradiction.candidate_top_k=4",
                "contradiction.max_pairs=4000"],
    "no_3b": ["contradiction.check_sibling_sources=false"],
    "fast": ["contradiction.pair_min_similarity=0.45", "contradiction.candidate_top_k=8",
             "contradiction.max_pairs=8000", "contradiction.check_sibling_sources=false"],
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="a saved --json run (Stages 1-2 are reused)")
    ap.add_argument("--only", default="", help="comma-separated subset of: " + ", ".join(CONFIGS) + ", diagnostic")
    ap.add_argument("--repeat", type=int, default=1, help="repeat each configuration (reports the fastest)")
    ap.add_argument("--out", type=Path, default=None, help="write the rows as JSON here")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "scripts"))
    from rerun_stage3 import load_run

    from hcv_sum.cli import _quiet_third_party
    _quiet_third_party()
    sys.stdout.reconfigure(encoding="utf-8")
    from hcv_sum.config import load_config
    from hcv_sum.contradiction import check_section_summaries, diagnose_source_sections
    from hcv_sum.models import ModelRegistry
    from hcv_sum.scoring import DocumentIndex

    _d, sections, summaries = load_run(args.run)
    reg = ModelRegistry(load_config())
    emb, nli = reg.embedder, reg.nli
    index = DocumentIndex.build(sections, emb)      # outside the timed region
    print(f"{args.run.name}: {len(sections)} sections, {len(index.sentences)} source sentences, "
          f"{sum(len(s.sentences) for s in summaries)} summary sentences")
    names = [n.strip() for n in args.only.split(",") if n.strip()] or list(CONFIGS)
    print(f"{'config':<12}{'stage3 s':>10}{'3a pairs':>10}{'3b pairs':>10}{'flags':>7}{'3a':>5}{'3b':>5}"
          f"{'unscored':>10}   settings")
    rows = []
    for name in names:
        diagnostic = name == "diagnostic"
        overrides = [] if diagnostic else CONFIGS[name]
        cfg = load_config(overrides=overrides).contradiction
        best = None
        for _ in range(max(1, args.repeat)):
            t = time.perf_counter()
            if diagnostic:
                # The opt-in --diagnose-sources check: raw source sentence vs raw source sentence.
                rep = diagnose_source_sections(index, emb, nli, cfg)
                a, b = rep.pair_stats, None
            else:
                rep = check_section_summaries(summaries, index, emb, nli, cfg)
                a, b = rep.pair_stats, rep.one_sided_stats
            seconds = time.perf_counter() - t
            n3a = len(rep.contradictions)
            n3b = len(rep.one_sided)
            row = {"config": name, "overrides": overrides, "seconds": round(seconds, 1),
                   "pairs_3a": a.checked, "pairs_3b": (b.checked if b else 0),
                   "eligible_3a": a.total, "eligible_3b": (b.total if b else 0),
                   "flags": n3a + n3b, "flags_3a": n3a, "flags_3b": n3b, "unscored": rep.unscored}
            best = row if best is None or row["seconds"] < best["seconds"] else best
        rows.append(best)
        print(f"{name:<12}{best['seconds']:>10.1f}{best['pairs_3a']:>10,}{best['pairs_3b']:>10,}"
              f"{best['flags']:>7}{best['flags_3a']:>5}{best['flags_3b']:>5}{best['unscored']:>10}   "
              f"{' '.join(o.split('contradiction.')[-1] for o in best['overrides']) or '(current defaults)'}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"\nrows written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
