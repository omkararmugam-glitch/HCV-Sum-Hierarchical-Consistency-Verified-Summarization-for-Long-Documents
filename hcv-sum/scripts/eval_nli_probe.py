"""Compare NLI cross-encoders on the contradiction probe set (independent of the sample documents).

Usage:  python scripts/eval_nli_probe.py [--models m1 m2 ...]

For every model it reports, for both direction aggregations (max, min) and a few
thresholds: precision / recall / F1 of "contradiction" and false positives per
category, plus the CPU time per pair. Use it to choose models.nli,
contradiction.direction_aggregation and contradiction.threshold without looking
at the evaluation documents.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from hcv_sum.cli import _quiet_third_party

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = [
    "cross-encoder/nli-deberta-v3-small",
    "cross-encoder/nli-MiniLM2-L6-H768",
    "cross-encoder/nli-distilroberta-base",
    "cross-encoder/nli-deberta-v3-base",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--probe", default=str(ROOT / "data" / "probes" / "contradiction_pairs.jsonl"))
    parser.add_argument("--show-pairs", action="store_true")
    args = parser.parse_args()

    _quiet_third_party()
    sys.stdout.reconfigure(encoding="utf-8")
    from hcv_sum.models import CONTRADICTION, CrossEncoderNLI

    rows = [json.loads(line) for line in open(args.probe, encoding="utf-8") if line.strip()]
    labels = np.array([r["label"] for r in rows])
    print(f"probe pairs: {len(rows)} ({labels.sum()} contradictions, {len(rows) - labels.sum()} non-contradictions)")

    for name in args.models:
        try:
            nli = CrossEncoderNLI(name, "cpu", 16)
        except Exception as exc:  # report and continue with the next model
            print(f"\n### {name}: FAILED TO LOAD: {exc}")
            continue
        t = time.perf_counter()
        ab = nli.predict([(r["a"], r["b"]) for r in rows])[:, CONTRADICTION]
        ba = nli.predict([(r["b"], r["a"]) for r in rows])[:, CONTRADICTION]
        ms_per_pair = 1000 * (time.perf_counter() - t) / (2 * len(rows))
        print(f"\n### {name}   ({ms_per_pair:.0f} ms per NLI call on CPU)")
        for agg_name, score in (("max", np.maximum(ab, ba)), ("min", np.minimum(ab, ba)), ("mean", (ab + ba) / 2)):
            for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
                pred = score >= thr
                tp = int((pred & (labels == 1)).sum())
                fp = int((pred & (labels == 0)).sum())
                fn = int((~pred & (labels == 1)).sum())
                p = tp / (tp + fp) if tp + fp else 0.0
                rc = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
                fp_cats = defaultdict(int)
                fn_cats = defaultdict(int)
                for r, pr in zip(rows, pred):
                    if pr and r["label"] == 0:
                        fp_cats[r["category"]] += 1
                    if not pr and r["label"] == 1:
                        fn_cats[r["category"]] += 1
                print(f"  agg={agg_name} thr={thr:.1f}  P={p:.2f} R={rc:.2f} F1={f1:.2f}  TP={tp} FP={fp} FN={fn}"
                      f"  FP by cat={dict(fp_cats)}  FN by cat={dict(fn_cats)}")
        if args.show_pairs:
            for r, x, y in zip(rows, ab, ba):
                print(f"    [{r['label']}] {x:.2f}/{y:.2f} {r['category']:<24} {r['a'][:45]} || {r['b'][:45]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
