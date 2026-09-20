"""Measure Stage 2 batching on a real run: speedup per batch size, and how full the batches really are.

Usage:  python scripts/bench_stage2.py data/bench/medium_50sections.md [--limit 24]

Stage 2 only batches sections that share the SAME (min_new_tokens, max_new_tokens) budget, because one
generate() call applies one length limit to the whole batch (anchored_summarization.py). Budgets are
derived from each section's length, so on real documents many buckets hold one section and batching
cannot help them. This reports the bucket sizes alongside the timings, and checks that every batch
size produced byte-identical summaries.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("document", type=Path)
    ap.add_argument("--limit", type=int, default=0, help="only the first N sections (0 = all)")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from hcv_sum.cli import _quiet_third_party
    _quiet_third_party()
    sys.stdout.reconfigure(encoding="utf-8")
    from hcv_sum.anchored_summarization import _prepare, summarize_sections
    from hcv_sum.config import load_config
    from hcv_sum.models import ModelRegistry
    from hcv_sum.scoring import DocumentIndex
    from hcv_sum.segmentation import segment_document

    cfg = load_config()
    reg = ModelRegistry(cfg)
    text = args.document.read_text(encoding="utf-8")
    sections = segment_document(text, reg.embedder, reg.summarizer.count_tokens, cfg.segmentation)
    if args.limit:
        sections = sections[: args.limit]
    index = DocumentIndex.build(sections, reg.embedder)
    prepared = [_prepare(s, index, reg.summarizer, cfg.summarization) for s in sections]
    budgets = Counter((p.min_new, p.max_new) for p in prepared if p.needs_generation)
    n_gen = sum(budgets.values())
    print(f"{args.document.name}: {len(sections)} sections, {n_gen} need generation, {len(budgets)} distinct "
          f"length budgets; bucket sizes: {sorted(budgets.values(), reverse=True)}")
    reg.summarizer.summarize_batch([prepared[0].model_input], min_new_tokens=5, max_new_tokens=10)   # warm-up

    reference, base_time = None, None
    for batch in (1, 4, 8):
        scfg = dataclasses.replace(cfg.summarization, batch_size=batch)
        t = time.perf_counter()
        out = summarize_sections(sections, index, reg.summarizer, reg.embedder, scfg)
        seconds = time.perf_counter() - t
        texts = [s.sentences for s in out]
        same = "reference" if reference is None else ("IDENTICAL" if texts == reference else "DIFFERENT")
        reference = reference or texts
        base_time = base_time or seconds
        calls = sum(-(-size // batch) for size in budgets.values())   # generate() calls at this batch size
        print(f"  batch_size {batch}: {seconds:6.1f} s  ({seconds / max(n_gen, 1):.2f} s/section, "
              f"speedup {base_time / seconds:.2f}x, {calls} generate() calls for {n_gen} sections)  output {same}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
