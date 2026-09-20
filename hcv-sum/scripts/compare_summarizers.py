"""Compare Stage 2 summarizer / context-mode choices with generic, claim-agnostic metrics.

Usage:  python scripts/compare_summarizers.py

Per configuration, over all sections of all sample documents:
- retention    : share of source sentences entailed (NLI >= 0.5) by their section summary
- faithfulness : share of summary sentences entailed (NLI >= 0.5) by their own section text
- leaks        : summary sentences removed because they came from the retrieved context
- tokens/sec   : mean summary length, and CPU seconds per section
It deliberately does not look for the planted contradictions.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from hcv_sum.cli import _quiet_third_party

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "distilbart / append": ["models.summarizer=sshleifer/distilbart-cnn-12-6", "summarization.context_mode=append"],
    "distilbart / none": ["models.summarizer=sshleifer/distilbart-cnn-12-6", "summarization.context_mode=none"],
    "flan-t5-base / instruct": ["models.summarizer=google/flan-t5-base", "summarization.context_mode=instruct"],
    "flan-t5-large / instruct": ["models.summarizer=google/flan-t5-large", "summarization.context_mode=instruct"],
}


def main() -> int:
    _quiet_third_party()
    sys.stdout.reconfigure(encoding="utf-8")
    from hcv_sum.anchored_summarization import summarize_sections
    from hcv_sum.config import load_config
    from hcv_sum.models import ENTAILMENT, ModelRegistry
    from hcv_sum.scoring import DocumentIndex
    from hcv_sum.segmentation import segment_document

    base = load_config()
    shared = ModelRegistry(base)
    docs = sorted(p for p in (ROOT / "data" / "samples").iterdir() if p.suffix in {".md", ".txt"} and p.stem[0].isdigit())

    for label, overrides in CONFIGS.items():
        cfg = load_config(overrides=overrides)
        reg = ModelRegistry(cfg, embedder=shared.embedder, nli=shared.nli)
        summarizer = reg.summarizer
        retained = faithful = total_src = total_sum = leaks = generated = 0
        tokens, seconds = [], 0.0
        print(f"\n### {label}")
        for doc in docs:
            sections = segment_document(doc.read_text(encoding="utf-8"), reg.embedder, summarizer.count_tokens,
                                        cfg.segmentation)
            index = DocumentIndex.build(sections, reg.embedder)
            t = time.perf_counter()
            summaries = summarize_sections(sections, index, summarizer, reg.embedder, cfg.summarization)
            seconds += time.perf_counter() - t
            for sec, ss in zip(sections, summaries):
                if not ss.generated:
                    continue
                generated += 1
                text = " ".join(ss.sentences)
                tokens.append(summarizer.count_tokens(text))
                leaks += sum("context leak" in why for _, why in ss.dropped)
                ret = reg.nli.predict([(text, s) for s in sec.sentences])[:, ENTAILMENT] >= 0.5
                fai = reg.nli.predict([(sec.text, s) for s in ss.sentences])[:, ENTAILMENT] >= 0.5
                retained += int(ret.sum()); total_src += len(sec.sentences)
                faithful += int(fai.sum()); total_sum += len(ss.sentences)
                print(f"  {doc.stem[:14]} sec{sec.index}: retained {int(ret.sum())}/{len(sec.sentences)} "
                      f"faithful {int(fai.sum())}/{len(ss.sentences)} | {text[:150]}")
        print(f"  => retention {retained / total_src:.2f} | faithfulness {faithful / max(total_sum, 1):.2f} | "
              f"leaks dropped {leaks} | mean summary tokens {np.mean(tokens):.0f} | "
              f"{seconds / max(generated, 1):.1f}s per generated section")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
