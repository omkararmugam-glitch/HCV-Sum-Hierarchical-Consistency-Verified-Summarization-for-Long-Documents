"""Download every configured model, load it on CPU, and run a tiny sanity check.

Usage:  python scripts/download_models.py [--config path] [--set section.key=value ...]
"""

from __future__ import annotations

import argparse
import platform
import sys
import time

from hcv_sum.config import load_config
from hcv_sum.models import ModelRegistry
from hcv_sum.text_utils import ensure_nltk, split_sentences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--set", action="append", default=[], dest="overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)

    import nltk
    import sentence_transformers
    import torch
    import transformers

    print(f"python {platform.python_version()} ({sys.executable})")
    print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()} | threads: {torch.get_num_threads()}")
    print(f"transformers {transformers.__version__} | sentence-transformers {sentence_transformers.__version__} "
          f"| nltk {nltk.__version__}")

    ensure_nltk()
    print("nltk punkt_tab OK:", split_sentences("Dr. Smith arrived at 3 p.m. on Monday. He left early."))

    registry = ModelRegistry(cfg)

    t = time.perf_counter()
    emb = registry.embedder.encode(["The plant was shut down.", "Production stopped at the factory.", "I like tea."])
    sims = emb @ emb.T
    print(f"\n[embedder] {cfg.models.embedder} loaded in {time.perf_counter() - t:.1f}s, dim={emb.shape[1]}")
    print(f"  sim(shutdown, production stopped)={sims[0, 1]:.3f}  sim(shutdown, tea)={sims[0, 2]:.3f}")

    t = time.perf_counter()
    nli = registry.nli
    print(f"\n[nli] {cfg.models.nli} loaded in {time.perf_counter() - t:.1f}s")
    print(f"  checkpoint id2label: {getattr(nli, 'id2label', '?')}")
    probe = [
        ("Revenue fell because of the shutdown.", "The revenue decline was unrelated to the shutdown."),
        ("A man is playing a guitar on stage.", "A man is performing music."),
        ("A man is playing a guitar on stage.", "The weather is sunny in Paris."),
    ]
    for (p, h), row in zip(probe, nli.predict(probe)):
        print(f"  P={p!r}\n  H={h!r}\n    contradiction={row[0]:.3f} entailment={row[1]:.3f} neutral={row[2]:.3f}")

    t = time.perf_counter()
    summarizer = registry.summarizer
    print(f"\n[summarizer] {cfg.models.summarizer} loaded in {time.perf_counter() - t:.1f}s")
    text = ("The city council voted on Tuesday to approve a new budget that increases spending on public "
            "transport by 12 percent. The plan adds three new bus routes and extends service hours on weekends. "
            "Critics said the increase would require higher parking fees, which the council will debate next month.")
    t = time.perf_counter()
    out = summarizer.summarize(text, min_new_tokens=10, max_new_tokens=60)
    print(f"  input tokens={summarizer.count_tokens(text)}  generation took {time.perf_counter() - t:.1f}s")
    print(f"  summary: {out}")
    print("\nAll models loaded OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
