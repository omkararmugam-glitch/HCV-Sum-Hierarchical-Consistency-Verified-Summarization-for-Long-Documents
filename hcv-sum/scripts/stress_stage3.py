"""Stress test of Stage 3 pair selection at 500-page scale, with FAKE models (no real text, no NLI model).

Usage:  python scripts/stress_stage3.py [--sentences 25000] [--sections 1000] [--claims-per-section 5]

What it measures: wall-clock and peak memory of Stage 3a (claims vs claims), 3b (claims vs source
sentences) and the source diagnostic (source vs source) at the size a 500-page document would reach,
using random embeddings and a constant NLI stub. This isolates the pipeline's OWN scaling cost --
matrix sizes, candidate lists, sorting -- from model inference cost, which is linear in the number
of pairs actually compared and is capped by max_pairs.

It also runs the OLD dense selection at smaller sizes and reports its memory growth, so the
difference is measured rather than asserted. The dense path is not run at full size: at 25,000 x
25,000 it needs several GB.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class RandomEmbedder:
    """Deterministic pseudo-embeddings: a component shared by the whole document, one per section, and
    per-sentence noise. Cross-section cosine is ~0.35, so most pairs clear the 0.2 similarity floor and
    the top-k and budget limits actually bind, as they do on real prose."""

    def __init__(self, dim: int = 384, seed: int = 0):
        self.dim, self.rng = dim, np.random.default_rng(seed)
        self.cache: dict[str, np.ndarray] = {}

    def encode(self, texts):
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            v = self.cache.get(t)
            if v is None:
                section = int(t.split()[1]) if t.startswith("Unit ") else 0
                shared = np.random.default_rng(10_000_019).standard_normal(self.dim)
                base = np.random.default_rng(section).standard_normal(self.dim)
                noise = np.random.default_rng(abs(hash(t)) % 2**32).standard_normal(self.dim)
                v = (shared + 0.8 * base + 1.0 * noise).astype(np.float32)
                v /= np.linalg.norm(v)
                self.cache[t] = v
            out[i] = v
        return out


class NeutralNLI:
    def __init__(self):
        self.pairs = 0

    def predict(self, pairs):
        self.pairs += len(pairs)
        out = np.zeros((len(pairs), 3), dtype=np.float32)
        out[:, 2] = 1.0
        return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sentences", type=int, default=25_000)
    ap.add_argument("--sections", type=int, default=1_000)
    ap.add_argument("--claims-per-section", type=int, default=5)
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from hcv_sum.config import load_config
    from hcv_sum.contradiction import detect_contradictions, detect_one_sided
    from hcv_sum.resources import StageMonitor
    from hcv_sum.scoring import DocumentIndex
    from hcv_sum.types import Section

    cfg = load_config().contradiction
    embedder, nli = RandomEmbedder(), NeutralNLI()
    per = args.sentences // args.sections
    sections, offset = [], 0
    for s in range(args.sections):
        sents = [f"Unit {s} reported figure {k} changed by {(s * 7 + k) % 97} percent in the period." for k in range(per)]
        sections.append(Section(s, f"S{s}", sents, offset, "heading"))
        offset += per
    claims = [sec.sentences[: args.claims_per_section] for sec in sections]
    with StageMonitor() as m:
        index = DocumentIndex.build(sections, embedder)
    print(f"synthetic: {args.sections} sections, {offset} source sentences, "
          f"{sum(len(c) for c in claims)} summary claims (index built in {m.seconds:.1f}s, {m.peak_mb:.0f} MB)")
    print(f"config: candidate_top_k={cfg.candidate_top_k}, max_pairs={cfg.max_pairs}, "
          f"diagnostic_max_pairs={cfg.diagnostic_max_pairs}, dense_pair_limit={cfg.dense_pair_limit}\n")

    def run(label, fn):
        before = nli.pairs
        with StageMonitor() as mon:
            stats, found = fn()
        print(f"  {label:<44} {mon.seconds:>7.1f} s   peak {mon.peak_mb:>7.0f} MB (+{mon.peak_mb - mon.start_mb:.0f})"
              f"   possible {stats.total:>12,}   compared {stats.checked:>7,}   over budget {stats.skipped_budget:>10,}"
              f"   NLI pairs {nli.pairs - before:,}")

    print("CHUNKED selection (what large documents now use):")
    run("3a  claims x claims", lambda: detect_contradictions(claims, embedder, nli, cfg))
    run("3b  claims x sibling source sentences", lambda: detect_one_sided(claims, set(), index, embedder, nli, cfg))
    diag = [sec.sentences for sec in sections]
    import dataclasses
    run("diagnostic  source x source", lambda: detect_contradictions(
        diag, embedder, nli, dataclasses.replace(cfg, max_pairs=cfg.diagnostic_max_pairs)))

    print("\nOLD dense selection, diagnostic (source x source), at increasing sizes:")
    dense_cfg = dataclasses.replace(cfg, dense_pair_limit=0, max_pairs=cfg.diagnostic_max_pairs)
    for n_sec in sorted({max(1, args.sections // 20), max(1, args.sections // 10), max(1, args.sections // 5)}):
        sub = [sec.sentences for sec in sections[:n_sec]]
        run(f"dense diagnostic, {n_sec * per:,} sentences", lambda: detect_contradictions(sub, embedder, nli, dense_cfg))
    return 0


if __name__ == "__main__":
    t = time.perf_counter()
    code = main()
    print(f"\ntotal {time.perf_counter() - t:.0f} s")
    raise SystemExit(code)
