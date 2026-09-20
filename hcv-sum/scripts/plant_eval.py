"""Document-level test: plant known contradictions inside the real documents and see if Stage 3 finds them.

Usage:  python scripts/plant_eval.py [--tag NAME] [--set section.key=value ...]

The pair set (eval_pairs.py) scores isolated sentence pairs; run_real_documents.py counts flags on
documents with no known contradiction. Neither shows whether a real contradiction survives the whole
pipeline -- segmentation, compression, the pre-filters, the pair budget -- or where it would rank
among hundreds of false flags. Here, for each document:

1. Pick 3 of its 9 constructed contradictions from data/eval/constructed_pairs.jsonl at random
   (seed 7, fixed; chosen WITHOUT looking at whether the pair was caught in eval_pairs.py).
2. Sentence A is already in the document. Insert the counter-claim B at the end of the paragraph
   half the document away from A, so the two land in different sections.
3. Run the pipeline and look for a flag whose two sides match A and B (cosine >= MATCH).

Reported per plant: caught or not, its rank among ALL flags ordered by score, and whether A and B
survived into the section summaries (3a needs both; 3b needs one side in a summary).

Limitation: B was hand-written against A and reuses its wording, which makes the pair easier to
retrieve (higher cosine) than a contradiction occurring naturally between two authors' sections.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOCS = {"research": "research_refnli.md", "financial": "financial_fomc_minutes.md",
        "contract": "contract_imageware.txt", "transcript": "transcript_fomc_presconf.txt"}
PER_DOC = 3
SEED = 7
MATCH = 0.70


def plant(text: str, pairs: list[dict]) -> tuple[str, list[dict]]:
    paragraphs = text.split("\n\n")
    placed = []
    for p in pairs:
        key = p["a"][:60]
        home = next((i for i, para in enumerate(paragraphs) if key in " ".join(para.split())), None)
        if home is None:
            placed.append({**p, "planted": False, "why": "sentence A not found verbatim in the document"})
            continue
        n = len(paragraphs)
        target = (home + n // 2) % n
        # Skip headings and very short paragraphs so B lands inside running text.
        while paragraphs[target].lstrip().startswith("#") or len(paragraphs[target].split()) < 15:
            target = (target + 1) % n
        paragraphs[target] = paragraphs[target].rstrip() + " " + p["b"]
        placed.append({**p, "planted": True, "home_paragraph": home, "target_paragraph": target})
    return "\n\n".join(paragraphs), placed


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
    from hcv_sum.report import save_json

    constructed = [json.loads(line) for line in
                   (ROOT / "data" / "eval" / "constructed_pairs.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()]
    cfg = load_config(overrides=args.overrides)
    reg = ModelRegistry(cfg)
    pipeline = HCVSumPipeline(cfg, reg)
    out = ROOT / "outputs" / (f"planted_{args.tag}" if args.tag else "planted")
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    print(f"overrides: {args.overrides or 'none (config/default.yaml)'}")
    results = []
    for doc_type, name in DOCS.items():
        candidates = [p for p in constructed if p["doc_type"] == doc_type]
        chosen = rng.sample(candidates, PER_DOC)
        text = (ROOT / "data" / "external" / "eval_docs" / name).read_text(encoding="utf-8")
        planted_text, placed = plant(text, chosen)
        t = time.perf_counter()
        result = pipeline.run(planted_text, f"{name} (+{PER_DOC} planted)")
        minutes = (time.perf_counter() - t) / 60
        save_json(result, out / f"{Path(name).stem}.json")

        flags = sorted(result.contradictions.contradictions + result.contradictions.one_sided, key=lambda c: -c.score)
        summary_sents = [s for ss in result.section_summaries for s in ss.sentences]
        print(f"\n== {name}: {len(flags)} flags, {minutes:.1f} min")
        for p in placed:
            if not p["planted"]:
                print(f"  {p['id']:<16} NOT PLANTED: {p['why']}")
                results.append({**p, "caught": None})
                continue
            ea, eb = reg.embedder.encode([p["a"], p["b"]])
            rank, score, kind = None, None, None
            if flags:
                fa = reg.embedder.encode([c.claim_a.text for c in flags])
                fb = reg.embedder.encode([c.claim_b.text for c in flags])
                match = np.maximum(np.minimum(fa @ ea, fb @ eb), np.minimum(fa @ eb, fb @ ea))
                hits = np.flatnonzero(match >= MATCH)
                if hits.size:
                    rank, score, kind = int(hits[0]) + 1, flags[hits[0]].score, flags[hits[0]].kind
            if summary_sents:
                se = reg.embedder.encode(summary_sents)
                a_in, b_in = float((se @ ea).max()), float((se @ eb).max())
            else:
                a_in = b_in = 0.0
            status = f"CAUGHT rank {rank}/{len(flags)} ({kind}, P={score:.2f})" if rank else "missed"
            print(f"  {p['id']:<16} {p['category']:<20} {status:<40} A in summary {a_in:.2f}, B in summary {b_in:.2f}")
            results.append({"id": p["id"], "doc": name, "category": p["category"], "caught": rank is not None,
                            "rank": rank, "flags": len(flags), "score": score, "kind": kind,
                            "a_in_summary": a_in, "b_in_summary": b_in})
    caught = [r for r in results if r.get("caught")]
    print(f"\nplanted contradictions found: {len(caught)} / {sum(r.get('caught') is not None for r in results)}")
    (out / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
