"""Sample cross-section sentence pairs from the four real documents, for hand labelling.

Usage:  python scripts/build_pair_candidates.py [--per-doc 16] [--seed 13]

Runs Stage 1 exactly as the pipeline does, then samples pairs of source sentences from DIFFERENT
sections that pass the pipeline's own gates (both are claims, cosine >= pair_min_similarity).
Per document it takes the highest-similarity pairs (the look-alikes most likely to be flagged) plus a
seeded random sample from the rest of the eligible range (the typical candidate).

The NLI model is deliberately NOT loaded: labels must be written without knowing what the model
thinks, or the evaluation would measure agreement with the model rather than correctness.
Output: data/eval/candidates.jsonl (unlabelled).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = {
    "research": "research_refnli.md",
    "financial": "financial_fomc_minutes.md",
    "contract": "contract_imageware.txt",
    "transcript": "transcript_fomc_presconf.txt",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-doc", type=int, default=16)
    ap.add_argument("--top", type=int, default=8, help="of which: highest-similarity pairs")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from hcv_sum.cli import _quiet_third_party
    _quiet_third_party()
    from transformers import AutoTokenizer

    from hcv_sum.config import load_config
    from hcv_sum.contradiction import is_claim
    from hcv_sum.models import SentenceTransformerEmbedder
    from hcv_sum.scoring import DocumentIndex
    from hcv_sum.segmentation import segment_document

    cfg = load_config()
    embedder = SentenceTransformerEmbedder(cfg.models.embedder, cfg.models.device, cfg.models.batch_size)
    tok = AutoTokenizer.from_pretrained(cfg.models.summarizer)       # tokenizer only; no generation

    def count(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    rng = random.Random(args.seed)
    out_rows = []
    for doc_type, name in DOCS.items():
        text = (ROOT / "data" / "external" / "eval_docs" / name).read_text(encoding="utf-8")
        sections = segment_document(text, embedder, count, cfg.segmentation)
        index = DocumentIndex.build(sections, embedder)
        titles = {s.index: s.title or f"(segment {s.index})" for s in sections}
        claims = [i for i, s in enumerate(index.sentences) if is_claim(s, cfg.contradiction)]
        sims = index.embeddings @ index.embeddings.T

        eligible = [(float(sims[i, j]), i, j) for a, i in enumerate(claims) for j in claims[a + 1:]
                    if index.section_of[i] != index.section_of[j] and sims[i, j] >= cfg.contradiction.pair_min_similarity]
        eligible.sort(key=lambda t: -t[0])

        chosen, used = [], {}
        def take(item) -> bool:
            _, i, j = item
            if used.get(i, 0) >= 2 or used.get(j, 0) >= 2:        # no sentence dominates the sample
                return False
            used[i] = used.get(i, 0) + 1
            used[j] = used.get(j, 0) + 1
            chosen.append(item)
            return True

        for item in eligible:
            if len(chosen) >= args.top:
                break
            take(item)
        rest = [e for e in eligible if e not in chosen]
        rng.shuffle(rest)
        for item in rest:
            if len(chosen) >= args.per_doc:
                break
            take(item)

        for k, (sim, i, j) in enumerate(sorted(chosen, key=lambda t: -t[0]), 1):
            out_rows.append({
                "id": f"{doc_type}-n{k:02d}", "doc_type": doc_type, "doc": name, "origin": "natural",
                "a_section": titles[int(index.section_of[i])], "b_section": titles[int(index.section_of[j])],
                "a": index.sentences[i], "b": index.sentences[j], "cosine": round(sim, 3),
                "sampled_as": "top_similarity" if (sim, i, j) in chosen[: args.top] else "random_eligible",
            })
        print(f"{doc_type:<11} {len(sections):>3} sections, {len(claims):>4} claims, "
              f"{len(eligible):>6} eligible cross-section pairs -> {len(chosen)} sampled")

    out = ROOT / "data" / "eval" / "candidates.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n", encoding="utf-8")
    print(f"wrote {len(out_rows)} candidate pairs to {out} (unlabelled; NLI not run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
