"""Condense a run folder (outputs/<tag>/*.json) and score Stage 3 against data/samples/ground_truth.json.

Usage:  python scripts/digest_run.py outputs/run2_fixed

Stage 3 scoring
- source level : a flagged diagnostic pair whose two sentences contain the two ground-truth substrings.
- summary level: a flagged summary pair whose sentences are each semantically close (cosine >= 0.6,
                 MiniLM) to one of the two ground-truth source sentences. Every other flagged pair
                 counts as a false positive. The 0.6 cut is only an evaluation aid, stated here openly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hcv_sum.cli import _quiet_third_party

ROOT = Path(__file__).resolve().parents[1]
MATCH = 0.6


def main() -> int:
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "outputs" / "run2_fixed")
    _quiet_third_party()
    sys.stdout.reconfigure(encoding="utf-8")
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    truth = json.loads((ROOT / "data" / "samples" / "ground_truth.json").read_text(encoding="utf-8"))

    for path in sorted(run_dir.glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        gt = truth.get(path.stem, {}).get("contradictions", [])
        sources = [s for sec in d["sections"] for s in sec["sentences"]]
        print(f"\n===== {d['document_name']}  (timings {d['timings']})")
        for sec, ss in zip(d["sections"], d["section_summaries"]):
            dropped = f" dropped={[w.split(' (')[0] for _, w in ss['dropped']]}" if ss["dropped"] else ""
            print(f"  sec{sec['index']} {len(sec['sentences'])}->{len(ss['sentences'])} sents{dropped}")

        # ---- source-level diagnostic ----
        diag = d["source_diagnostic"]["contradictions"] if d["source_diagnostic"] else []
        gt_src = []
        for a_sub, b_sub in gt:
            a = next(s for s in sources if a_sub in s)
            b = next(s for s in sources if b_sub in s)
            gt_src.append((a, b))
        src_tp = [c for c in diag if any({c["claim_a"]["text"], c["claim_b"]["text"]} == {a, b} for a, b in gt_src)]
        print(f"  STAGE 3 source diagnostic: flagged {len(diag)} | planted found {len(src_tp)}/{len(gt)} | "
              f"other flags {len(diag) - len(src_tp)}")
        for c in diag:
            mark = "TRUE " if c in src_tp else "false"
            print(f"     [{mark}] {c['score']:.2f} ({c['score_ab']:.2f}/{c['score_ba']:.2f}) "
                  f"{c['claim_a']['text'][:60]} || {c['claim_b']['text'][:60]}")

        # ---- summary-level ----
        flagged = d["contradictions"]["contradictions"]
        tp = []
        for c in flagged:
            texts = [c["claim_a"]["text"], c["claim_b"]["text"]]
            for a, b in gt_src:
                e = embedder.encode(texts + [a, b], normalize_embeddings=True)
                if (e[0] @ e[2] >= MATCH and e[1] @ e[3] >= MATCH) or (e[0] @ e[3] >= MATCH and e[1] @ e[2] >= MATCH):
                    tp.append(c)
                    break
        in_summaries = []
        all_summary = [s for ss in d["section_summaries"] for s in ss["sentences"]]
        for a, b in gt_src:
            e_sum = embedder.encode(all_summary, normalize_embeddings=True)
            e_ab = embedder.encode([a, b], normalize_embeddings=True)
            in_summaries.append(((e_sum @ e_ab[0]).max() >= MATCH, (e_sum @ e_ab[1]).max() >= MATCH))
        print(f"  STAGE 3 summaries: checked {d['contradictions']['pairs_checked']}/{d['contradictions']['pairs_total']} "
              f"| flagged {len(flagged)} | planted found {len(tp)}/{len(gt)} | false positives {len(flagged) - len(tp)}"
              f" | planted claims present in section summaries (A, B): {in_summaries}")
        for c in flagged:
            mark = "TRUE " if c in tp else "false"
            print(f"     [{mark}] {c['score']:.2f} {c['resolution']:<10} {c['claim_a']['text'][:55]} || "
                  f"{c['claim_b']['text'][:55]}  ({c['support_a']:.2f} vs {c['support_b']:.2f})")

        one = d["contradictions"].get("one_sided", [])
        one_tp = []
        for c in one:
            e = embedder.encode([c["claim_a"]["text"]], normalize_embeddings=True)[0]
            for a, b in gt_src:
                ea, eb = embedder.encode([a, b], normalize_embeddings=True)
                if (c["claim_b"]["text"] == b and e @ ea >= MATCH) or (c["claim_b"]["text"] == a and e @ eb >= MATCH):
                    one_tp.append(c)
                    break
        print(f"  STAGE 3b one-sided: checked {d['contradictions'].get('one_sided_checked', 0)} | flagged {len(one)} | "
              f"planted found {len(one_tp)}/{len(gt)} | false positives {len(one) - len(one_tp)}")
        for c in one:
            mark = "TRUE " if c in one_tp else "false"
            print(f"     [{mark}] {c['score']:.2f} ({c['score_ab']:.2f}/{c['score_ba']:.2f}) {c['resolution']:<10} "
                  f"{c['claim_a']['text'][:55]} || {c['claim_b']['text'][:55]}  ({c['support_a']:.2f} vs {c['support_b']:.2f})")
        print(f"  FINAL ({len(d['merge']['sentences'])} sents, mode={d['merge']['mode']}): {d['merge']['summary']}")
        statuses = [p["status"] for p in d["provenance"]]
        covered = sorted({p["citation"]["section_index"] for p in d["provenance"] if p["citation"]})
        print(f"  provenance: {statuses} | sections cited: {covered} of {len(d['sections'])} | "
              f"resurrected: {len(d['merge']['resurrected'])}")
        for p in d["provenance"]:
            for note in p["stage3_notes"] + p["flags"]:
                print(f"     S{p['index']} note: {note[:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
