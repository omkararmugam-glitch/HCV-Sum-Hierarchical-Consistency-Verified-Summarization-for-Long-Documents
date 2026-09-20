"""Aggregate evaluation across all sample documents against their ground truth.

Usage:  python scripts/score_samples.py [outputs/<tag>] [--truth data/samples/ground_truth.json]

Metrics (defined here, not inherited from anywhere):

  contradiction catch rate   of the documents that DO contain a planted contradiction, the fraction
                             where Stage 3 flagged it -- by the sibling check (3a), the one-sided
                             check (3b) or the source diagnostic. Reported per level too, because
                             3a and 3b fail for different reasons.
  false positive rate        flagged pairs that match no planted contradiction, counted per
                             document and summed. For the contradiction-free document this is the
                             number that must be zero.
  provenance support rate    share of final-summary sentences by status (supported / weakly /
                             unsupported) and how many carry a DISPUTED flag.

A flagged pair counts as catching a planted contradiction when each of its two claims is within
MATCH cosine of a different one of the two planted source sentences. That tolerance exists because
Stage 2 rewrites sentences; it is an evaluation aid and is printed with the results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATCH = 0.6


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?", default=str(ROOT / "outputs" / "cli_json"), type=Path)
    ap.add_argument("--truth", type=Path, default=ROOT / "data" / "samples" / "ground_truth.json")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from hcv_sum.cli import _quiet_third_party
    _quiet_third_party()
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    truth = json.loads(args.truth.read_text(encoding="utf-8"))

    def close(a: str, b: str) -> bool:
        e = embedder.encode([a, b], normalize_embeddings=True)
        return float(e[0] @ e[1]) >= MATCH

    docs = sorted(p for p in args.run_dir.glob("*.json"))
    if not docs:
        print(f"no run JSON found in {args.run_dir}")
        return 1

    print(f"run: {args.run_dir}   ground truth: {args.truth}   match tolerance: cosine >= {MATCH}\n")
    header = f"{'document':<30}{'planted':>8}{'3a':>5}{'3b':>5}{'diag':>6}{'caught':>8}{'FP 3a':>7}{'FP 3b':>7}"
    print(header)
    print("-" * len(header))

    with_planted = caught_any = caught_default = caught_3a = caught_3b = caught_diag = 0
    fp_3a = fp_3b = 0
    clean_docs: list[tuple[str, int]] = []
    statuses: dict[str, int] = {}
    disputed = unsupported_sentences = total_final = 0

    for path in docs:
        d = json.loads(path.read_text(encoding="utf-8"))
        planted = truth.get(path.stem, {}).get("contradictions", [])
        src = [s for sec in d["sections"] for s in sec["sentences"]]
        pairs = []
        for a_sub, b_sub in planted:
            a = next((s for s in src if a_sub[:60] in s), None)
            b = next((s for s in src if b_sub[:60] in s), None)
            if a and b:
                pairs.append((a, b))

        def matches(claim_a: str, claim_b: str) -> bool:
            return any((close(claim_a, a) and close(claim_b, b)) or (close(claim_a, b) and close(claim_b, a))
                       for a, b in pairs)

        c = d["contradictions"]
        hits_3a = [x for x in c["contradictions"] if matches(x["claim_a"]["text"], x["claim_b"]["text"])]
        hits_3b = [x for x in c["one_sided"] if matches(x["claim_a"]["text"], x["claim_b"]["text"])]
        diag = (d.get("source_diagnostic") or {}).get("contradictions", [])
        hits_diag = [x for x in diag if matches(x["claim_a"]["text"], x["claim_b"]["text"])]

        doc_fp_3a = len(c["contradictions"]) - len(hits_3a)
        doc_fp_3b = len(c["one_sided"]) - len(hits_3b)
        fp_3a += doc_fp_3a
        fp_3b += doc_fp_3b

        if pairs:
            with_planted += 1
            caught_3a += bool(hits_3a)
            caught_3b += bool(hits_3b)
            caught_diag += bool(hits_diag)
            caught_any += bool(hits_3a or hits_3b or hits_diag)
            caught_default += bool(hits_3a or hits_3b)
        else:
            clean_docs.append((path.stem, len(c["contradictions"]) + len(c["one_sided"])))

        for p in d["provenance"]:
            statuses[p["status"]] = statuses.get(p["status"], 0) + 1
            total_final += 1
            disputed += any(f.startswith("DISPUTED") for f in p["flags"])
            unsupported_sentences += p["status"] == "unsupported"

        caught = "yes" if (hits_3a or hits_3b or hits_diag) else ("no" if pairs else "-")
        print(f"{path.stem[:29]:<30}{len(pairs):>8}{len(hits_3a):>5}{len(hits_3b):>5}{len(hits_diag):>6}"
              f"{caught:>8}{doc_fp_3a:>7}{doc_fp_3b:>7}")

    print("\n=== CONTRADICTION CATCH RATE (documents with a planted contradiction) ===")
    if with_planted:
        print(f"  DEFAULT PATH (3a or 3b, what every run does)  "
              f"{caught_default}/{with_planted} = {caught_default / with_planted:.0%}")
        print(f"  any level incl. opt-in diagnostic             "
              f"{caught_any}/{with_planted} = {caught_any / with_planted:.0%}")
        print(f"  Stage 3a only      {caught_3a}/{with_planted} = {caught_3a / with_planted:.0%}")
        print(f"  Stage 3b only      {caught_3b}/{with_planted} = {caught_3b / with_planted:.0%}")
        print(f"  source diagnostic  {caught_diag}/{with_planted} = {caught_diag / with_planted:.0%}"
              + ("" if diag else "   (not run: no --diagnose-sources in this run)"))
    else:
        print("  no document in this run has a planted contradiction")

    print("\n=== FALSE POSITIVES ===")
    print(f"  total flagged pairs matching no planted contradiction: {fp_3a} (3a) + {fp_3b} (3b) = {fp_3a + fp_3b}")
    for stem, n in clean_docs:
        verdict = "ZERO false positives, confirmed clean" if n == 0 else f"{n} FALSE POSITIVES"
        print(f"  contradiction-free document {stem}: {verdict}")

    print("\n=== PROVENANCE SUPPORT RATE (all final-summary sentences) ===")
    for status, n in sorted(statuses.items()):
        print(f"  {status:<18} {n:>4} / {total_final} = {n / max(total_final, 1):.0%}")
    print(f"  {'DISPUTED flag':<18} {disputed:>4} / {total_final} = {disputed / max(total_final, 1):.0%}"
          f"   (supported by its own section, contradicted by another)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
