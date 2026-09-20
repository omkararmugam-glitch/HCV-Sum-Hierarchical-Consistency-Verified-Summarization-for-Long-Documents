"""Re-run ONLY Stage 3 (and the source diagnostic) on saved runs, under different Stage 3 settings.

Usage:  python scripts/rerun_stage3.py [--set contradiction.key=value ...] [--tag NAME]

Stages 1-2 are deterministic and unaffected by Stage 3 settings, so their saved output is reused
(regenerating the summaries is the slow part). Inputs: outputs/cur_small/0[1-4]*.json and
outputs/new_data_05/05*.json -- the same current-default runs used in FINDINGS 12.2.

Reports per document: flags (3a + 3b), true flags, false flags, and for every planted contradiction
whether the default path (3a/3b) and the source diagnostic caught it, with the best matching score.
A flag is true when each side is within cosine MATCH of a different side of a planted pair (same rule
as score_samples.py).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATCH = 0.60
RUNS = ["cur_small/01_planted_contradiction", "cur_small/02_no_contradiction", "cur_small/03_subtle_contradiction",
        "cur_small/04_contract_term", "new_data_05/05_medium_paper_graphfault"]


def load_run(path: Path):
    from hcv_sum.types import RetrievedSentence, Section, SectionSummary
    d = json.loads(path.read_text(encoding="utf-8"))
    sections = [Section(x["index"], x["title"], x["sentences"], x["sentence_offset"], x["origin"]) for x in d["sections"]]
    summaries = []
    for x in d["section_summaries"]:
        ctx = [RetrievedSentence(**c) for c in x["context"]]
        summaries.append(SectionSummary(x["section_index"], x["model_input"], x["summary"], list(x["sentences"]), ctx,
                                        x["generated"], x["input_truncated"], x.get("context_leaks", []),
                                        [tuple(t) for t in x.get("dropped", [])], x.get("notes", []),
                                        x.get("protected", [])))
    return d, sections, summaries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-diagnostic", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    from hcv_sum.cli import _quiet_third_party
    _quiet_third_party()
    sys.stdout.reconfigure(encoding="utf-8")
    from hcv_sum.config import load_config
    from hcv_sum.contradiction import check_section_summaries, diagnose_source_sections
    from hcv_sum.models import ModelRegistry
    from hcv_sum.scoring import DocumentIndex

    cfg = load_config(overrides=args.overrides)
    reg = ModelRegistry(cfg)
    emb = reg.embedder
    truth = json.loads((ROOT / "data" / "samples" / "ground_truth.json").read_text(encoding="utf-8"))
    print(f"Stage 3 re-run, overrides: {args.overrides or 'none'}")
    print(f"  {'document':<30}{'flags':>7}{'true':>6}{'false':>7}   planted contradictions: default path | diagnostic")
    out = {}
    for rel in RUNS:
        path = ROOT / "outputs" / f"{rel}.json"
        d, sections, summaries = load_run(path)
        stem = path.stem
        index = DocumentIndex.build(sections, emb)
        report = check_section_summaries(summaries, index, emb, reg.nli, cfg.contradiction)
        diag = None if args.no_diagnostic else diagnose_source_sections(index, emb, reg.nli, cfg.contradiction)
        flags = report.contradictions + report.one_sided
        src = [s for sec in sections for s in sec.sentences]
        planted = truth.get(stem, {}).get("contradictions", [])
        pvecs = []
        for ka, kb in planted:
            a = next(s for s in src if ka in s)
            b = next(s for s in src if kb in s)
            pvecs.append(emb.encode([a, b]))

        def best_match(fl):
            """For each planted pair: best score among flags matching it, else None."""
            res = [None] * len(pvecs)
            if not fl:
                return res, [False] * 0
            fa = emb.encode([c.claim_a.text for c in fl])
            fb = emb.encode([c.claim_b.text for c in fl])
            is_true = [False] * len(fl)
            for p, (ea, eb) in enumerate(pvecs):
                m = [max(min(fa[i] @ ea, fb[i] @ eb), min(fa[i] @ eb, fb[i] @ ea)) >= MATCH for i in range(len(fl))]
                hits = [fl[i].score for i in range(len(fl)) if m[i]]
                res[p] = max(hits) if hits else None
                is_true = [x or y for x, y in zip(is_true, m)]
            return res, is_true

        default_best, is_true = best_match(flags)
        diag_best, _ = best_match(diag.contradictions if diag else [])
        n_true = sum(is_true)
        cells = " ".join(f"C{p + 1} {('%.2f' % default_best[p]) if default_best[p] is not None else 'miss'}"
                         f"|{('%.2f' % diag_best[p]) if diag_best[p] is not None else ('miss' if diag else '-')}"
                         for p in range(len(pvecs)))
        print(f"  {stem:<30}{len(flags):>7}{n_true:>6}{len(flags) - n_true:>7}   {cells or '(none planted)'}")
        orig = d["contradictions"]
        out[stem] = {"flags": len(flags), "true": n_true, "false": len(flags) - n_true,
                     "saved_flags": len(orig["contradictions"]) + len(orig["one_sided"]),
                     "default": default_best, "diagnostic": diag_best}
    total = {k: sum(v[k] for v in out.values()) for k in ("flags", "true", "false", "saved_flags")}
    print(f"  {'TOTAL':<30}{total['flags']:>7}{total['true']:>6}{total['false']:>7}   "
          f"(flags in the saved runs: {total['saved_flags']})")
    if args.tag:
        (ROOT / "outputs" / f"rerun_stage3_{args.tag}.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
