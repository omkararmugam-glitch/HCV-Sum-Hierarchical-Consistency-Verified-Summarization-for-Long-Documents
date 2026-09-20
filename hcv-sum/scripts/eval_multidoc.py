"""Evaluate the multi-document mode on the small synthetic quarterly-update test case.

Usage:  python scripts/eval_multidoc.py        (writes outputs/multidoc_eval.txt and outputs/multidoc/run.json)

Test case: data/multidoc/q{1,2,3}_2026_update.md, ground truth data/multidoc/ground_truth.json (written
before any run). The mode is an unsupervised, retrieval-and-centrality-based approximation inspired by the
goal of Liu & Lapata (2019); this is our own 3-document test, not their benchmark.

A flag is matched to a labelled pair only if its two claims GROUND to the labelled source sentences
(summary claim -> most similar source sentence of its own section; a 3b source sentence is itself).
Reported: for each genuine contradiction, whether it was flagged and whether it stayed a contradiction or
was downgraded; for each legitimate update, whether it was flagged and, if so, whether it was downgraded;
every other flag (a false flag unless downgraded); and which documents the final summary draws on.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "multidoc"


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from hcv_sum.cli import _quiet_third_party
    _quiet_third_party()
    sys.stdout.reconfigure(encoding="utf-8")
    from hcv_sum.config import load_config
    from hcv_sum.models import ModelRegistry
    from hcv_sum.multidoc import MultiDocPipeline
    from hcv_sum.referent import ground_claim
    from hcv_sum.report import render_brief, save_json
    from hcv_sum.scoring import DocumentIndex

    truth = json.loads((DATA / "ground_truth.json").read_text(encoding="utf-8"))
    docs = [(name, (DATA / name).read_text(encoding="utf-8")) for name in truth["documents"]]
    cfg = load_config()
    reg = ModelRegistry(cfg)
    result = MultiDocPipeline(cfg, reg).run(docs)
    save_json(result, ROOT / "outputs" / "multidoc" / "run.json")
    s = result.stats
    index = DocumentIndex.build(result.sections, reg.embedder)
    src = index.sentences
    lines: list[str] = []

    def say(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    say("=== documents ===")
    for d in s["documents"]:
        say(f"  {d['name']:<22} {d['sections']} sections, period marker: {d['period']}")
    say(f"  pooled: {s['sections']} sections, {s['source_sentences']} source sentences, {s['summary_claims']} summary claims; "
        f"compared {s['stage3a_pairs']['compared']} summary pairs (3a) + {s['stage3b_pairs']['compared']} summary-vs-source (3b)")

    def gid(claim, is_source):
        g = ground_claim(claim, index, reg.embedder, is_source=is_source)
        return next(k for k, t in enumerate(src) if t == g.sentence)

    def ids(key):
        return {k for k, t in enumerate(src) if key in t}

    labelled = {name: ("contradiction", ids(a), ids(b)) for name, (a, b) in truth["contradictions"].items()}
    labelled.update({name: ("update", ids(a), ids(b)) for name, (a, b) in truth["superseded"].items()})
    sec_doc = s["section_documents"]
    names = [n for n, _ in docs]
    found = {name: [] for name in labelled}
    others = []
    for c in result.contradictions.all_contradictions:
        i, j = gid(c.claim_a, False), gid(c.claim_b, c.kind == "one_sided")
        hit = next((n for n, (_, sa, sb) in labelled.items() if (i in sa and j in sb) or (j in sa and i in sb)), None)
        (found[hit] if hit else others).append(c)

    say("\n=== labelled pairs ===")
    outcome = {}
    for name, (kind, _, _) in labelled.items():
        flags = found[name]
        if not flags:
            verdict = "NOT FLAGGED" + (" (missed)" if kind == "contradiction" else " (correct: no false flag)")
        else:
            res = {c.resolution for c in flags}
            best = max(flags, key=lambda c: c.score)
            if kind == "contradiction":
                verdict = ("caught, kept as a contradiction" if res - {"possibly_superseded"}
                           else "caught but DOWNGRADED to possibly_superseded (wrong)")
            else:
                verdict = ("flagged and downgraded to possibly_superseded (correct)" if res == {"possibly_superseded"}
                           else "flagged as a CONTRADICTION (wrong)")
            verdict += f"; P={best.score:.2f}, {best.kind}, resolution {sorted(res)}"
        outcome[name] = verdict
        say(f"  {name:<24} [{kind}] {verdict}")

    say(f"\n=== all other flags: {len(others)} ===")
    for c in sorted(others, key=lambda c: -c.score):
        da, db = names[sec_doc[c.claim_a.section_index]], names[sec_doc[c.claim_b.section_index]]
        say(f"  P={c.score:.2f} {c.resolution:<20} {da[:7]} vs {db[:7]}  A: {c.claim_a.text[:70]}")
        say(f"  {'':<31}  B: {c.claim_b.text[:70]}")
    still = sum(c.resolution != "possibly_superseded" for c in others)
    say(f"  -> {still} of {len(others)} other flags remain reported as contradictions (false flags); "
        f"{len(others) - still} downgraded to possibly_superseded")

    say(f"\n=== flags by scope === {s['flags']}")
    say(f"=== final summary === {s['final_sentences']} sentences (salience: {s['salience']}); "
        f"documents cited: {s['documents_cited']}")
    say("\n--brief view of the run:")
    say(render_brief(result))
    (ROOT / "outputs" / "multidoc_eval.txt").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
