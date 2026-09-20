"""Validate the grounded referent re-check against its pre-registered criteria.

Usage:  python scripts/eval_referent_check.py

Design and pass criteria: data/eval/REFERENT_CHECK_PREREGISTRATION.md (written before this ran).
Nothing in the pipeline changes; this only measures what the check WOULD confirm or reject.

(a) MUST KEEP  true flags: samples 01, 03, 05 (Stage 3 re-run on their saved summaries with the current
               default config, incl. reporting-frame stripping; sample 05 also through the source
               diagnostic for C3), and the planted pairs in the two completed planted real documents.
(b) SHOULD REMOVE  sample 05's pattern-A flags ("referent lost in Stage 2 compression"); the other
               false-flag patterns are reported alongside.
(c) EXTERNAL   every flag on the four real documents (outputs/real_docs_v2_gate_off), untouched by the
               design. Their saved flags are re-checked as they are (those runs predate frame stripping).
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def flags_from_json(items):
    from hcv_sum.types import ClaimRef, Contradiction
    out = []
    for x in items:
        c = Contradiction(ClaimRef(**x["claim_a"]), ClaimRef(**x["claim_b"]), x["score"], x["score_ab"], x["score_ba"])
        c.kind = x.get("kind", "summary")
        out.append(c)
    return out


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from hcv_sum.cli import _quiet_third_party
    _quiet_third_party()
    sys.stdout.reconfigure(encoding="utf-8")
    from analyze_false_flags import RULES
    from rerun_stage3 import load_run
    from hcv_sum.config import load_config
    from hcv_sum.contradiction import check_section_summaries, diagnose_source_sections
    from hcv_sum.models import ModelRegistry
    from hcv_sum.referent import referent_recheck
    from hcv_sum.scoring import DocumentIndex
    from hcv_sum.types import Section

    cfg = load_config()
    reg = ModelRegistry(cfg)
    emb, c = reg.embedder, cfg.contradiction
    truth = json.loads((ROOT / "data" / "samples" / "ground_truth.json").read_text(encoding="utf-8"))
    lines: list[str] = []

    def say(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    def matches(flag, a_text, b_text, tol):
        fa, fb = emb.encode([flag.claim_a.text, flag.claim_b.text])
        ea, eb = emb.encode([a_text, b_text])
        return max(min(fa @ ea, fb @ eb), min(fa @ eb, fb @ ea)) >= tol

    def show(label, flag, r):
        say(f"    {label}: detector {flag.score:.3f} -> re-check {r['score']:.3f} "
            f"({r['p_a_context']:.2f}/{r['p_b_context']:.2f}) {'CONFIRMED' if r['confirmed'] else 'REJECTED'}")
        say(f"      A: {flag.claim_a.text[:120]}")
        say(f"         grounded as: {r['grounding_a'].as_premise()[:160]}")
        say(f"      B: {flag.claim_b.text[:120]}")
        say(f"         grounded as: {r['grounding_b'].as_premise()[:160]}")

    must_keep: list[tuple[str, bool]] = []

    # ---------------------------------------------------------------- (a) + (b): samples 01, 03, 05
    say("=== (a) true flags on the samples, and (b) sample 05's false flags ===")
    for rel in ("cur_small/01_planted_contradiction", "cur_small/03_subtle_contradiction",
                "new_data_05/05_medium_paper_graphfault"):
        path = ROOT / "outputs" / f"{rel}.json"
        _, sections, summaries = load_run(path)
        index = DocumentIndex.build(sections, emb)
        report = check_section_summaries(summaries, index, emb, reg.nli, c)
        flags = report.contradictions + report.one_sided
        src = [s for sec in sections for s in sec.sentences]
        planted = [(next(s for s in src if ka in s), next(s for s in src if kb in s))
                   for ka, kb in truth[path.stem]["contradictions"]]
        results = referent_recheck(flags, index, emb, reg.nli, c)
        say(f"\n  {path.stem}: {len(flags)} flags after frame stripping")
        groups = defaultdict(list)
        for f, r in zip(flags, results):
            hit = [p for p, (a, b) in enumerate(planted) if matches(f, a, b, 0.60)]
            if hit:
                name = f"C{hit[0] + 1}" if path.stem.startswith("05") else "planted"
                show(f"TRUE {name}", f, r)
                must_keep.append((f"{path.stem[:2]} {name} ({f.kind})", r["confirmed"]))
                continue
            for pname, rule, _ in RULES:
                if rule(f.claim_a.text, f.claim_b.text):
                    groups[pname].append((f, r))
                    break
        if path.stem.startswith("05"):
            diag = diagnose_source_sections(index, emb, reg.nli, c)
            dres = referent_recheck(diag.contradictions, index, emb, reg.nli, c)
            for f, r in zip(diag.contradictions, dres):
                for p, (a, b) in enumerate(planted):
                    if p == 2 and matches(f, a, b, 0.60):             # C3 is only reachable here
                        show("TRUE C3 (source diagnostic)", f, r)
                        must_keep.append(("05 C3 (diagnostic)", r["confirmed"]))
                        break
            say("\n  sample 05 false flags by pattern: rejected by the re-check / total")
            for pname, _, _ in RULES:
                g = groups.get(pname, [])
                if g:
                    rej = sum(not r["confirmed"] for _, r in g)
                    say(f"    {pname:<56}{rej:>3} / {len(g):<3}")
            pattern_a = groups.get(RULES[0][0], [])
            b_rejected, b_total = sum(not r["confirmed"] for _, r in pattern_a), len(pattern_a)
            other = [x for name, g in groups.items() if name != RULES[0][0] for x in g]
            say("\n  pattern A examples:")
            for f, r in sorted(pattern_a, key=lambda x: -x[0].score)[:3]:
                show("A", f, r)
            say("\n  confirmed false flags from other patterns (highest re-check score):")
            for f, r in sorted([x for x in other if x[1]["confirmed"]], key=lambda x: -x[1]["score"])[:3]:
                show("still flagged", f, r)

    # ---------------------------------------------------------------- (a): planted real documents
    say("\n=== (a) planted contradictions in real documents (outputs/planted_gate_off) ===")
    constructed = [json.loads(line) for line in (ROOT / "data" / "eval" / "constructed_pairs.jsonl")
                   .read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = random.Random(7)                           # same draw as scripts/plant_eval.py
    chosen = {}
    for doc_type in ("research", "financial", "contract", "transcript"):
        chosen[doc_type] = rng.sample([p for p in constructed if p["doc_type"] == doc_type], 3)
    for doc_type, stem in (("research", "research_refnli"), ("financial", "financial_fomc_minutes")):
        d, sections, _ = load_run(ROOT / "outputs" / "planted_gate_off" / f"{stem}.json")
        index = DocumentIndex.build([Section(s.index, s.title, s.sentences, s.sentence_offset, s.origin)
                                     for s in sections], emb)
        flags = flags_from_json(d["contradictions"]["contradictions"] + d["contradictions"]["one_sided"])
        results = referent_recheck(flags, index, emb, reg.nli, c)
        say(f"\n  {stem}: {len(flags)} saved flags")
        for p in chosen[doc_type]:
            hits = [(f, r) for f, r in zip(flags, results) if matches(f, p["a"], p["b"], 0.70)]
            if not hits:
                say(f"    {p['id']} ({p['category']}): not flagged by the detector -- nothing to keep")
                continue
            f, r = max(hits, key=lambda x: x[0].score)
            show(f"TRUE {p['id']} ({p['category']})", f, r)
            must_keep.append((f"{stem} {p['id']}", r["confirmed"]))
        fp = [(f, r) for f, r in zip(flags, results)
              if not any(matches(f, p["a"], p["b"], 0.70) for p in chosen[doc_type])]
        say(f"    other flags rejected: {sum(not r['confirmed'] for _, r in fp)} / {len(fp)}")

    # ---------------------------------------------------------------- (c): real documents
    say("\n=== (c) external: all flags on the four real documents (outputs/real_docs_v2_gate_off) ===")
    c_total = c_rej = 0
    for stem in ("research_refnli", "financial_fomc_minutes", "contract_imageware", "transcript_fomc_presconf"):
        d, sections, _ = load_run(ROOT / "outputs" / "real_docs_v2_gate_off" / f"{stem}.json")
        index = DocumentIndex.build(sections, emb)
        flags = flags_from_json(d["contradictions"]["contradictions"] + d["contradictions"]["one_sided"])
        results = referent_recheck(flags, index, emb, reg.nli, c)
        rej = sum(not r["confirmed"] for r in results)
        c_total, c_rej = c_total + len(flags), c_rej + rej
        say(f"  {stem:<28} rejected {rej:>4} / {len(flags):<4} ({100 * rej / max(len(flags), 1):.0f}%)")

    # ---------------------------------------------------------------- verdict
    kept = sum(ok for _, ok in must_keep)
    say("\n=== pre-registered criteria ===")
    for name, ok in must_keep:
        say(f"    {'kept   ' if ok else 'LOST   '} {name}")
    crit1 = kept == len(must_keep)
    crit2 = b_total > 0 and b_rejected / b_total >= 0.5
    crit3 = c_total > 0 and c_rej / c_total >= 0.25
    say(f"  1. true flags confirmed: {kept} / {len(must_keep)}  -> {'PASS' if crit1 else 'FAIL'}")
    say(f"  2. pattern-A flags rejected: {b_rejected} / {b_total}  -> {'PASS' if crit2 else 'FAIL'}")
    say(f"  3. real-document flags rejected: {c_rej} / {c_total} ({100 * c_rej / max(c_total, 1):.0f}%) "
        f"-> {'PASS' if crit3 else 'FAIL'}")
    (ROOT / "outputs" / "referent_check_eval.txt").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
