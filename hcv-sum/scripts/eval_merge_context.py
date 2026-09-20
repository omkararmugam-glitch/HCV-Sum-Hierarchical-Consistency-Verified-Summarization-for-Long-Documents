"""Evaluate context-anchored abstractive merging (merging.context_anchoring; after Ou & Lapata 2025).

Usage:  python scripts/eval_merge_context.py [--only 05]   (writes outputs/merge_context_eval.txt / .json)
        --only SUBSTR   run only the documents whose name contains SUBSTR ("bench31" = the synthetic one)

Stage 4 runs AFTER Stage 3, so the setting cannot change Stage 3's flags; Stage 3 is run once per sample
and reported for completeness. What it can change -- and what is measured -- is Stage 4 and 5:

  coverage       sections cited by >= 1 final sentence (Stage 5 citation) / sections with a summary
  introduced     contradictions the merge rounds CREATED (merging.check_between_rounds)
  copied         final sentences that are near-verbatim copies (cosine >= 0.9) of a source sentence that
                 was given ONLY as merge context -- i.e. coverage bought by copying the context through
  cost           Stage 4 wall-clock and peak memory (StageMonitor), generations

Documents: samples 01-05 (their saved Stage 1-2 output, current Stage 3) and the 31-section synthetic
benchmark data/bench/medium_15pages.md (Stages 1-2 run here). Abstractive merging is compared WITHOUT and
WITH context at max_group_sections 3, 5 (the large-document default) and 8, so a result that holds at one
fan-in only is visible as such. DistilBART decodes with beam search (deterministic), so repeated runs of
the same configuration give the same text; differences between rows come from the configuration.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ["cur_small/01_planted_contradiction", "cur_small/02_no_contradiction", "cur_small/03_subtle_contradiction",
           "cur_small/04_contract_term", "new_data_05/05_medium_paper_graphfault"]
# Strict ground truth: a flag is true only if its two claims ground to these source sentences.
STRICT = {
    "01_planted_contradiction": {"C1": (["backlog orders went unfilled and industrial sensor revenue fell 18%"],
                                        ["unrelated to the Monterrey shutdown"])},
    "03_subtle_contradiction": {"C1": (["Every enterprise customer is now running on the new Helix Core platform"],
                                       ["roughly sixty enterprise accounts are still being served"])},
    "04_contract_term": {"C1": (["initial term of twenty-four (24) months"],
                                ["expires at the end of the initial term on December 31, 2026"])},
    "05_medium_paper_graphfault": {
        "C1": (["The automotive dataset spans eighteen months"], ["automotive dataset, spanning twenty-four months"]),
        "C2": (["transfer effectively across plants with different equipment vendors"],
               ["only evaluated between plants using similar underlying", "We have not yet evaluated transfer between plants"]),
        "C3": (["despite differing equipment vendors and process types"],
               ["only evaluated between plants using similar underlying", "We have not yet evaluated transfer between plants"]),
    },
}
FAN_INS = (3, 5, 8)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="")
    ap.add_argument("--merge-model", default="", help="models.merge_summarizer (default: the Stage 2 model)")
    ap.add_argument("--prompt", default="none", choices=["none", "instruct"], help="merging.merge_prompt")
    ap.add_argument("--tag", default="", help="suffix for the output files")
    args = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from hcv_sum.cli import _quiet_third_party
    _quiet_third_party()
    sys.stdout.reconfigure(encoding="utf-8")
    from rerun_stage3 import load_run
    from hcv_sum.anchored_summarization import summarize_sections
    from hcv_sum.config import load_config
    from hcv_sum.contradiction import check_section_summaries, diagnose_source_sections
    from hcv_sum.merging import merge_summaries
    from hcv_sum.models import CountingSummarizer, ModelRegistry
    from hcv_sum.provenance import tag_provenance
    from hcv_sum.referent import ground_claim
    from hcv_sum.resources import StageMonitor
    from hcv_sum.scoring import DocumentIndex
    from hcv_sum.segmentation import segment_document
    from hcv_sum.text_utils import content_tokens

    overrides = [f"merging.merge_prompt={args.prompt}"]
    if args.merge_model:
        overrides.append(f"models.merge_summarizer={args.merge_model}")
    cfg = load_config(overrides=overrides)
    reg = ModelRegistry(cfg)
    merge_model_name = cfg.models.merge_summarizer or cfg.models.summarizer
    emb = reg.embedder
    lines: list[str] = []
    results: dict = {}

    def say(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    def merge_variant(summaries, index, report, label, mode, **overrides):
        mcfg = dataclasses.replace(cfg.merging, **overrides)
        summarizer = CountingSummarizer(reg.merge_summarizer)
        corrected = report.corrected_sentences if report else [s.sentences for s in summaries]
        with StageMonitor() as mon:
            merge = merge_summaries(corrected, [], summarizer, emb, reg.nli, mcfg,
                                    contradiction_cfg=cfg.contradiction, mode=mode, index=index)
        prov = tag_provenance(merge, index, emb, reg.nli, cfg.provenance, report)
        with_summary = {s.section_index for s in summaries if s.sentences}
        cited = {r.citation.section_index for r in prov if r.citation is not None} & with_summary
        copied = 0
        given = [c for r in merge.rounds for group in r.contexts for c in group]
        if given and merge.sentences:
            block_sents = [s for sec in corrected for s in sec]
            fe = emb.encode(merge.sentences)
            ge, be = emb.encode(given), emb.encode(block_sents)
            copied = int((((fe @ ge.T).max(axis=1) >= 0.9) & ((fe @ be.T).max(axis=1) < 0.9)).sum())
        # Softer evidence of context use: content words in the output that occur in the given context but
        # in none of the summaries being merged (so the model can only have taken them from the context).
        out_words = content_tokens(merge.summary)
        block_words = set().union(*(content_tokens(s) for sec in corrected for s in sec)) if corrected else set()
        ctx_words = set().union(*(content_tokens(c) for c in given)) if given else set()
        ctx_only = sorted((out_words & ctx_words) - block_words)
        row = {"label": label, "final": len(merge.sentences), "words": len(merge.summary.split()),
               "cited": len(cited), "sections": len(with_summary), "rounds": len(merge.rounds),
               "introduced": sum(len(r.introduced) for r in merge.rounds), "copied": copied,
               "context_sentences": len(given), "context_only_words": len(ctx_only),
               "context_only_examples": ctx_only[:8], "generations": summarizer.generations,
               "seconds": round(mon.seconds, 1), "peak_mb": round(mon.peak_mb), "start_mb": round(mon.start_mb),
               "supported": sum(r.status == "supported" for r in prov),
               "unsupported": sum(r.status == "unsupported" for r in prov),
               "disputed": sum(any(f.startswith("DISPUTED") for f in r.flags) for r in prov)}
        return row

    def show(rows):
        say(f"  merge model: {merge_model_name}, merge prompt: {cfg.merging.merge_prompt}")
        say(f"  {'configuration':<34}{'cited':>10}{'final':>7}{'words':>7}{'rounds':>7}{'introduced':>11}"
            f"{'copied':>8}{'ctx-only words':>15}{'ctx':>5}{'gens':>6}{'sec':>7}{'peak MB':>9}{'unsupported':>12}")
        for r in rows:
            say(f"  {r['label']:<34}{r['cited']:>5} / {r['sections']:<3}{r['final']:>7}{r['words']:>7}{r['rounds']:>7}"
                f"{r['introduced']:>11}{r['copied']:>8}{r['context_only_words']:>15}{r['context_sentences']:>5}"
                f"{r['generations']:>6}"
                f"{r['seconds']:>7.1f}{r['peak_mb']:>9}{r['unsupported']:>12}")

    def coverage_sweep(summaries, index, report):
        rows = [merge_variant(summaries, index, report, "extractive (reference)", "extractive")]
        for k in FAN_INS:
            rows.append(merge_variant(summaries, index, report, f"abstractive g{k}, no context", "abstractive",
                                      max_group_sections=k, context_anchoring=False))
            rows.append(merge_variant(summaries, index, report, f"abstractive g{k}, WITH context", "abstractive",
                                      max_group_sections=k, context_anchoring=True))
        return rows

    # ------------------------------------------------------------ 31-section synthetic benchmark
    if not args.only or args.only in "bench31":
        say("=== 1. Coverage: 31-section synthetic benchmark (data/bench/medium_15pages.md) ===")
        text = (ROOT / "data" / "bench" / "medium_15pages.md").read_text(encoding="utf-8")
        sections = segment_document(text, emb, reg.summarizer.count_tokens, cfg.segmentation)
        index = DocumentIndex.build(sections, emb)
        summaries = summarize_sections(sections, index, reg.summarizer, emb, cfg.summarization)
        say(f"  {len(sections)} sections; Stage 3 not needed here (action=flag: merge input = the section summaries)")
        rows = coverage_sweep(summaries, index, None)
        show(rows)
        results["bench31"] = rows

    # ------------------------------------------------------------ samples 01-05
    say("\n=== 2. Samples 01-05: Stage 3 (unaffected by Stage 4) and Stage 4/5 with and without context ===")
    stage3 = []
    for rel in [r for r in SAMPLES if not args.only or args.only in r]:
        path = ROOT / "outputs" / f"{rel}.json"
        stem = path.stem
        _, sections, summaries = load_run(path)
        index = DocumentIndex.build(sections, emb)
        report = check_section_summaries(summaries, index, emb, reg.nli, cfg.contradiction)
        diag = diagnose_source_sections(index, emb, reg.nli, cfg.contradiction)
        src = [s for sec in sections for s in sec.sentences]

        def ids(keys):
            return {k for k, s in enumerate(src) if any(key in s for key in keys)}
        truth = {name: (ids(a), ids(b)) for name, (a, b) in STRICT.get(stem, {}).items()}

        def gid(claim, is_source):
            g = ground_claim(claim, index, emb, is_source=is_source)
            return next(k for k, s in enumerate(src) if s == g.sentence)

        def label(c):
            i = gid(c.claim_a, c.kind == "diagnostic")
            j = gid(c.claim_b, c.kind in ("one_sided", "diagnostic"))
            return next((n for n, (sa, sb) in truth.items() if (i in sa and j in sb) or (j in sa and i in sb)), None)

        l3a = [label(c) for c in report.contradictions]
        l3b = [label(c) for c in report.one_sided]
        ldg = [label(c) for c in diag.contradictions]
        caught = {name: ("3a" if name in l3a else "3b" if name in l3b else "diag" if name in ldg else "missed")
                  for name in truth}
        stage3.append({"doc": stem, "planted": len(truth), "caught": caught,
                       "fp_3a": sum(x is None for x in l3a), "fp_3b": sum(x is None for x in l3b),
                       "pairs": report.pairs_checked + report.one_sided_checked})
        say(f"\n  {stem}: Stage 3 -- planted {len(truth)}: {caught or '-'}; false flags 3a {stage3[-1]['fp_3a']}, "
            f"3b {stage3[-1]['fp_3b']} over {stage3[-1]['pairs']} compared pairs")
        rows = ([merge_variant(summaries, index, report, "extractive (default)", "extractive"),
                 merge_variant(summaries, index, report, "abstractive g5, no context", "abstractive",
                               max_group_sections=5, context_anchoring=False),
                 merge_variant(summaries, index, report, "abstractive g5, WITH context", "abstractive",
                               max_group_sections=5, context_anchoring=True)]
                if not stem.startswith("05") else coverage_sweep(summaries, index, report))
        show(rows)
        results[stem] = {"stage3": stage3[-1], "merge": rows}

    say("\n=== 3. Stage 3 summary (unchanged by any merge setting: Stage 4 runs after Stage 3) ===")
    if not stage3:
        (ROOT / "outputs" / f"merge_context_eval{'_' + args.tag if args.tag else ''}.txt").write_text(
            "\n".join(lines), encoding="utf-8")
        return 0
    planted = sum(s["planted"] for s in stage3)
    default_path = sum(v in ("3a", "3b") for s in stage3 for v in s["caught"].values())
    any_level = sum(v != "missed" for s in stage3 for v in s["caught"].values())
    fp = sum(s["fp_3a"] + s["fp_3b"] for s in stage3)
    pairs = sum(s["pairs"] for s in stage3)
    say(f"  planted {planted}: caught by 3a {sum(v == '3a' for s in stage3 for v in s['caught'].values())}, "
        f"by 3b only {sum(v == '3b' for s in stage3 for v in s['caught'].values())}, default path {default_path}, "
        f"any level incl. diagnostic {any_level}; false flags {fp} over {pairs} pairs "
        f"({100 * fp / max(pairs, 1):.1f} per 100)")
    suffix = f"_{args.tag}" if args.tag else ""
    (ROOT / "outputs" / f"merge_context_eval{suffix}.txt").write_text("\n".join(lines), encoding="utf-8")
    (ROOT / "outputs" / f"merge_context_eval{suffix}.json").write_text(
        json.dumps(results, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    raise SystemExit(main())
