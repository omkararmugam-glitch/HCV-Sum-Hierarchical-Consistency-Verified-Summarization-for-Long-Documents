"""Stage 4 trade-off: final summary length vs section coverage, across merge configurations.

Usage:  python scripts/merge_sweep.py data/bench/medium_50sections.md

Runs the full default pipeline once (saving it to outputs/scale/<stem>.json -- this is also the
medium-document scale report), then re-runs ONLY Stage 4 (+ Stage 5 provenance, to measure which
sections the final summary cites) on the same Stage 3 output under each configuration below.

Coverage = sections cited by at least one final sentence / sections that had any summary sentence.
Citations come from Stage 5 (best-entailed source sentence), so a final sentence that cites nothing
(unsupported) covers nothing.

Caveat: the medium benchmark is TEMPLATED prose (scripts/make_benchmark_document.py): every section
reads alike, which makes DistilBART's choices close to arbitrary. Numbers here show the mechanics of
each configuration; the coverage figures must be re-checked on a real long document.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

VARIANTS = [
    ("extractive, all sentences (small-doc default)", "extractive", {}),
    ("extractive, 2 per section", "extractive", {"extractive_max_per_section": 2}),
    ("extractive, 1 per section", "extractive", {"extractive_max_per_section": 1}),
    ("abstractive, group 2", "abstractive", {"max_group_sections": 2}),
    ("abstractive, group 3", "abstractive", {"max_group_sections": 3}),
    ("abstractive, group 5 (large-doc default)", "abstractive", {"max_group_sections": 5}),
    ("abstractive, group 8", "abstractive", {"max_group_sections": 8}),
    ("abstractive, packed by token budget only", "abstractive", {"max_group_sections": 0}),
    ("abstractive, group 5, 1 round only", "abstractive", {"max_group_sections": 5, "max_rounds": 1}),
    ("abstractive, group 5 + coverage guard", "abstractive", {"max_group_sections": 5, "coverage_guard": True}),
    ("abstractive, group 3 + coverage guard", "abstractive", {"max_group_sections": 3, "coverage_guard": True}),
]


def main() -> int:
    doc = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "data" / "bench" / "medium_50sections.md")
    sys.path.insert(0, str(ROOT / "src"))
    from hcv_sum.cli import _quiet_third_party
    _quiet_third_party()
    sys.stdout.reconfigure(encoding="utf-8")
    from hcv_sum.config import load_config
    from hcv_sum.merging import merge_summaries
    from hcv_sum.models import CountingSummarizer, ModelRegistry
    from hcv_sum.pipeline import HCVSumPipeline
    from hcv_sum.provenance import tag_provenance
    from hcv_sum.report import render_limits, save_json
    from hcv_sum.scoring import DocumentIndex

    cfg = load_config()
    reg = ModelRegistry(cfg)
    out_dir = ROOT / "outputs" / "scale"
    out_dir.mkdir(parents=True, exist_ok=True)
    text = doc.read_text(encoding="utf-8")
    beats: list[str] = []
    result = HCVSumPipeline(cfg, reg).run(text, doc.name, heartbeat=lambda m: (beats.append(m), print(m, flush=True)),
                                          checkpoint_dir=str(out_dir / "checkpoints"))
    save_json(result, out_dir / f"{doc.stem}.json")
    s = result.stats
    print(f"\n=== full default run: {doc.name} ===")
    print(f"sections {s['sections']}, source sentences {s['source_sentences']}, summary claims {s['summary_claims']}, "
          f"final sentences {s['final_sentences']}, total {s['seconds_total']} s")
    print(f"{'stage':<32}{'seconds':>9}{'peak MB':>9}{'NLI pairs':>11}{'generations':>13}")
    for name, st in s["per_stage"].items():
        print(f"{name:<32}{st['seconds']:>9.1f}{st['peak_rss_mb']:>9.0f}{st['nli_pairs']:>11}{st['generations']:>13}")
    print(f"process peak RSS: {s['process_peak_rss_mb']} MB")
    print(render_limits(result))
    print(f"progress lines emitted: {len(beats)}")

    index = DocumentIndex.build(result.sections, reg.embedder)
    rejected = [c.rejected.text for c in result.contradictions.all_contradictions if c.rejected is not None]
    with_summary = {x.section_index for x in result.section_summaries if x.sentences}
    rows = []
    print(f"\n=== Stage 4 sweep ({len(with_summary)} sections have summary sentences) ===")
    print(f"{'configuration':<46}{'final sents':>12}{'words':>7}{'sections cited':>16}{'gens':>6}{'rounds':>7}{'sec':>7}")
    for label, mode, overrides in VARIANTS:
        mcfg = dataclasses.replace(cfg.merging, **overrides)
        summarizer = CountingSummarizer(reg.summarizer)
        t = time.perf_counter()
        merge = merge_summaries(result.contradictions.corrected_sentences, rejected, summarizer, reg.embedder, reg.nli,
                                mcfg, contradiction_cfg=cfg.contradiction, mode=mode)
        seconds = time.perf_counter() - t
        prov = tag_provenance(merge, index, reg.embedder, reg.nli, cfg.provenance, result.contradictions)
        cited = {r.citation.section_index for r in prov if r.citation is not None} & with_summary
        words = len(merge.summary.split())
        rows.append({"configuration": label, "mode": mode, **overrides, "final_sentences": len(merge.sentences),
                     "words": words, "sections_cited": len(cited), "sections": len(with_summary),
                     "generations": summarizer.generations, "rounds": len(merge.rounds), "seconds": round(seconds, 1),
                     "unsupported": sum(r.status == "unsupported" for r in prov)})
        print(f"{label:<46}{len(merge.sentences):>12}{words:>7}{len(cited):>10} / {len(with_summary):<3}"
              f"{summarizer.generations:>6}{len(merge.rounds):>7}{seconds:>7.1f}", flush=True)
    (out_dir / f"{doc.stem}_merge_sweep.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
