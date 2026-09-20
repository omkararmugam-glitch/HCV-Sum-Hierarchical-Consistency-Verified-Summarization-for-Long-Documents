"""Multi-document mode: an unsupervised, retrieval-and-centrality-based approximation.

What this is: several related documents (e.g. successive quarterly updates) summarised together, using
only components the single-document pipeline already has -- sentence embeddings, retrieval, NLI and
graph centrality. It is INSPIRED by the goal of Liu & Lapata (2019, "Hierarchical Transformers for
Multi-Document Summarization"): let information flow across documents and decide which content is
salient. It is NOT their method, which uses a TRAINED cross-document attention mechanism and a TRAINED
paragraph ranker learned from labelled multi-document data. Nothing here is learned or trained, and it
has only been evaluated on our own small synthetic test case (FINDINGS 17), not on their benchmark.

How it works:
1. Each document is segmented on its own; all sections are then pooled into one index, with each section
   title prefixed by its document name.
2. Stage 2 context retrieval and Stage 3 contradiction checking run over the POOLED sections, so a
   section of document A can be anchored against, and compared with, a section of document B.
3. Cross-document flags are checked for supersession: if the two documents carry different explicit
   period/version markers (periods.py) and the two claims are not about the same explicit period, the
   flag is downgraded to "possibly_superseded" -- a later document may be updating an earlier one. If
   both claims name the SAME period, it stays a contradiction.
4. Final summary (in place of a learned salience ranker): the de-duplicated section-summary sentences of
   all documents are ranked by PageRank on their kNN embedding graph (centrality.py), and the top
   ``max_summary_sentences`` are kept, in document order.
5. Stage 5 provenance cites every final sentence to its source sentence in whichever document it came from.
"""

from __future__ import annotations

import dataclasses
import time
from types import SimpleNamespace
from typing import Any, Callable


from .anchored_summarization import summarize_sections
from .centrality import centrality, knn_graph
from .config import Config
from .contradiction import check_section_summaries
from .merging import deduplicate
from .models import CountingEmbedder, CountingNLI, CountingSummarizer, ModelRegistry
from .periods import claim_period, describe, document_period
from .preflight import run_preflight
from .provenance import tag_provenance
from .resources import StageMonitor, process_peak_mb
from .scoring import DocumentIndex
from .segmentation import segment_document
from .types import MergeResult, PipelineResult, Section


def select_salient(sections_sentences: list[list[str]], embedder, limit: int, measure: str, k: int,
                   dedup_similarity: float = 0.85) -> tuple[list[str], list[tuple[str, str, float]], list[str]]:
    """De-duplicate, then keep the ``limit`` most central sentences (document order). Returns (kept, removed, all)."""
    deduped, removed = deduplicate(sections_sentences, embedder, dedup_similarity)
    flat = [(s_i, j, text) for s_i, sec in enumerate(deduped) for j, text in enumerate(sec)]
    if len(flat) <= limit or limit <= 0:
        return [t for _, _, t in flat], removed, [t for _, _, t in flat]
    emb = embedder.encode([t for _, _, t in flat])
    scores = centrality(knn_graph(emb, min(k, len(flat) - 1)), measure)
    top = sorted(sorted(range(len(flat)), key=lambda i: -scores[i])[:limit])
    return [flat[i][2] for i in top], removed, [t for _, _, t in flat]


class MultiDocPipeline:
    def __init__(self, cfg: Config, models: ModelRegistry | None = None):
        self.cfg = cfg
        self.models = models or ModelRegistry(cfg)

    def run(self, documents: list[tuple[str, str]], *, progress: Callable[[str], None] | None = None,
            heartbeat: Callable[[str], None] | None = None, always_heartbeat: bool = False) -> PipelineResult:
        """``progress``: stage messages (verbose). ``heartbeat``: Stage 2 progress lines, sent when the pooled
        sections reach ``scale.large_document_sections`` (or always, with ``always_heartbeat``)."""
        cfg = self.cfg
        say = progress or (lambda _m: None)
        timings: dict[str, float] = {}
        work: dict[str, dict[str, Any]] = {}
        embedder = CountingEmbedder(self.models.embedder)
        nli = CountingNLI(self.models.nli)
        summarizer = CountingSummarizer(self.models.summarizer)

        def timed(name, fn):
            # Same instrumentation as the single-document pipeline: wall-clock, peak memory, work counts.
            before = (nli.pairs, embedder.texts, summarizer.generations)
            with StageMonitor() as mon:
                out = fn()
            timings[name] = round(mon.seconds, 2)
            work[name] = {"nli_pairs": nli.pairs - before[0], "embedded_texts": embedder.texts - before[1],
                          "generations": summarizer.generations - before[2],
                          "peak_rss_mb": round(mon.peak_mb, 1), "rss_start_mb": round(mon.start_mb, 1)}
            say(f"[multi-document] {name} done in {timings[name]}s (peak memory {mon.peak_mb:.0f} MB)")
            return out

        # Stage 1, per document; then pool with global indices and document-prefixed titles.
        sections: list[Section] = []
        section_doc: list[int] = []
        periods: list[tuple | None] = []
        doc_info: list[dict[str, Any]] = []
        preflights = []
        t0 = time.perf_counter()
        for d, (name, text) in enumerate(documents):
            # The period comes from the ORIGINAL text: the dialogue rewrite drops greetings such as
            # "welcome to the ... third quarter 2026 earnings call", which can be a transcript's only marker.
            periods.append(document_period(text))
            preflights.append(run_preflight(text, summarizer.count_tokens, cfg, summarizer.model_max_input))
            for warning in preflights[-1].warnings:
                say(f"[{name}] pre-flight: {warning}")
            if cfg.preprocessing.dialogue_to_description:
                from .dialogue import dialogue_to_description
                text, _ = dialogue_to_description(text, cfg.preprocessing.organisation_reference)
            own = segment_document(text, embedder, summarizer.count_tokens, cfg.segmentation)
            doc_info.append({"name": name, "sections": len(own), "period": describe(periods[-1])})
            offset = sum(len(s.sentences) for s in sections)
            for sec in own:
                title = f"[{name}] {sec.title}" if sec.title else f"[{name}]"
                sections.append(Section(len(sections), title, list(sec.sentences), offset, sec.origin))
                offset += len(sec.sentences)
                section_doc.append(d)
        timings["stage1_segmentation"] = round(time.perf_counter() - t0, 2)
        work["stage1_segmentation"] = {"nli_pairs": 0, "embedded_texts": embedder.texts, "generations": 0,
                                       "peak_rss_mb": 0.0, "rss_start_mb": 0.0}
        if not sections:
            raise ValueError("no sentences found in any input document")
        index = timed("index_source_sentences", lambda: DocumentIndex.build(sections, embedder))

        large = len(sections) >= cfg.scale.large_document_sections
        beat = heartbeat if heartbeat and (large or always_heartbeat) else None
        if beat:
            beat(f"multi-document: {len(documents)} documents, {len(sections)} pooled sections, "
                 f"{len(index.sentences)} source sentences")
        summaries = timed("stage2_anchored_summarization", lambda: summarize_sections(
            sections, index, summarizer, embedder, cfg.summarization, heartbeat=beat,
            heartbeat_every=(cfg.scale.progress_every_sections, cfg.scale.progress_every_seconds)))
        contradiction_cfg = cfg.contradiction
        if cfg.preprocessing.dialogue_to_description:
            contradiction_cfg = dataclasses.replace(contradiction_cfg, strip_attribution_frames=True)
        report = timed("stage3_contradictions", lambda: check_section_summaries(
            summaries, index, embedder, nli, contradiction_cfg, heartbeat=beat,
            heartbeat_every=(cfg.scale.progress_every_pairs, cfg.scale.progress_every_seconds)))

        within = cross = superseded = 0
        for c in report.all_contradictions:
            da, db = section_doc[c.claim_a.section_index], section_doc[c.claim_b.section_index]
            if da == db:
                within += 1
                continue
            cross += 1
            pa_doc, pb_doc = periods[da], periods[db]
            if not (cfg.multidoc.supersede_by_period and pa_doc and pb_doc and pa_doc != pb_doc):
                continue
            pa, pb = claim_period(c.claim_a.text, pa_doc), claim_period(c.claim_b.text, pb_doc)
            if pa == pb:
                c.reason += (f" | cross-document, both claims about {describe(pa)}: a genuine conflict between "
                             f"'{documents[da][0]}' and '{documents[db][0]}', not an update")
                continue
            superseded += 1
            c.resolution = "possibly_superseded"
            c.reason = (f"cross-document: '{documents[da][0]}' covers {describe(pa_doc)}, '{documents[db][0]}' covers "
                        f"{describe(pb_doc)}, and the claims are about different periods ({describe(pa)} vs "
                        f"{describe(pb)}); the later document may update the earlier one rather than contradict it")

        mm = cfg.multidoc
        kept, removed, pool = timed("stage4_salience", lambda: select_salient(
            report.corrected_sentences, embedder, mm.max_summary_sentences, mm.salience_measure, mm.salience_knn,
            cfg.merging.dedup_similarity))
        merge = MergeResult("salience", [s for sec in report.corrected_sentences for s in sec], removed, [],
                            " ".join(kept), kept, [])
        provenance = timed("stage5_provenance",
                           lambda: tag_provenance(merge, index, embedder, nli, cfg.provenance, report))

        from .pipeline import _pair_stats, collect_limits   # local import: pipeline is the single-document module

        cited_docs = sorted({section_doc[r.citation.section_index] for r in provenance if r.citation is not None})
        # Segmentation limits apply per document; report a ceiling as hit if it bound in ANY document.
        aggregate_preflight = SimpleNamespace(
            section_ceiling_binding=any(p.section_ceiling_binding for p in preflights),
            min_segment_sentences=max(p.min_segment_sentences for p in preflights))
        stats: dict[str, Any] = {
            "documents": doc_info,
            "section_documents": section_doc,
            "sections": len(sections),
            "source_sentences": len(index.sentences),
            "summary_claims": sum(len(s.sentences) for s in summaries),
            "final_sentences": len(merge.sentences),
            "salience": {"candidates_after_dedup": len(pool), "kept": len(kept), "measure": mm.salience_measure},
            "flags": {"within_document": within, "cross_document": cross, "possibly_superseded": superseded},
            "documents_cited": [documents[d][0] for d in cited_docs],
            "summarizer_generations": summarizer.generations,
            "nli_pairs_total": nli.pairs,
            "embedded_texts_total": embedder.texts,
            "seconds_total": round(sum(timings.values()), 2),
            "per_stage": {name: {**work[name], "seconds": timings[name]} for name in timings},
            "process_peak_rss_mb": process_peak_mb(),
            "stage3a_pairs": _pair_stats(report.pair_stats),
            "stage3b_pairs": _pair_stats(report.one_sided_stats),
            "large_document": large,
            "preflight_per_document": [{"name": n, **p.to_dict()} for (n, _), p in zip(documents, preflights)],
        }
        stats["limits"] = collect_limits(cfg, aggregate_preflight, sections, summaries, report, None, merge, provenance,
                                         "multi-document mode: salience-ranked sentences, not merged", False)
        return PipelineResult(" + ".join(n for n, _ in documents), sections, summaries, report, None, merge,
                              provenance, timings, cfg.to_dict(), stats)
