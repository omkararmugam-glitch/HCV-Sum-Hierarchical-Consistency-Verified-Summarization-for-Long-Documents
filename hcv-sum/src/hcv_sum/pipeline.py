"""Pipeline orchestrator: runs Stages 1-5, keeps every intermediate result, and counts the work.

Instrumentation exists because the two dominant CPU costs (summarizer generations and NLI
sentence-pair scorings) grow very differently with document size: generations are linear in the
number of sections, NLI calls are quadratic in the number of claims unless Stage 3's candidate
limits bite. ``PipelineResult.stats`` reports both, per stage, plus wall-clock and peak memory per
stage, a pre-flight estimate, and every cap or limit the run hit (``stats["limits"]``), so a long run
can be predicted from a short one and a constrained result is never mistaken for a complete one.

Size-based behaviour (config ``scale``): a document with at least ``large_document_sections``
sections gets Stage 2 checkpointing (``checkpoint: auto``), progress lines, and
``large_document_merge_mode`` for Stage 4. Smaller documents run exactly as before.
"""

from __future__ import annotations

import dataclasses
from dataclasses import asdict
from typing import Any, Callable

from .anchored_summarization import summarize_sections
from .checkpoint import Stage2Checkpoint, checkpoint_path
from .config import Config
from .contradiction import check_section_summaries, diagnose_source_sections
from .merging import merge_summaries
from .models import CountingEmbedder, CountingNLI, CountingSummarizer, ModelRegistry
from .preflight import run_preflight
from .provenance import tag_provenance
from .resources import StageMonitor, process_peak_mb
from .scoring import DocumentIndex
from .segmentation import segment_document
from .types import PipelineResult


class HCVSumPipeline:
    def __init__(self, cfg: Config, models: ModelRegistry | None = None):
        self.cfg = cfg
        self.models = models or ModelRegistry(cfg)

    def run(self, text: str, document_name: str = "document", *, diagnose_sources: bool = False,
            progress: Callable[[str], None] | None = None,
            heartbeat: Callable[[str], None] | None = None, always_heartbeat: bool = False,
            checkpoint_dir: str | None = None, checkpoint_mode: str | None = None) -> PipelineResult:
        """``progress``: stage start/end messages (verbose). ``heartbeat``: Stage 2 progress lines, sent
        for large documents, or always if ``always_heartbeat``. ``checkpoint_dir``/``checkpoint_mode``
        override config ``scale.checkpoint_dir`` / ``scale.checkpoint`` (auto | always | never)."""
        cfg = self.cfg
        say = progress or (lambda _msg: None)
        preprocessing = None
        if cfg.preprocessing.dialogue_to_description:
            from .dialogue import dialogue_to_description
            text, preprocessing = dialogue_to_description(text, cfg.preprocessing.organisation_reference)
            # The rewrite adds "Name, Title, said that" frames; strip them before NLI like speaker labels.
            cfg = dataclasses.replace(cfg, contradiction=dataclasses.replace(cfg.contradiction,
                                                                             strip_attribution_frames=True))
            say(f"[{document_name}] dialogue-to-description: {preprocessing}")
        timings: dict[str, float] = {}
        work: dict[str, dict[str, Any]] = {}

        embedder = CountingEmbedder(self.models.embedder)
        nli = CountingNLI(self.models.nli)
        summarizer = CountingSummarizer(self.models.summarizer)
        merge_summarizer = (summarizer if self.models.merge_summarizer is self.models.summarizer
                            else CountingSummarizer(self.models.merge_summarizer))

        def timed(name: str, fn, note: str = ""):
            say(f"[{document_name}] {name} ...{(' ' + note) if note else ''}")
            before = (nli.pairs, embedder.texts, summarizer.generations)
            with StageMonitor() as mon:
                out = fn()
            timings[name] = round(mon.seconds, 2)
            work[name] = {"nli_pairs": nli.pairs - before[0], "embedded_texts": embedder.texts - before[1],
                          "generations": summarizer.generations - before[2],
                          "peak_rss_mb": round(mon.peak_mb, 1), "rss_start_mb": round(mon.start_mb, 1)}
            say(f"[{document_name}] {name} done in {timings[name]}s "
                f"(nli_pairs={work[name]['nli_pairs']}, generations={work[name]['generations']}, "
                f"peak memory {mon.peak_mb:.0f} MB)")
            return out

        preflight = timed("preflight", lambda: run_preflight(text, summarizer.count_tokens, cfg,
                                                             summarizer.model_max_input))
        for w in preflight.warnings:
            say(f"[{document_name}] pre-flight: {w}")
        if heartbeat and (preflight.large_document or always_heartbeat) and progress is None:
            from .preflight import render_preflight
            heartbeat(render_preflight(preflight, document_name))

        sections = timed("stage1_segmentation", lambda: segment_document(
            text, embedder, summarizer.count_tokens, cfg.segmentation))
        if not sections:
            raise ValueError(f"{document_name}: no sentences found in input")
        index = timed("index_source_sentences", lambda: DocumentIndex.build(sections, embedder))
        say(f"[{document_name}] {len(sections)} sections, {len(index.sentences)} source sentences")

        large = len(sections) >= cfg.scale.large_document_sections
        mode = checkpoint_mode or cfg.scale.checkpoint
        checkpoint = None
        if mode == "always" or (mode == "auto" and large):
            settings = {"summarizer": cfg.models.summarizer, "segmentation": asdict(cfg.segmentation),
                        "summarization": asdict(cfg.summarization)}
            checkpoint = Stage2Checkpoint(checkpoint_path(checkpoint_dir or cfg.scale.checkpoint_dir, text, settings))
            say(f"[{document_name}] Stage 2 checkpoint: {checkpoint.path} ({checkpoint.loaded} generations on disk)")
        beat = heartbeat if heartbeat and (large or always_heartbeat) else None

        summaries = timed("stage2_anchored_summarization", lambda: summarize_sections(
            sections, index, summarizer, embedder, cfg.summarization, checkpoint=checkpoint, heartbeat=beat,
            heartbeat_every=(cfg.scale.progress_every_sections, cfg.scale.progress_every_seconds)),
            f"{len(sections)} sections")

        report = timed("stage3_contradictions", lambda: check_section_summaries(
            summaries, index, embedder, nli, cfg.contradiction, heartbeat=beat,
            heartbeat_every=(cfg.scale.progress_every_pairs, cfg.scale.progress_every_seconds)))
        diagnostic = None
        if diagnose_sources:
            diagnostic = timed("stage3_source_diagnostic", lambda: diagnose_source_sections(
                index, embedder, nli, cfg.contradiction))

        merge_mode = cfg.scale.large_document_merge_mode if large else cfg.merging.mode
        merge_reason = (f"{len(sections)} sections >= scale.large_document_sections="
                        f"{cfg.scale.large_document_sections}: scale.large_document_merge_mode" if large
                        else "merging.mode (document below the large-document threshold)")
        rejected = [c.rejected.text for c in report.all_contradictions if c.rejected is not None]
        merge = timed("stage4_merge", lambda: merge_summaries(
            report.corrected_sentences, rejected, merge_summarizer, embedder, nli, cfg.merging,
            contradiction_cfg=cfg.contradiction, mode=merge_mode, index=index))

        provenance = timed("stage5_provenance", lambda: tag_provenance(
            merge, index, embedder, nli, cfg.provenance, report))

        stats: dict[str, Any] = {
            "sections": len(sections),
            "source_sentences": len(index.sentences),
            "summary_claims": sum(len(s.sentences) for s in summaries),
            "final_sentences": len(merge.sentences),
            "summarizer_generations": summarizer.generations,
            "nli_pairs_total": nli.pairs,
            "embedded_texts_total": embedder.texts,
            "seconds_total": round(sum(timings.values()), 2),
            "per_stage": {name: {**work[name], "seconds": timings[name]} for name in timings},
            "process_peak_rss_mb": process_peak_mb(),
            "merge_rounds": [{"round": r.round_number, "groups": len(r.inputs), "outputs": len(r.outputs),
                              "introduced_contradictions": len(r.introduced)} for r in merge.rounds],
            "stage3a_pairs": _pair_stats(report.pair_stats),
            "stage3b_pairs": _pair_stats(report.one_sided_stats),
            "large_document": large,
            "merge_mode": {"mode": merge.mode, "reason": merge_reason},
            "checkpoint": ({"path": str(checkpoint.path), "loaded": checkpoint.loaded, "reused": checkpoint.hits,
                            "written": checkpoint.writes} if checkpoint else None),
            "preflight": preflight.to_dict(),
        }
        if diagnostic is not None:
            stats["stage3_diagnostic_pairs"] = _pair_stats(diagnostic.pair_stats)
        if preprocessing is not None:
            stats["preprocessing"] = {"dialogue_to_description": preprocessing}
        stats["limits"] = collect_limits(cfg, preflight, sections, summaries, report, diagnostic, merge,
                                         provenance, merge_reason, large)

        return PipelineResult(document_name, sections, summaries, report, diagnostic, merge, provenance,
                              timings, cfg.to_dict(), stats)


def _pair_stats(stats) -> dict[str, int]:
    return {
        "possible": stats.total,
        "compared": stats.checked,
        "nli_calls": stats.nli_calls,
        "skipped_low_similarity": stats.skipped_similarity,
        "skipped_candidate_top_k": stats.skipped_top_k,
        "skipped_max_pairs_budget": stats.skipped_budget,
        "skipped_no_shared_entity": stats.skipped_entity,
        "skipped_comparative_framing": stats.skipped_comparative,
        "skipped_no_shared_content_word": stats.skipped_topic,
    }


def collect_limits(cfg: Config, preflight, sections, summaries, report, diagnostic, merge, provenance,
                   merge_reason: str, large: bool) -> list[dict[str, Any]]:
    """Every cap, ceiling or size-based switch, with whether THIS run hit it (report.render_limits)."""
    def item(stage: str, limit: str, hit: bool, detail: str) -> dict[str, Any]:
        return {"stage": stage, "limit": limit, "hit": bool(hit), "detail": detail}

    split = sum(1 for s in sections if s.origin.endswith("+split"))
    truncated = sum(1 for s in summaries if s.input_truncated)
    out = [
        item("stage1", "segmentation.max_sections", preflight.section_ceiling_binding,
             f"{len(sections)} sections (ceiling {cfg.segmentation.max_sections}); minimum segment "
             f"{preflight.min_segment_sentences} sentences"),
        item("stage1", "segmentation.max_section_tokens", split > 0,
             f"{split} sections came from splitting an over-budget section at a semantic valley"),
        item("stage2", "summarization.max_input_tokens", truncated > 0,
             f"{truncated} of {len(summaries)} sections had their INPUT TRUNCATED (text beyond the budget unseen)"),
    ]
    for label, st in (("stage3a", report.pair_stats), ("stage3b", report.one_sided_stats)):
        if st is None:
            continue
        out.append(item(label, "contradiction.candidate_top_k", st.skipped_top_k > 0,
                        f"{st.skipped_top_k} candidate pairs beyond each claim's top {cfg.contradiction.candidate_top_k} "
                        f"not compared"))
        out.append(item(label, "contradiction.max_pairs", st.skipped_budget > 0,
                        f"{st.skipped_budget} candidate pairs over the {cfg.contradiction.max_pairs}-pair budget "
                        f"not compared"))
    out.append(item("stage3", "resolution budget", report.unscored > 0,
                    f"{report.unscored} flagged pairs left unscored (budget {report.resolution_budget} = "
                    f"max(max_resolutions, max_resolutions_per_section x sections)); still flagged"))
    if diagnostic is not None:
        st = diagnostic.pair_stats
        out.append(item("stage3-diagnostic", "contradiction.diagnostic_max_pairs", st.skipped_budget > 0,
                        f"{st.skipped_budget} source pairs over the {cfg.contradiction.diagnostic_max_pairs}-pair "
                        f"ceiling not compared: the diagnostic is incomplete"))
    out.append(item("stage4", "scale.large_document_merge_mode", large,
                    f"merge mode '{merge.mode}' chosen by {merge_reason}"))
    if merge.mode == "abstractive":
        unfinished = bool(merge.rounds) and len(merge.rounds[-1].outputs) > 1
        out.append(item("stage4", "merging.max_rounds", unfinished,
                        f"{len(merge.rounds)} merge rounds; "
                        + (f"{len(merge.rounds[-1].outputs)} blocks were still unmerged when the round limit hit"
                           if unfinished else "merged to a single block")))
        with_summary = {s.section_index for s in summaries if s.sentences}
        cited = {r.citation.section_index for r in provenance if r.citation is not None}
        out.append(item("stage4", "abstractive coverage", len(cited) < len(with_summary),
                        f"final summary cites {len(cited & with_summary)} of {len(with_summary)} sections "
                        f"(abstractive merging drops sections; see SCALING.md)"))
    return out
