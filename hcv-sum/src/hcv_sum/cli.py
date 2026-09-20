"""Command-line entry point: ``hcv-sum document.txt``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


# TorchScript deprecation, emitted while importing transformers' DeBERTa-v2 modelling code.
#
# Where it comes from: transformers/models/deberta_v2/modeling_deberta_v2.py applies
# @torch.jit.script to six module-level helpers, so the warning fires at IMPORT time as soon as
# our NLI model (MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli) is loaded. It is not raised by
# our code, and not by sentence-transformers (which uses no TorchScript at all).
#
# Why it cannot be fixed by upgrading: transformers 5.17.0 is the latest release and its `main`
# branch still has all six decorators, so no available version avoids it.
#
# Why it is safe to suppress: torch 2.14 promoted these notices from a hidden DeprecationWarning
# to a user-visible FutureWarning. TorchScript still works; this is an advance notice about a
# future removal, addressed to the authors of transformers, not to us. We never call
# torch.jit.script ourselves and we ship no TorchScript artefacts.
#
# What would change this: when transformers stops using torch.jit.script in DeBERTa (no upstream
# issue exists yet as of 2026-09), or when TorchScript is actually removed from torch (announced,
# not scheduled) -- at which point loading DeBERTa would FAIL rather than warn, and this filter
# must be removed so the failure is visible. On Python 3.14+ torch emits a different, stronger
# message ("not supported in Python 3.14+ and may break"); that variant is matched separately
# below so that it, too, stays quiet only while it is merely a warning.
_TORCHSCRIPT_WARNINGS = (
    r"`torch\.jit\.script` is deprecated",
    r"`torch\.jit\.script` is not supported in Python 3\.14\+",
)


def silence_torchscript_deprecation() -> None:
    """Suppress ONLY torch's TorchScript deprecation notice, only when raised by torch.jit._script."""
    import warnings

    for message in _TORCHSCRIPT_WARNINGS:
        warnings.filterwarnings("ignore", message=message, category=FutureWarning, module=r"torch\.jit\._script")


def _quiet_third_party() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import warnings

    silence_torchscript_deprecation()
    warnings.filterwarnings("ignore", category=UserWarning, module="huggingface_hub")
    # These libraries log at WARNING on every run (e.g. huggingface_hub's "unauthenticated
    # requests to the HF Hub" notice, which is about rate limits, not about this document).
    # Without this the default run would print them alongside the summary. --verbose skips
    # this function entirely, so nothing is hidden when you ask to see everything.
    #
    # Each library is silenced through its OWN logging API, and only after importing it:
    # both configure their root logger on import, so a plain logging.getLogger(...).setLevel()
    # made before the import is overwritten by the library a moment later.
    try:
        from huggingface_hub.utils import logging as hub_logging

        hub_logging.set_verbosity_error()
    except ImportError:
        pass
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except ImportError:
        pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hcv-sum", description="Hierarchical Consistency-Verified Summarization",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=OUTPUT_MODES)
    p.add_argument("document", type=Path, nargs="+",
                   help="input file: .txt or .md (UTF-8), .pdf (with a text layer), .json (see README for the "
                        "accepted shapes) or .docx; give several related documents, in any mix of these formats, "
                        "to summarise them together (multi-document mode, see README)")
    p.add_argument("--config", help="YAML config (default: config/default.yaml)")
    p.add_argument("--set", action="append", default=[], dest="overrides", metavar="SECTION.KEY=VALUE",
                   help="override a config value, e.g. --set contradiction.threshold=0.4 (repeatable)")
    p.add_argument("--evidence", action="store_true",
                   help="print the final summary plus the evidence panel: per-sentence citations, support "
                        "status, and the Stage 3 contradiction decisions")
    p.add_argument("--diagnose-sources", action="store_true",
                   help="also run the Stage 3 detector over raw source sentences (slower)")
    p.add_argument("--json", type=Path, help="write the full run (all stages) to this JSON file")
    p.add_argument("--preflight", action="store_true",
                   help="only estimate sections, section size and runtime (loads the tokenizer, no models), then exit")
    p.add_argument("--checkpoint", choices=["auto", "always", "never"],
                   help="Stage 2 checkpoint/resume (default from config scale.checkpoint: auto = large documents)")
    p.add_argument("--checkpoint-dir", help="where Stage 2 checkpoints are kept (default: scale.checkpoint_dir)")
    detail = p.add_mutually_exclusive_group()
    detail.add_argument("--brief", action="store_true",
                        help="print a few short lines saying what the run did, with counts, before the final "
                             "summary; no sentences, scores, citations or timings. Cannot be combined with --verbose")
    detail.add_argument("-v", "--verbose", action="store_true",
                        help="print every stage (segmentation, section summaries, contradictions, merge) plus the "
                             "evidence panel, progress on stderr, model-loading logs and library warnings")
    return p


OUTPUT_MODES = """\
output modes, least to most detail (--json FILE writes the complete run in every mode):
  (no flag)   the final summary text only; safe to pipe
                GraphFault is a dynamic graph neural network framework for early fault detection ...
  --brief     a few short lines saying what the run did, with counts, then the final summary
                Compared 76 claim pairs across sections and found 7 unresolved contradictions; kept ...
  --evidence  the final summary, then every final sentence with its status, score and source citation
                S3 [SUPPORTED]  entailment=0.982  best similarity=0.911
  --verbose   every stage in full plus the evidence panel; progress and library logs on stderr
                [doc.md] stage2_anchored_summarization done in 81.26s (nli_pairs=0, generations=20, ...)
--brief can be added to --evidence (stage lines first); --brief and --verbose are mutually exclusive."""


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.verbose:
        _quiet_third_party()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    from .config import ConfigError, load_config
    from .pipeline import HCVSumPipeline
    from .report import render_brief, render_limits, render_result, render_stage_lines, render_summary, save_json

    try:
        cfg = load_config(args.config, args.overrides)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    # Every input goes through ingestion (format detection + conversion to plain text) before Stage 1; the
    # stages only ever see plain text. .txt/.md are read exactly as before (UTF-8, BOM dropped). Formats may be
    # mixed in one multi-document run. Anything unsupported stops here, before any model is loaded.
    from .ingestion import IngestionError, ingest

    texts = []
    for path in args.document:
        if not path.is_file():
            print(f"error: file not found: {path}", file=sys.stderr)
            return 2
        try:
            ingested = ingest(path)
        except IngestionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        if args.verbose:
            print(f"[{path.name}] {ingested.summary}", file=sys.stderr, flush=True)
        texts.append(ingested.text)
    multi = len(texts) > 1
    if multi and args.preflight:
        print("error: --preflight takes one document at a time", file=sys.stderr)
        return 2
    text = texts[0]
    args.document = args.document[0] if not multi else args.document
    if args.preflight:
        from .models import TokenCounter
        from .preflight import render_preflight, run_preflight
        counter = TokenCounter(cfg.models.summarizer)
        print(render_preflight(run_preflight(text, counter.count_tokens, cfg, counter.model_max_input),
                               args.document.name))
        return 0

    # Stage-by-stage progress belongs to --verbose: with no flags stdout carries the summary and nothing
    # else. Large documents (config scale) also get Stage 2 progress lines on STDERR, so a long run is
    # visibly alive without polluting the summary on stdout.
    def to_stderr(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    progress = to_stderr if args.verbose else None
    if multi:
        from .multidoc import MultiDocPipeline
        ignored = [flag for flag, on in (("--diagnose-sources", args.diagnose_sources), ("--checkpoint", args.checkpoint),
                                         ("--checkpoint-dir", args.checkpoint_dir)) if on]
        if ignored:
            one = len(ignored) == 1
            print(f"note: {', '.join(ignored)} {'is a single-document option' if one else 'are single-document options'}"
                  f" and {'is' if one else 'are'} ignored with several documents", file=sys.stderr)
        result = MultiDocPipeline(cfg).run([(p.name, t) for p, t in zip(args.document, texts)], progress=progress,
                                           heartbeat=to_stderr, always_heartbeat=args.verbose)
    else:
        result = HCVSumPipeline(cfg).run(text, args.document.name, diagnose_sources=args.diagnose_sources,
                                         progress=progress, heartbeat=to_stderr, always_heartbeat=args.verbose,
                                         checkpoint_dir=args.checkpoint_dir, checkpoint_mode=args.checkpoint)

    # Output modes, least to most detail (see OUTPUT_MODES):
    #   (no flags)      just the final summary text
    #   --brief         one line per stage + final summary
    #   --evidence      final summary + evidence panel (preceded by the stage lines with --brief)
    #   --verbose       every stage + final summary + evidence panel
    if args.evidence:
        if args.brief:
            print("\n".join(render_stage_lines(result)) + "\n")
        print(render_result(result, show_stages=False))
    elif args.verbose:
        print(render_result(result, show_stages=True))
    else:
        print(render_brief(result) if args.brief else render_summary(result))
        # Only when something was capped, and only on stderr: stdout stays the summary alone.
        if any(x["hit"] for x in result.stats.get("limits", [])):
            print("\n" + render_limits(result, only_hits=True), file=sys.stderr)
    # --evidence and --verbose get the RUN LIMITS block from render_result(); printing it here too
    # duplicated it in --evidence output.
    if args.json:
        save_json(result, args.json)
        print(f"\nfull run written to {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
