"""Task 2: current informal handling vs the rule-based dialogue-to-description preprocessing.

Usage:  python scripts/eval_dialogue.py        (writes outputs/dialogue_eval.txt and outputs/dialogue/<sample>_{off,on}.json)

Runs the full default pipeline twice on samples 01 and 03 -- preprocessing.dialogue_to_description off
(current behaviour: speaker labels kept in the text, stripped only before NLI) and on -- and reports:

  preprocessing   speaker turns rewritten, sentences kept / dropped as pleasantries
  segmentation    number of sections and sentences per section (03 has no headings, so its sections come
                  from embedding-based topic segmentation and can move when the text changes)
  Stage 2         summary sentences; invented speaker attributions removed by the Stage 2 guard;
                  summary sentences still containing first-person pronouns (we / our / us / I / my)
  Stage 3         planted contradiction caught (3a / 3b / missed), false flags, pairs compared
  Stage 5         final sentences by support status, DISPUTED count
It also prints the first three section summaries side by side, for a by-eye quality check.

Sample 01 contains no speaker-labelled lines, so the rewrite should leave it unchanged: the script checks
that its final summary is identical in both runs.

Planted contradictions are matched strictly: a flag counts only if its two claims ground to the planted
source sentences. With the rewrite on, those sentences may be reworded (pronouns, "X said that ...");
the match uses distinctive substrings that the rules do not change -- if one is not found, the script
says so instead of guessing.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ["01_planted_contradiction.md", "03_subtle_contradiction.txt"]
FIRST_PERSON = re.compile(r"\b(we|our|us|ours|I|my|me)\b", re.I)


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from hcv_sum.cli import _quiet_third_party
    _quiet_third_party()
    sys.stdout.reconfigure(encoding="utf-8")
    from hcv_sum.config import load_config
    from hcv_sum.models import ModelRegistry
    from hcv_sum.pipeline import HCVSumPipeline
    from hcv_sum.referent import ground_claim
    from hcv_sum.report import save_json
    from hcv_sum.scoring import DocumentIndex

    truth = json.loads((ROOT / "data" / "samples" / "ground_truth.json").read_text(encoding="utf-8"))
    base = load_config()
    reg = ModelRegistry(base)                      # models loaded once, shared by both configurations
    lines: list[str] = []

    def say(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    for name in SAMPLES:
        text = (ROOT / "data" / "samples" / name).read_text(encoding="utf-8-sig")
        stem = Path(name).stem
        runs = {}
        for mode in ("off", "on"):
            cfg = load_config(overrides=[f"preprocessing.dialogue_to_description={'true' if mode == 'on' else 'false'}"])
            result = HCVSumPipeline(cfg, reg).run(text, name)
            save_json(result, ROOT / "outputs" / "dialogue" / f"{stem}_{mode}.json")
            runs[mode] = result

        say(f"\n=== {name} ===")
        say(f"  {'':<44}{'OFF (current)':>16}{'ON (rewrite)':>16}")

        def row(label, fn):
            say(f"  {label:<44}{str(fn(runs['off'])):>16}{str(fn(runs['on'])):>16}")

        row("preprocessing: turns / kept / dropped", lambda r: "-" if "preprocessing" not in r.stats else
            "{turns}/{sentences_kept}/{sentences_dropped}".format(**r.stats["preprocessing"]["dialogue_to_description"]))
        row("sections", lambda r: len(r.sections))
        row("sentences per section", lambda r: ",".join(str(len(s.sentences)) for s in r.sections))
        row("summary sentences", lambda r: sum(len(s.sentences) for s in r.section_summaries))
        row("invented attributions removed (Stage 2)",
            lambda r: sum(n.startswith("removed speaker attribution") for s in r.section_summaries for n in s.notes))
        row("summary sentences with first person",
            lambda r: sum(bool(FIRST_PERSON.search(t)) for s in r.section_summaries for t in s.sentences))
        row("pairs compared (3a + 3b)", lambda r: r.contradictions.pairs_checked + r.contradictions.one_sided_checked)

        def planted_status(r):
            planted = truth.get(stem, {}).get("contradictions", [])
            if not planted:
                return "none planted"
            index = DocumentIndex.build(r.sections, reg.embedder)
            src = index.sentences
            ka, kb = planted[0]
            sa = {k for k, t in enumerate(src) if ka in t}
            sb = {k for k, t in enumerate(src) if kb in t}
            if not sa or not sb:
                return "SUBSTRING NOT FOUND"
            for c in r.contradictions.contradictions + r.contradictions.one_sided:
                gi = ground_claim(c.claim_a, index, reg.embedder).sentence
                gj = ground_claim(c.claim_b, index, reg.embedder, is_source=c.kind == "one_sided").sentence
                i, j = src.index(gi), src.index(gj)
                if (i in sa and j in sb) or (j in sa and i in sb):
                    return f"caught {c.kind[:3]} {c.score:.2f}"
            return "missed"

        statuses = {m: planted_status(runs[m]) for m in runs}
        say(f"  {'planted contradiction':<44}{statuses['off']:>16}{statuses['on']:>16}")
        row("flags (3a + 3b)", lambda r: len(r.contradictions.all_contradictions))
        row("final sentences", lambda r: len(r.merge.sentences))
        row("  supported / weak / unsupported", lambda r: "/".join(
            str(sum(p.status == s for p in r.provenance)) for s in ("supported", "weakly_supported", "unsupported")))
        row("  DISPUTED", lambda r: sum(any(f.startswith("DISPUTED") for f in p.flags) for p in r.provenance))
        if runs["off"].merge.summary == runs["on"].merge.summary:
            say("  final summary: IDENTICAL in both runs")
        else:
            say("  final summary: differs between runs")

        # Verification of the rewrite itself (ON run): every dropped sentence with its category, every
        # attributed sentence, and an explicit check for the bug found in the first evaluation.
        pre = runs["on"].stats.get("preprocessing", {}).get("dialogue_to_description", {})
        if pre.get("turns"):
            say(f"\n  dropped by category: {pre.get('dropped_by_category')}")
            for line in pre.get("dropped_sentences", []):
                say(f"    DROPPED {line}")
            on_text = " ".join(s for sec in runs["on"].sections for s in sec.sentences)
            attributed = re.findall(r"[^.?!]*(?:said that|asked:)[^.?!]*[.?!]", on_text)
            say(f"  attributed sentences in the rewritten text: {len(attributed)}")
            for sentence in attributed:
                say(f"    {sentence.strip()}")
            bug = "said that good afternoon" in on_text.lower()
            say(f"  'said that good afternoon' present: {'YES -- BUG NOT FIXED' if bug else 'no'}")
        say("\n  first three section summaries (OFF | ON):")
        for k in range(3):
            for mode in ("off", "on"):
                sums = runs[mode].section_summaries
                txt = " ".join(sums[k].sentences) if k < len(sums) else "(no such section)"
                say(f"    [{k}] {mode.upper():<3} {txt[:300]}")
    (ROOT / "outputs" / "dialogue_eval.txt").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
