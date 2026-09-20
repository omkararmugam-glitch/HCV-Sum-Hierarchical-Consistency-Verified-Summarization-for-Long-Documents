"""Two fixes aimed at measured failures of the contradiction mechanism.

1. Cross-reference protection (Stage 2): Stage 3a can only find a contradiction when BOTH sides
   survive summarization. Sentences that another section retrieved as context are re-attached.
2. Comparative-framing filter (Stage 3): sentences framed as comparisons ("the baseline is...")
   describe two different things on purpose; NLI reads them as conflicting.
"""

import pytest

from hcv_sum.anchored_summarization import summarize_sections
from hcv_sum.contradiction import check_section_summaries, detect_contradictions, filter_comparative_pairs
from hcv_sum.text_utils import comparative_framing

from conftest import FakeEmbedder, FakeNLI, FakeSummarizer, build_index, make_cfg

CAUSE = "Sensor revenue fell because of the Monterrey plant shutdown."
DENIAL = "Sensor revenue fell not because of the Monterrey plant shutdown."

# Two sections that reference each other's key sentence, plus filler the summarizer will prefer.
SPEC = [
    ("Operations", ["The Monterrey plant closed for three weeks after a transformer fire.",
                    "No employees were injured during the incident.",
                    CAUSE]),
    ("Revenue", ["Automotive revenue grew four percent across every region.",
                 "European demand weakened sharply during the period.",
                 DENIAL]),
]


class LeadOnly(FakeSummarizer):
    """Mimics DistilBART's lead bias: keeps only the first sentence, dropping later claims."""

    def summarize(self, text, *, min_new_tokens, max_new_tokens):
        self.calls.append({"text": text})
        return text.split(". ")[0].rstrip(".") + "."


# ------------------------------------------------------------------ cross-reference protection

def summaries_with(protection: bool):
    embedder = FakeEmbedder()
    index = build_index(SPEC, embedder)
    cfg = make_cfg("summarization.passthrough_tokens=1", "summarization.context_min_similarity=0.1",
                   "summarization.drop_context_leaks=false",
                   f"summarization.protect_cross_referenced={str(protection).lower()}").summarization
    return index, summarize_sections(index.sections, index, LeadOnly(), embedder, cfg), embedder


def test_without_protection_the_cross_referenced_claim_is_lost():
    _, summaries, _ = summaries_with(protection=False)
    kept = [s for ss in summaries for s in ss.sentences]
    assert CAUSE not in kept and DENIAL not in kept, "lead-biased summariser should have dropped both"


def test_protection_restores_the_cross_referenced_claims():
    index, summaries, _ = summaries_with(protection=True)
    kept = [s for ss in summaries for s in ss.sentences]
    assert CAUSE in kept and DENIAL in kept
    assert any(ss.protected for ss in summaries)
    assert any("another section retrieved it as context" in n for ss in summaries for n in ss.notes)


def test_protection_makes_stage_3a_able_to_see_the_contradiction():
    """The whole point: with both sides present, sibling-vs-sibling comparison works."""
    index, summaries, embedder = summaries_with(protection=True)
    cfg = make_cfg("contradiction.pair_min_similarity=0.0").contradiction
    report = check_section_summaries(summaries, index, embedder, FakeNLI(), cfg)
    pairs = [{c.claim_a.text, c.claim_b.text} for c in report.contradictions]
    assert {CAUSE, DENIAL} in pairs, "Stage 3a must now compare the two protected claims"


def test_protection_does_not_duplicate_what_the_summary_already_says():
    embedder = FakeEmbedder()
    index = build_index(SPEC, embedder)
    cfg = make_cfg("summarization.passthrough_tokens=1", "summarization.context_min_similarity=0.1",
                   "summarization.drop_context_leaks=false").summarization

    class Verbatim(FakeSummarizer):
        def summarize(self, text, *, min_new_tokens, max_new_tokens):
            self.calls.append({"text": text})
            return " ".join(index.sections[0].sentences)      # already contains CAUSE

    summaries = summarize_sections(index.sections[:1], index, Verbatim(), embedder, cfg)
    assert summaries[0].protected == [], "already-covered sentences must not be appended again"


def test_protection_can_be_disabled(cfg):
    assert cfg.summarization.protect_cross_referenced is True      # default on
    _, summaries, _ = summaries_with(protection=False)
    assert all(ss.protected == [] for ss in summaries)


# ------------------------------------------------------------------ comparative framing

@pytest.mark.parametrize("sentence,expected", [
    ("The baseline is the same six-layer classifier fine-tuned without retrieval.", "baseline"),
    ("Unlike RACN, the dense model stores full vectors.", "unlike"),
    ("Training takes longer compared with the phase-one system.", "compared with"),
    ("RACN runs faster than the dense-retrieval baseline.", "baseline"),
    ("Revenue grew four percent in the quarter.", None),
    ("The plant closed for three weeks after a fire.", None),
])
def test_comparative_framing_detection(sentence, expected):
    assert comparative_framing(sentence) == expected


def test_comparative_pairs_are_skipped():
    rows = ["A six-layer transformer classifier is fine-tuned on these augmented inputs."]
    cols = ["The baseline is the same six-layer classifier fine-tuned without retrieval.",
            "We evaluate on two datasets of clinical notes."]
    kept, skipped = filter_comparative_pairs([[0, 1]], rows, cols, symmetric=False)
    assert kept == [[1]] and skipped == 1


def test_comparative_filter_reduces_flags_on_baseline_descriptions(embedder):
    claims = [["A six-layer transformer classifier is then fine-tuned on these augmented inputs."],
              ["The baseline is the same six-layer classifier fine-tuned without retrieval."]]
    nli = FakeNLI({(claims[0][0], claims[1][0]): [0.9, 0.05, 0.05],
                   (claims[1][0], claims[0][0]): [0.9, 0.05, 0.05]})
    off = make_cfg("contradiction.skip_comparative_framing=false",
                   "contradiction.pair_min_similarity=0.0").contradiction
    on = make_cfg("contradiction.skip_comparative_framing=true",
                  "contradiction.pair_min_similarity=0.0").contradiction
    assert len(detect_contradictions(claims, embedder, nli, off)[1]) == 1     # flagged without the filter
    stats_on, found_on = detect_contradictions(claims, embedder, nli, on)
    assert found_on == [] and stats_on.checked == 0                          # not even scored with it


def test_comparative_filter_keeps_a_plain_contradiction(embedder, nli, cfg):
    """The filter must not touch pairs with no comparative framing."""
    stats, found = detect_contradictions([[CAUSE], [DENIAL]], embedder, nli, cfg.contradiction)
    assert len(found) == 1 and stats.checked == 1


def test_comparative_filter_is_off_by_default(cfg):
    """Switched off after the real-document evaluation (FINDINGS 10.3): no false flags removed, one
    true contradiction lost. The filter itself stays available and is tested above."""
    assert cfg.contradiction.skip_comparative_framing is False


def test_comparative_skips_are_counted(embedder, nli):
    claims = [["Revenue grew four percent in every region."],
              ["Unlike last year, revenue fell sharply in every region."]]
    cfg = make_cfg("contradiction.skip_comparative_framing=true",
                   "contradiction.pair_min_similarity=0.0").contradiction
    stats, _ = detect_contradictions(claims, embedder, nli, cfg)
    assert stats.skipped_comparative == 1, "the skip must be reported, not silent"
    assert stats.skipped_entity == 0, "a comparative skip must not be reported as an entity skip"
