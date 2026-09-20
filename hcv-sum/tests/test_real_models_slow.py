"""Real-model behaviour on the sample documents (marked ``slow``: minutes on CPU).

These tests pin what the REAL models actually did in the runs recorded under
``outputs/``, so a model, threshold or logic change that alters that behaviour is
noticed. Where current behaviour is imperfect (known false positives, a planted
contradiction that Stage 2 hides), the test asserts the imperfect behaviour and
says so, instead of pretending the pipeline is better than it is.

Run with:  pytest -m slow
"""

import pytest

from hcv_sum.config import load_config
from hcv_sum.models import CONTRADICTION, ModelRegistry
from hcv_sum.pipeline import HCVSumPipeline

from conftest import SAMPLES

pytestmark = pytest.mark.slow

PLANTED_A = ("Because the shutdown halted all industrial sensor output, backlog orders went unfilled and "
             "industrial sensor revenue fell 18% in the quarter.")
PLANTED_B = ("The decline in industrial sensor revenue was unrelated to the Monterrey shutdown: safety stock held "
             "in our regional distribution centres covered every customer order during the outage, and no backlog "
             "orders went unfilled.")


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def registry(cfg):
    return ModelRegistry(cfg)


@pytest.fixture(scope="module")
def result_01(cfg, registry):
    text = (SAMPLES / "01_planted_contradiction.md").read_text(encoding="utf-8")
    return HCVSumPipeline(cfg, registry).run(text, "01", diagnose_sources=True)


@pytest.fixture(scope="module")
def result_03(cfg, registry):
    text = (SAMPLES / "03_subtle_contradiction.txt").read_text(encoding="utf-8")
    return HCVSumPipeline(cfg, registry).run(text, "03")


# ------------------------------------------------------------------ Stage 1

def test_headings_drive_segmentation(result_01):
    titles = [s.title for s in result_01.sections]
    assert len(result_01.sections) == 5
    assert all(s.origin == "heading" for s in result_01.sections)
    assert "Manufacturing Operations" in titles and "Outlook" in titles


def test_transcript_falls_back_to_embeddings(result_03):
    assert len(result_03.sections) > 1
    assert all(s.origin.startswith("embedding") for s in result_03.sections)
    # The CEO migration remarks and the CFO financials must land in DIFFERENT sections,
    # otherwise the cross-section checks cannot compare them.
    sec_of = {}
    for s in result_03.sections:
        for sent in s.sentences:
            if "Every enterprise customer is now running" in sent:
                sec_of["ceo"] = s.index
            if "roughly sixty enterprise accounts are still being served" in sent:
                sec_of["cfo"] = s.index
    assert sec_of["ceo"] != sec_of["cfo"]


# ------------------------------------------------------------------ Stage 2

def test_section_summaries_are_short_and_nonempty(result_01):
    for ss in result_01.section_summaries:
        section = result_01.sections[ss.section_index]
        assert ss.sentences
        generated = [s for s in ss.sentences if s not in ss.protected]
        assert len(generated) <= len(section.sentences)
        # protected sentences are the section's own source sentences, copied verbatim
        assert all(p in section.sentences for p in ss.protected)


def test_retrieved_context_comes_from_other_sections(result_01):
    for ss in result_01.section_summaries:
        assert all(c.section_index != ss.section_index for c in ss.context)


def test_no_summary_sentence_is_cut_off_mid_sentence(result_01):
    for ss in result_01.section_summaries:
        for sentence in ss.sentences:
            assert sentence.rstrip()[-1] in ".!?\"')"


# ------------------------------------------------------------------ Stage 3

def test_nli_detects_the_planted_pair_directly(cfg, registry):
    """The DETECTOR itself: both directions on the two planted source sentences."""
    probs = registry.nli.predict([(PLANTED_A, PLANTED_B), (PLANTED_B, PLANTED_A)])[:, CONTRADICTION]
    assert min(probs) > 0.9                                       # observed 1.00 / 0.99
    assert probs.mean() >= cfg.contradiction.threshold


def test_planted_pair_is_found_at_source_level(result_01):
    pairs = [{c.claim_a.text, c.claim_b.text} for c in result_01.source_diagnostic.contradictions]
    assert {PLANTED_A, PLANTED_B} in pairs


def test_protection_lets_stage_3a_catch_the_planted_pair(result_01):
    """Replaces a test that pinned Stage 3a MISSING this pair, as that test's docstring asked.

    Before cross-reference protection, DistilBART dropped both planted sentences and Stage 3a had
    nothing to compare: it caught no planted contradiction in any normal run. Both sentences were
    retrieved as each other's context, so protection now re-attaches them verbatim, and Stage 3a
    flags the pair at 0.996 -- the same score as the no-compression ablation.
    """
    protected = [s for ss in result_01.section_summaries for s in ss.protected]
    assert PLANTED_A in protected and PLANTED_B in protected
    hits = [c for c in result_01.contradictions.contradictions
            if {c.claim_a.text, c.claim_b.text} == {PLANTED_A, PLANTED_B}]
    assert hits, "Stage 3a must flag the planted pair once both sides survive"
    assert hits[0].score > 0.9
    # Both claims are fully supported by their own sections: the document itself is inconsistent.
    assert hits[0].resolution == "unresolved"


def test_one_sided_check_catches_the_dropped_side_in_the_transcript(result_03):
    """Stage 3b: the surviving CEO claim vs the CFO source sentence its summary dropped."""
    hits = [c for c in result_03.contradictions.one_sided
            if "Every enterprise customer is now running" in c.claim_a.text
            and "legacy" in c.claim_b.text]
    assert hits, "expected the one-sided check to flag the migration claim"
    c = hits[0]
    assert c.kind == "one_sided"
    # Both sides are supported by their own sections: the DOCUMENT is inconsistent, so
    # the honest outcome is "unresolved" with the claim kept and flagged.
    assert c.resolution == "unresolved"
    assert c.support_a > 0.8 and c.support_b > 0.8


def test_contradiction_free_document_stays_nearly_clean(cfg, registry):
    """Upper bound on FALSE POSITIVES for a document with no contradictions.

    Observed with the comparison filter ON: 1 summary-level and 4 one-sided false flags. Bound raised
    from 6 to 8 one-sided flags when skip_comparative_framing was switched off by default (FINDINGS 10.3,
    11.1): on this sample the filter had removed baseline-comparison pairs, and 8 is what the current
    default produces. The bound documents that precision is imperfect; it is not a target.
    """
    text = (SAMPLES / "02_no_contradiction.md").read_text(encoding="utf-8")
    result = HCVSumPipeline(cfg, registry).run(text, "02")
    assert len(result.contradictions.contradictions) <= 3
    assert len(result.contradictions.one_sided) <= 8
    # Nothing may be deleted from a consistent document (action=flag keeps every claim, marked).
    assert result.contradictions.corrected_sentences == [s.sentences for s in result.section_summaries]


def test_pleasantries_are_not_compared(result_03):
    for c in result_03.contradictions.all_contradictions:
        for claim in (c.claim_a, c.claim_b):
            assert "Thank you" not in claim.text


# ------------------------------------------------------------------ Stages 4 and 5

def test_final_summary_sentences_are_traceable(result_01):
    assert result_01.merge.sentences
    source_text = " ".join(s for sec in result_01.sections for s in sec.sentences)
    for record in result_01.provenance:
        assert record.status in ("supported", "weakly_supported", "unsupported")
        if record.citation:
            assert record.citation.text.split()[0] in source_text
            assert record.citation.section_index < len(result_01.sections)


def test_evidence_panel_reports_stage3_decisions(result_03):
    notes = [n for r in result_03.provenance for n in r.stage3_notes]
    assert any("UNRESOLVED (one-sided)" in n for n in notes), (
        "the surviving migration claim must carry its Stage 3 audit note")


def test_extractive_merge_keeps_every_section(result_01):
    """The default (extractive) merge must not lose sections."""
    cited = {r.citation.section_index for r in result_01.provenance if r.citation}
    assert result_01.merge.mode == "extractive"
    assert len(cited) == len(result_01.sections)


def test_abstractive_merge_loses_section_coverage(registry):
    """KNOWN LIMITATION, and the reason extractive is the default.

    DistilBART's lead bias drops later sections when fusing. Observed: 3 of 5 sections
    cited for sample 01. Recorded so an improvement (or regression) shows up.
    """
    cfg = load_config(overrides=["merging.mode=abstractive"])
    text = (SAMPLES / "01_planted_contradiction.md").read_text(encoding="utf-8")
    result = HCVSumPipeline(cfg, registry).run(text, "01-abstractive")
    cited = {r.citation.section_index for r in result.provenance if r.citation}
    assert len(cited) < len(result.sections)


# ------------------------------------------------------------------ documented NLI limitation

CONTRACT_TERM = ("This Agreement commences on the Effective Date and continues for an initial term of "
                 "twenty-four (24) months.")
CONTRACT_EXPIRY = "Unless renewed, this Agreement expires at the end of the initial term on December 31, 2026."


def test_nli_cannot_detect_a_contradiction_that_needs_date_arithmetic(registry):
    """KNOWN LIMITATION, measured on sample 04 (see FINDINGS.md).

    "Effective Date" is defined elsewhere as January 1, 2026, so a 24-month term ends on
    December 31, 2027 -- contradicting the stated expiry. Finding that requires looking up a
    definition and adding 24 months. The NLI model does semantic opposition, not arithmetic.
    If this test starts failing, the model has improved and FINDINGS.md should be updated.
    """
    probs = registry.nli.predict([(CONTRACT_TERM, CONTRACT_EXPIRY),
                                  (CONTRACT_EXPIRY, CONTRACT_TERM)])[:, CONTRADICTION]
    assert probs.mean() < 0.2, f"observed ~0.02; got {probs.mean():.3f}"


def test_nli_does_detect_the_same_conflict_once_the_date_is_explicit(registry):
    """Control for the test above: remove the arithmetic and the model catches it."""
    explicit = "This Agreement runs for twenty-four months from January 1, 2026, ending on December 31, 2027."
    probs = registry.nli.predict([(explicit, CONTRACT_EXPIRY), (CONTRACT_EXPIRY, explicit)])[:, CONTRADICTION]
    assert probs.mean() > 0.8, f"observed ~0.91; got {probs.mean():.3f}"
