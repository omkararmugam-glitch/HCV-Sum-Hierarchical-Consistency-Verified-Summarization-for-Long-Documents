"""Task 2: rule-based dialogue-to-description preprocessing (src/hcv_sum/dialogue.py, FINDINGS 16)."""

import re
from pathlib import Path

import pytest

from hcv_sum.contradiction import nli_text
from hcv_sum.dialogue import dialogue_to_description, rewrite_sentence, speech_act, strip_attribution_frame
from hcv_sum.models import ModelRegistry
from hcv_sum.pipeline import HCVSumPipeline

from conftest import FakeEmbedder, FakeNLI, FakeSummarizer, make_cfg

SAMPLE_03 = Path(__file__).resolve().parents[1] / "data" / "samples" / "03_subtle_contradiction.txt"
OPERATOR_GREETING = ("Operator: Good afternoon, and welcome to the Helix Cloud Systems third quarter 2026 earnings "
                     "conference call.")


def test_speaker_turn_becomes_attributed_prose_and_pleasantries_are_dropped():
    text = "Daniel Okafor, Chief Executive Officer: Thanks, Priya. We plan to add forecasting features next year."
    out, counts = dialogue_to_description(text)
    assert out == ("Daniel Okafor, Chief Executive Officer, said that the company plans to add forecasting "
                   "features next year.")
    assert (counts["turns"], counts["sentences_kept"], counts["sentences_dropped"]) == (1, 1, 1)
    assert counts["dropped_by_category"] == {"thanks": 1}


def test_greeting_is_not_rewritten_as_a_reported_statement():
    """Regression, found in the sample 03 evaluation: the operator's greeting has nine content words, so the
    word-count pleasantry rule kept it, and the rewrite produced the ungrammatical and misleading
    'The conference operator said that good afternoon, and welcome to ...'."""
    out, counts = dialogue_to_description(OPERATOR_GREETING)
    assert "said that good afternoon" not in out.lower()
    assert "good afternoon" not in out.lower() and "said that" not in out
    assert counts["dropped_by_category"] == {"greeting": 1}
    assert counts["dropped_sentences"] == [f"[greeting] Operator: {OPERATOR_GREETING.split(': ', 1)[1]}"]


@pytest.mark.parametrize("sentence, category", [
    ("Good afternoon, and welcome to the Helix Cloud Systems third quarter 2026 earnings conference call.", "greeting"),
    ("Thank you, operator.", "thanks"),
    ("Thanks for taking my question.", "thanks"),
    ("That concludes today's conference call.", "closing"),
    ("At this time all participants are in a listen-only mode.", "procedure"),
    ("We will now begin the question-and-answer session.", "procedure"),
    ("I would now like to turn the call over to Priya Raman, head of investor relations.", "procedure"),
    ("Our first question comes from Aaron Feld at Brookline Securities.", "procedure"),
    ("Joining me today are Daniel Okafor, our chief executive officer, and Mei Lin Zhou.", "introduction"),
    ("Before we begin, please note that today's remarks include forward-looking statements.", "boilerplate"),
    ("A reconciliation of non-GAAP measures is available in our earnings release.", "boilerplate"),
])
def test_non_substantive_speech_acts_are_recognised(sentence, category):
    assert speech_act(sentence) == category


@pytest.mark.parametrize("sentence", [
    "Thanks to strong demand, revenue grew 12 percent.",
    "We welcomed 50 new customers in the quarter.",
    "The biggest milestone this quarter was the completion of our platform migration.",
    "Total revenue for the quarter was 186 million dollars, up 14 percent year over year.",
])
def test_substantive_sentences_are_not_speech_acts(sentence):
    assert speech_act(sentence) is None


def test_questions_are_never_put_in_said_that_form():
    out, _ = dialogue_to_description("Aaron Feld, Brookline Securities: Thanks for taking my question. Can you talk "
                                     "about hiring plans for next year, especially in sales?")
    assert out == ("Aaron Feld, Brookline Securities, asked: Can you talk about hiring plans for next year, "
                   "especially in sales?")


def test_no_malformed_attribution_anywhere_in_sample_03():
    """Every 'said that' in the rewritten transcript must introduce a statement, not a greeting, thanks,
    question or procedural line."""
    out, counts = dialogue_to_description(SAMPLE_03.read_text(encoding="utf-8"))
    bad = re.findall(r"said that (?:good|thank|thanks|welcome|hello|can|could|would you|will you|please|joining|"
                     r"the company's first question|at this time)\b", out, re.I)
    assert bad == []
    assert "Operator:" not in out and "Investor Relations:" not in out
    # the planted contradiction's two sentences must survive the rewrite
    assert "Every enterprise customer is now running on the new Helix Core platform" in out
    assert "roughly sixty enterprise accounts are still being served" in out
    assert counts["turns"] == 9


def test_first_person_is_rewritten_with_verb_agreement():
    assert rewrite_sentence("I think revenue will grow next quarter.", "Mei Lin Zhou", "the company") == \
        "Mei Lin Zhou thinks revenue will grow next quarter."
    assert rewrite_sentence("We're seeing strong demand for our analytics product.", "X", "the company") == \
        "The company is seeing strong demand for the company's analytics product."
    assert rewrite_sentence("We expect costs to fall.", "X", "the Committee") == "The Committee expects costs to fall."


def test_non_dialogue_text_and_headings_are_untouched():
    text = "## Results\n\nRevenue grew 14 percent in the quarter.\n\nGross margin was 71 percent."
    out, counts = dialogue_to_description(text)
    assert out == text and counts["turns"] == 0


def test_procedural_operator_turn_disappears_and_later_sentences_are_not_reattributed():
    out, counts = dialogue_to_description("Operator: I would now like to turn the call over to Priya Raman for the "
                                          "opening remarks.")
    assert out == "" and counts["dropped_by_category"] == {"procedure": 1}
    out, _ = dialogue_to_description("Mei Lin Zhou, CFO: Revenue was 186 million dollars in the quarter. "
                                     "We ended the quarter with 540 million dollars in cash.")
    first, second = out.split(". ", 1)
    assert first.startswith("Mei Lin Zhou, CFO, said that revenue was 186 million")
    assert second.startswith("The company ended the quarter")          # only the first statement is attributed


def test_attribution_frame_is_stripped_for_nli_only_when_enabled():
    s = "Mei Lin Zhou, Chief Financial Officer, said that total revenue grew 14 percent."
    assert strip_attribution_frame(s) == "Total revenue grew 14 percent."
    assert nli_text(s, make_cfg().contradiction) == s                     # default: unchanged
    assert nli_text(s, make_cfg("contradiction.strip_attribution_frames=true").contradiction) == \
        "Total revenue grew 14 percent."


def test_off_by_default_and_on_in_the_pipeline():
    doc = ("Operator: Welcome to the call. I would now like to hand the call over to the chief executive officer.\n\n"
           "Daniel Okafor, Chief Executive Officer: We migrated every enterprise customer to the new platform in June. "
           "We plan to open a Frankfurt region next year.")

    def run(*overrides):
        cfg = make_cfg("summarization.passthrough_tokens=1000", *overrides)
        reg = ModelRegistry(cfg, embedder=FakeEmbedder(), nli=FakeNLI(), summarizer=FakeSummarizer())
        return HCVSumPipeline(cfg, reg).run(doc, "call.txt")

    off = run()
    assert "preprocessing" not in off.stats
    assert any("Daniel Okafor, Chief Executive Officer: We" in s for sec in off.sections for s in sec.sentences)
    on = run("preprocessing.dialogue_to_description=true")
    stats = on.stats["preprocessing"]["dialogue_to_description"]
    assert stats["turns"] == 2 and stats["dropped_by_category"] == {"greeting": 1, "procedure": 1}
    text = " ".join(s for sec in on.sections for s in sec.sentences)
    assert "said that the company migrated every enterprise customer" in text and ": We" not in text
    assert "Welcome to the call" not in text
