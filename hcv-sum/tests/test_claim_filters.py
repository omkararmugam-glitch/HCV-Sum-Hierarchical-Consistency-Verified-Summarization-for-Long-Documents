"""Stage 3 filters added after the real-document evaluation (FINDINGS.md sections 10 and 11):
the verb-or-number non-claim filter and the shared-content-word (topic overlap) gate."""

import pytest

from hcv_sum.contradiction import detect_contradictions, filter_by_shared_content_word, is_claim
from hcv_sum.text_utils import content_tokens, has_verb_or_number

from conftest import make_cfg


@pytest.mark.parametrize("sentence", [
    "Jeanna Smialek, New York Times.",
    "Elizabeth Schulze with ABC News.",
    "Michael McKee with Bloomberg TV and Radio.",
])
def test_reporter_introductions_are_not_claims(sentence, cfg):
    """65 of the press-conference transcript's 467 flags involved lines like these (FINDINGS 10.5)."""
    assert not has_verb_or_number(sentence)
    assert not is_claim(sentence, cfg.contradiction)


@pytest.mark.parametrize("sentence", [
    "The Committee decided to lower the target range.",   # verb
    "Revenue up 12% year over year.",                     # no verb, but a figure
    "Roughly sixty enterprise accounts on the old platform.",  # spelled-out number
    "There isn’t a bright line.",                    # curly apostrophe contraction
    "Don’t look for anything else.",
])
def test_checkable_sentences_are_claims(sentence, cfg):
    assert has_verb_or_number(sentence)
    assert is_claim(sentence, cfg.contradiction)


@pytest.mark.parametrize("sentence", [
    # Real claims the POS tagger mis-tagged as verbless (FINDINGS 11.2): must stay claims.
    "This paper studies the impact of the reference determinacy assumption in the NLI dataset creation process.",
    "IN NO EVENT SHALL, XIMAGE BE LIABLE TO MORPHO FOR ANY INCIDENTAL, CONSEQUENTIAL, SPECIAL OR INDIRECT DAMAGES.",
])
def test_long_or_all_caps_sentences_are_not_dropped_as_fragments(sentence, cfg):
    assert is_claim(sentence, cfg.contradiction)


def test_verb_filter_can_be_disabled():
    cfg = make_cfg("contradiction.require_verb_or_number=false").contradiction
    assert is_claim("Jeanna Smialek, New York Times.", cfg)


def test_content_tokens_keep_common_nouns_and_depluralise():
    assert {"labor", "market"} <= content_tokens("The labor market has cooled.")
    assert "rate" in content_tokens("Mortgage rates rose.") and "rate" in content_tokens("The rate fell.")
    assert "we" not in content_tokens("We construct the benchmark.")
    assert "4.2" in content_tokens("Net income was 4.2 billion.")


def test_content_tokens_ignore_speaker_labels():
    assert "powell" not in content_tokens("Chair Powell: Inflation has eased.")


def test_topic_gate_drops_same_subject_different_activity():
    """The dominant false flag on the research paper: shared subject 'we', nothing else in common."""
    texts = ["We conduct an experiment with the ChaosNLI dataset.", "We construct the RefNLI benchmark."]
    kept, skipped = filter_by_shared_content_word([[1], [0]], texts, texts, symmetric=True)
    assert kept == [[], []] and skipped == 1


def test_topic_gate_keeps_common_noun_contradiction():
    """Where the entity gate failed (FINDINGS 10.2): a contradiction about an ordinary noun phrase."""
    a, b = ["The labor market remains very tight."], ["The labor market has loosened considerably."]
    kept, skipped = filter_by_shared_content_word([[0]], a, b, symmetric=False)
    assert kept == [[0]] and skipped == 0


def test_topic_gate_skips_are_counted_separately(embedder, nli):
    claims = [["We conduct an experiment with the ChaosNLI dataset."],
              ["We construct the RefNLI benchmark from retrieved Wikipedia premises."]]
    cfg = make_cfg("contradiction.require_shared_content_word=true",
                   "contradiction.pair_min_similarity=0.0").contradiction
    stats, found = detect_contradictions(claims, embedder, nli, cfg)
    assert stats.skipped_topic == 1 and stats.checked == 0 and found == []
    assert stats.skipped_entity == 0 and stats.skipped_comparative == 0
