import pytest

from hcv_sum.contradiction import (check_section_summaries, detect_contradictions, diagnose_source_sections,
                                   is_claim, resolve_contradictions)
from hcv_sum.text_utils import strip_speaker_label
from hcv_sum.types import SectionSummary

from conftest import FakeNLI, build_index, make_cfg

CAUSE = "Sensor revenue fell because of the plant shutdown."
DENIAL = "Sensor revenue fell not because of the plant shutdown."


def summaries_from(claims):
    return [SectionSummary(i, "", " ".join(c), list(c), [], True, False) for i, c in enumerate(claims)]


def test_detects_cross_section_contradiction(embedder, nli, cfg):
    other = "Repair costs rose sharply in April."
    stats, found = detect_contradictions([[CAUSE, other], [DENIAL]], embedder, nli, cfg.contradiction)
    assert stats.total == 2
    assert len(found) == 1
    assert {found[0].claim_a.text, found[0].claim_b.text} == {CAUSE, DENIAL}
    assert found[0].score >= cfg.contradiction.threshold


def test_same_section_pairs_are_not_compared(embedder, nli, cfg):
    stats, found = detect_contradictions([[CAUSE, DENIAL]], embedder, nli, cfg.contradiction)
    assert (stats.total, stats.checked, found) == (0, 0, [])


def test_no_false_positive_on_consistent_claims(embedder, nli, cfg):
    claims = [["Sensor revenue fell eighteen percent."], ["Sensor revenue fell eighteen percent in Europe."],
              ["Resin lead times improved."]]
    _, found = detect_contradictions(claims, embedder, nli, cfg.contradiction)
    assert found == []


def test_similarity_prefilter_skips_unrelated_pairs(embedder, nli):
    cfg = make_cfg("contradiction.pair_min_similarity=0.5").contradiction
    stats, _ = detect_contradictions([[CAUSE], [DENIAL, "Resin lead times improved."]], embedder, nli, cfg)
    assert stats.total == 2 and stats.checked == 1
    assert stats.skipped_similarity == 1


@pytest.mark.parametrize("aggregation,expected", [("max", 0.9), ("min", 0.1), ("mean", 0.5)])
def test_direction_aggregation(embedder, aggregation, expected):
    a, b = "Alpha division revenue rose sharply.", "Beta division revenue fell sharply."
    nli = FakeNLI({(a, b): [0.1, 0.1, 0.8], (b, a): [0.9, 0.05, 0.05]})
    cfg = make_cfg("contradiction.pair_min_similarity=0", "contradiction.threshold=0.05",
                   f"contradiction.direction_aggregation={aggregation}").contradiction
    _, found = detect_contradictions([[a], [b]], embedder, nli, cfg)
    assert len(found) == 1
    assert found[0].score == pytest.approx(expected)
    assert found[0].score_ab == pytest.approx(0.1) and found[0].score_ba == pytest.approx(0.9)


def test_one_directional_score_filtered_by_symmetric_aggregation(embedder):
    a, b = "Alpha division revenue rose sharply.", "Beta division revenue fell sharply."
    nli = FakeNLI({(a, b): [0.1, 0.1, 0.8], (b, a): [0.9, 0.05, 0.05]})
    cfg = make_cfg("contradiction.pair_min_similarity=0", "contradiction.direction_aggregation=min").contradiction
    assert detect_contradictions([[a], [b]], embedder, nli, cfg)[1] == []


REMOVE = "contradiction.action=remove"


def test_resolution_keeps_claim_supported_by_its_own_source(embedder, nli):
    cfg = make_cfg(REMOVE)
    spec = [("Ops", ["Sensor revenue fell because of the plant shutdown.", "Repairs cost four million."]),
            ("Revenue", ["European demand weakened sharply.", "Automotive revenue grew."])]
    index = build_index(spec, embedder)
    report = check_section_summaries(summaries_from([[CAUSE], [DENIAL, "Automotive revenue grew."]]),
                                     index, embedder, nli, cfg.contradiction)
    assert len(report.contradictions) == 1
    c = report.contradictions[0]
    assert c.kept.text == CAUSE and c.rejected.text == DENIAL
    assert c.support_a > c.support_b
    assert report.corrected_sentences == [[CAUSE], ["Automotive revenue grew."]]


def test_default_action_flags_instead_of_removing(embedder, nli, cfg):
    """Default is action=flag (FINDINGS 10.5): the resolution is still decided and reported, but the
    losing claim stays in the summary, marked, because on real documents most removals were wrong."""
    assert cfg.contradiction.action == "flag"
    spec = [("Ops", ["Sensor revenue fell because of the plant shutdown.", "Repairs cost four million."]),
            ("Revenue", ["European demand weakened sharply.", "Automotive revenue grew."])]
    index = build_index(spec, embedder)
    report = check_section_summaries(summaries_from([[CAUSE], [DENIAL, "Automotive revenue grew."]]),
                                     index, embedder, nli, cfg.contradiction)
    c = report.contradictions[0]
    assert c.kept.text == CAUSE and c.rejected.text == DENIAL
    assert report.corrected_sentences == [[CAUSE], [DENIAL, "Automotive revenue grew."]]
    assert "1:0" in report.flagged


def test_unresolved_when_both_claims_are_supported(embedder, nli, cfg):
    spec = [("Ops", [CAUSE]), ("Revenue", [DENIAL])]      # the SOURCE itself is inconsistent
    index = build_index(spec, embedder)
    report = check_section_summaries(summaries_from([[CAUSE], [DENIAL]]), index, embedder, nli, cfg.contradiction)
    c = report.contradictions[0]
    assert c.resolution == "unresolved"
    assert report.corrected_sentences == [[CAUSE], [DENIAL]]
    assert set(report.flagged) == {"0:0", "1:0"}


def test_flag_action_keeps_rejected_claim_with_note(embedder, nli):
    cfg = make_cfg("contradiction.action=flag").contradiction
    spec = [("Ops", [CAUSE]), ("Revenue", ["European demand weakened."])]
    index = build_index(spec, embedder)
    report = check_section_summaries(summaries_from([[CAUSE], [DENIAL]]), index, embedder, nli, cfg)
    assert report.corrected_sentences == [[CAUSE], [DENIAL]]
    assert report.flagged["1:0"].startswith("REJECTED")


def test_superseded_when_claim_already_rejected(embedder, nli, cfg):
    denial2 = "Sensor revenue did not fall because of the plant shutdown."
    claims = [[CAUSE], [DENIAL], [denial2]]
    spec = [("Ops", [CAUSE]), ("Rev", ["Demand fell."]), ("Other", ["Demand fell again."])]
    index = build_index(spec, embedder)
    _, found = detect_contradictions(claims, embedder, nli, cfg.contradiction)
    resolve_contradictions(found, claims, index, embedder, nli, cfg.contradiction)
    resolutions = sorted(c.resolution for c in found)
    # CAUSE beats both denials; the denial-vs-denial pair is not a contradiction at all.
    assert resolutions == ["kept_a", "kept_a"]


def test_disabled_stage_passes_summaries_through(embedder, nli):
    cfg = make_cfg("contradiction.enabled=false").contradiction
    index = build_index([("A", [CAUSE]), ("B", [DENIAL])], embedder)
    report = check_section_summaries(summaries_from([[CAUSE], [DENIAL]]), index, embedder, nli, cfg)
    assert report.contradictions == [] and report.corrected_sentences == [[CAUSE], [DENIAL]]


def test_source_diagnostic_marks_but_never_removes(embedder, nli, cfg):
    index = build_index([("A", [CAUSE, "Costs rose."]), ("B", [DENIAL])], embedder)
    report = diagnose_source_sections(index, embedder, nli, cfg.contradiction)
    assert len(report.contradictions) == 1
    assert report.contradictions[0].resolution == "diagnostic"
    assert report.corrected_sentences == [[CAUSE, "Costs rose."], [DENIAL]]


# ------------------------------------------------------------------ non-claim filter


def test_pleasantries_and_speaker_labels_are_not_claims(cfg):
    c = cfg.contradiction
    assert not is_claim("Operator: Thank you.", c)
    assert not is_claim("Mei Lin Zhou, Chief Financial Officer: Thank you, Daniel.", c)
    assert is_claim("Mei Lin Zhou, Chief Financial Officer: Total revenue was 186 million dollars.", c)


def test_speaker_label_stripping():
    assert strip_speaker_label("Daniel Okafor, Chief Executive Officer: We plan to grow.") == "We plan to grow."
    assert strip_speaker_label("Operator: Good afternoon.") == "Good afternoon."
    assert strip_speaker_label("Revenue grew 4% in the quarter.") == "Revenue grew 4% in the quarter."


def test_different_speakers_are_not_compared_as_claims(embedder):
    nli = FakeNLI()
    claims = [["Priya Raman, Investor Relations: Thank you, operator."],
              ["Aaron Feld, Brookline Securities: Thanks for taking my question."]]
    stats, found = detect_contradictions(claims, embedder, nli, make_cfg().contradiction)
    assert (stats.total, found) == (0, []) and nli.calls == []


# ------------------------------------------------------------------ one-sided check

ALL_MIGRATED = "Every enterprise customer now runs on the new platform."
SOME_LEFT = "Every enterprise customer now runs on the new platform not yet."   # fake-NLI negation of the above


def one_sided_setup(embedder, own_support_for_claim=True):
    ceo_source = [ALL_MIGRATED if own_support_for_claim else "The platform launch went smoothly overall.",
                  "Support tickets dropped by a third."]
    cfo_source = ["Revenue grew fourteen percent year over year.", SOME_LEFT]
    index = build_index([("CEO", ceo_source), ("CFO", cfo_source)], embedder)
    # The CFO summary DROPPED the contradicting sentence; only the CEO claim survives.
    summaries = summaries_from([[ALL_MIGRATED], ["Revenue grew fourteen percent year over year."]])
    return index, summaries


def test_one_sided_contradiction_is_flagged_when_other_side_was_dropped(embedder, nli, cfg):
    index, summaries = one_sided_setup(embedder)
    report = check_section_summaries(summaries, index, embedder, nli, cfg.contradiction)
    assert report.contradictions == []                      # summary-vs-summary cannot see it
    assert len(report.one_sided) == 1
    c = report.one_sided[0]
    assert c.kind == "one_sided" and c.claim_a.text == ALL_MIGRATED and c.claim_b.text == SOME_LEFT
    assert c.resolution == "unresolved"                     # both sides are supported by their own sections
    assert "CONTRADICTED" in report.flagged["0:0"]
    assert report.corrected_sentences[0] == [ALL_MIGRATED]  # kept, but flagged


def test_one_sided_rejects_unsupported_summary_claim(embedder, nli):
    cfg = make_cfg(REMOVE)
    index, summaries = one_sided_setup(embedder, own_support_for_claim=False)
    report = check_section_summaries(summaries, index, embedder, nli, cfg.contradiction)
    c = report.one_sided[0]
    assert c.resolution == "kept_b" and c.rejected.text == ALL_MIGRATED
    assert report.corrected_sentences[0] == []


def test_one_sided_skips_source_sentences_already_in_their_summary(embedder, nli, cfg):
    index = build_index([("A", [CAUSE]), ("B", [DENIAL])], embedder)
    report = check_section_summaries(summaries_from([[CAUSE], [DENIAL]]), index, embedder, nli, cfg.contradiction)
    assert len(report.contradictions) == 1 and report.one_sided == []


def test_one_sided_can_be_disabled(embedder, nli):
    index, summaries = one_sided_setup(embedder)
    cfg = make_cfg("contradiction.check_sibling_sources=false").contradiction
    assert check_section_summaries(summaries, index, embedder, nli, cfg).one_sided == []


def test_flags_accumulate_for_a_claim_contradicted_twice(embedder, nli, cfg):
    """Regression: the second note used to overwrite the first, hiding a conflict."""
    claim = "Every branch was converted to the new system."
    other_a = "Every branch was converted to the new system not at all."
    other_b = "Every branch was converted to the new system not yet."
    index = build_index([("A", [claim]), ("B", [other_a]), ("C", [other_b])], embedder)
    report = check_section_summaries(summaries_from([[claim], [other_a], [other_b]]),
                                     index, embedder, nli, cfg.contradiction)
    assert len(report.contradictions) >= 2
    note = report.flagged.get("0:0", "")
    assert note.count("UNRESOLVED") >= 2, f"expected both contradictions in the note, got: {note}"
