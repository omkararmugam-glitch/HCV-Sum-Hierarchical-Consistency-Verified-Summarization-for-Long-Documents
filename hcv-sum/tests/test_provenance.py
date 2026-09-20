from hcv_sum.provenance import tag_provenance
from hcv_sum.types import ClaimRef, Contradiction, ContradictionReport, MergeResult

from conftest import FakeNLI, build_index, make_cfg

SPEC = [
    ("Ops", ["The Monterrey plant closed for three weeks.", "A transformer fire caused the closure.",
             "Repair costs were recorded."]),
    ("Revenue", ["Sensor revenue fell eighteen percent.", "Automotive revenue grew four percent."]),
]


def merge_of(*sentences):
    return MergeResult("extractive", list(sentences), [], [], " ".join(sentences), list(sentences), [])


def run(sentences, embedder, nli=None, report=None, *overrides, resurrected=None):
    index = build_index(SPEC, embedder)
    merge = merge_of(*sentences)
    if resurrected:
        merge.resurrected = resurrected
    cfg = make_cfg(*overrides).provenance
    return tag_provenance(merge, index, embedder, nli or FakeNLI(), cfg, report)


def test_supported_sentence_cites_correct_source(embedder):
    [rec] = run(["Sensor revenue fell eighteen percent."], embedder)
    assert rec.status == "supported"
    assert rec.citation.section_index == 1 and rec.citation.sentence_indices == [3]
    assert rec.citation.section_title == "Revenue"
    assert rec.entailment >= 0.7


def test_off_topic_sentence_is_unsupported_with_flag(embedder):
    [rec] = run(["Quantum computers will replace spreadsheets."], embedder)
    assert rec.status == "unsupported" and rec.citation is None
    assert "no source sentence above similarity" in rec.flags[0]


def test_similarity_alone_is_not_enough(embedder):
    # Highly similar to a source sentence but negated: must not be "supported".
    [rec] = run(["Sensor revenue did not fall eighteen percent."], embedder)
    assert rec.similarity > 0.7          # passes the embedding filter comfortably (floor 0.30)
    assert rec.status == "unsupported"
    assert any(f.startswith("contradicted_by_source") for f in rec.flags)


def test_window_premise_supports_fused_sentence(embedder):
    fused = "Monterrey plant closed three weeks transformer fire caused closure."
    [single] = run([fused], embedder, None, None, "provenance.window=1")
    [windowed] = run([fused], embedder, None, None, "provenance.window=2")
    assert single.status == "unsupported"
    assert windowed.status == "supported"
    assert windowed.citation.sentence_indices == [0, 1]


def test_stage3_decision_is_attached(embedder):
    kept = ClaimRef(1, 0, "Sensor revenue fell eighteen percent.")
    lost = ClaimRef(0, 2, "Sensor revenue did not fall.")
    c = Contradiction(kept, lost, 0.9, 0.9, 0.2, 0.95, 0.1, resolution="kept_a", reason="better supported")
    report = ContradictionReport(1, 1, [c], [[], []])
    [rec] = run(["Sensor revenue fell eighteen percent."], embedder, None, report)
    assert rec.stage3_notes and "kept this claim" in rec.stage3_notes[0]
    assert "Sensor revenue did not fall." in rec.stage3_notes[0]


def test_resurrection_flag_is_carried(embedder):
    s = "Sensor revenue fell eighteen percent."
    [rec] = run([s], embedder, None, None, resurrected=[(s, "rejected claim text", 0.9)])
    assert any("resurrected" in f for f in rec.flags)


def test_disputed_flag_for_one_sided_unresolved(embedder):
    sentence = "Sensor revenue fell eighteen percent."
    claim = ClaimRef(1, 0, sentence)
    source = ClaimRef(0, 2, "Sensor revenue did not fall eighteen percent.")
    c = Contradiction(claim, source, 0.9, 0.9, 0.9, 0.99, 0.98, resolution="unresolved",
                      reason="both supported", kind="one_sided")
    report = ContradictionReport(0, 0, [], [[], [sentence]], one_sided=[c])
    [rec] = run([sentence], embedder, None, report)
    assert any(f.startswith("DISPUTED") for f in rec.flags)
    assert any("one-sided" in n for n in rec.stage3_notes)


def test_novel_token_flag_catches_corrupted_number(embedder):
    # Real DistilBART corruption: "twenty-four (24) months" -> "twenty-24 months".
    [rec] = run(["The Monterrey plant closed for twenty-24 weeks."], embedder)
    assert any(f.startswith("not in source text") and "twenty-24" in f for f in rec.flags)


def test_novel_token_flag_quiet_on_verbatim_sentence(embedder):
    [rec] = run(["Sensor revenue fell eighteen percent."], embedder)
    assert not any(f.startswith("not in source text") for f in rec.flags)


def test_novel_token_flag_can_be_disabled(embedder):
    [rec] = run(["The Monterrey plant closed for twenty-24 weeks."], embedder, None, None,
                "provenance.flag_novel_tokens=false")
    assert not any(f.startswith("not in source text") for f in rec.flags)


# ------------------------------------------------------------------ duplicate Stage 3 notes (evidence panel)
def test_one_conflict_flagged_from_two_sections_is_noted_once(embedder):
    """Regression: S4/S8 of the 01_planted_contradiction run each listed the SAME Stage 3 note twice.

    A claim is a (section, sentence, text) reference, so a claim text that appears in two sections'
    summaries is two ClaimRefs, and one textual conflict becomes two Contradiction objects. Both render
    to the same note, because a note names only the other side's text and section.
    """
    sentence = "Sensor revenue fell eighteen percent."
    partner = ClaimRef(0, 2, "Automotive revenue grew four percent.")
    same_conflict = [Contradiction(ClaimRef(section, 0, sentence), partner, 0.9, 0.9, 0.9,
                                   resolution="unresolved", reason="both supported")
                     for section in (1, 2)]
    report = ContradictionReport(2, 2, same_conflict, [[], [sentence], [sentence]])
    [rec] = run([sentence], embedder, None, report)
    assert len(rec.stage3_notes) == len(set(rec.stage3_notes)), f"repeated notes: {rec.stage3_notes}"
    assert sum("Automotive revenue grew four percent." in n for n in rec.stage3_notes) == 1


def test_two_different_conflicts_over_one_sentence_are_both_noted(embedder):
    """The de-duplication must not collapse distinct conflicts: both partners have to stay visible."""
    sentence = "Sensor revenue fell eighteen percent."
    conflicts = [Contradiction(ClaimRef(1, 0, sentence), ClaimRef(0, 2, partner), 0.9, 0.9, 0.9,
                               resolution="unresolved", reason="both supported")
                 for partner in ("Automotive revenue grew four percent.", "Repair costs were recorded.")]
    report = ContradictionReport(2, 2, conflicts, [[], [sentence]])
    [rec] = run([sentence], embedder, None, report)
    assert sum("Automotive revenue grew" in n for n in rec.stage3_notes) == 1
    assert sum("Repair costs were recorded" in n for n in rec.stage3_notes) == 1


def test_conflict_whose_both_sides_resemble_the_sentence_is_noted_once(embedder):
    """Regression (S4/S8 of the 01_planted_contradiction run): one conflict, two notes.

    The two sides of a contradiction differ by a negation, which sentence embeddings largely ignore, so a
    final sentence taken from one side cleared link_similarity against BOTH sides. The panel then listed the
    same conflict twice back to back -- same prefix, same reason, same score, the quoted claims differing by
    one word -- and the second note named the sentence's own claim as the contradicting one.
    """
    sentence = "Sensor revenue fell eighteen percent."
    negated = "Sensor revenue fell not eighteen percent."
    c = Contradiction(ClaimRef(1, 0, sentence), ClaimRef(0, 2, negated), 0.9, 0.9, 0.9,
                      resolution="unresolved", reason="both supported")
    report = ContradictionReport(1, 1, [c], [[], [sentence]])
    [rec] = run([sentence], embedder, None, report)
    unresolved = [n for n in rec.stage3_notes if n.startswith("Stage 3 UNRESOLVED")]
    assert len(unresolved) == 1, f"one conflict must give one note, got: {unresolved}"
    assert negated in unresolved[0], "the note names the other side"
    assert "(section 0)" in unresolved[0], "...and the other side's section, not the sentence's own"


def test_possibly_superseded_note_is_also_listed_once(embedder):
    """The multi-document resolution used the same two-way loop, so it had the same duplication."""
    sentence = "Sensor revenue fell eighteen percent."
    negated = "Sensor revenue fell not eighteen percent."
    c = Contradiction(ClaimRef(1, 0, sentence), ClaimRef(0, 2, negated), 0.9, 0.9, 0.9,
                      resolution="possibly_superseded", reason="Q1 2026 vs Q2 2026")
    report = ContradictionReport(1, 1, [c], [[], [sentence]])
    [rec] = run([sentence], embedder, None, report)
    assert len([n for n in rec.stage3_notes if "POSSIBLY SUPERSEDED" in n]) == 1


def test_same_opposing_claim_flagged_twice_with_different_reasons_is_noted_once(embedder):
    """Grouping keys on the opposing CLAIM, not the rendered note.

    One conflict becomes one Contradiction per section holding the claim, and each reason quotes that
    section's own support scores -- so the rendered notes differ in their numbers and de-duplicating by
    text cannot collapse them, even though they tell the reader the same thing.
    """
    sentence = "Sensor revenue fell eighteen percent."
    partner = "Automotive revenue grew four percent."
    conflicts = [Contradiction(ClaimRef(section, 0, sentence), ClaimRef(0, 2, partner), 0.9, 0.9, 0.9,
                               resolution="unresolved", reason=reason)
                 for section, reason in ((1, "support gap 0.05 (A=0.95, B=0.90)"),
                                         (2, "support gap 0.09 (A=0.81, B=0.90)"))]
    report = ContradictionReport(2, 2, conflicts, [[], [sentence], [sentence]])
    [rec] = run([sentence], embedder, None, report)
    assert sum(partner in n for n in rec.stage3_notes) == 1, rec.stage3_notes


def test_grouping_keeps_every_distinct_opposing_claim(embedder):
    """Three different opposing claims over one sentence stay three notes (never collapsed to one)."""
    sentence = "Sensor revenue fell eighteen percent."
    partners = ["Automotive revenue grew four percent.", "Repair costs were recorded.",
                "A transformer fire caused the closure."]
    conflicts = [Contradiction(ClaimRef(1, 0, sentence), ClaimRef(0, i, partner), 0.9, 0.9, 0.9,
                               resolution="unresolved", reason="both supported")
                 for i, partner in enumerate(partners)]
    report = ContradictionReport(3, 3, conflicts, [[], [sentence]])
    [rec] = run([sentence], embedder, None, report)
    for partner in partners:
        assert sum(partner in n for n in rec.stage3_notes) == 1, f"{partner} missing from {rec.stage3_notes}"
