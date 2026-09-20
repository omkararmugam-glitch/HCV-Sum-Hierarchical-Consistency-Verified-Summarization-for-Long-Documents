"""Reporting-frame stripping before NLI (FINDINGS.md 12.1 and 13.1)."""

import pytest

from hcv_sum.contradiction import detect_contradictions, nli_text
from hcv_sum.text_utils import strip_reporting_frame

from conftest import FakeNLI, make_cfg


@pytest.mark.parametrize("framed, bare", [
    ("We also note that the automotive dataset spans twenty-four months.", "The automotive dataset spans twenty-four months."),
    ("We further show that the learned sensor embeddings transfer effectively.", "The learned sensor embeddings transfer effectively."),
    ("We found that revenue fell.", "Revenue fell."),
    ("It is worth noting that the plant closed.", "The plant closed."),
    ("It should be noted that the term is 24 months.", "The term is 24 months."),
    ("Note that the term is 24 months.", "The term is 24 months."),
    ("Our results show that lead time improved.", "Lead time improved."),
])
def test_reporting_frames_are_stripped(framed, bare):
    assert strip_reporting_frame(framed) == bare


@pytest.mark.parametrize("sentence", [
    "We believe that the pilot enrolled 85 patients.",      # belief hedge: changes what is asserted
    "We expect transfer performance to degrade.",
    "Our results suggest that representations generalize.",
    "We did not find that revenue fell.",                   # negated: not a frame
    "We note that.",                                        # nothing left after the frame
    "Notable results were found in Section 5.",
])
def test_hedges_negations_and_non_frames_are_kept(sentence):
    assert strip_reporting_frame(sentence) == sentence


def test_nli_sees_stripped_text_but_selection_and_claims_do_not_change(embedder):
    claims = [["We also note that sensor revenue fell because of the plant shutdown."],
              ["Sensor revenue fell not because of the plant shutdown."]]
    on_nli, off_nli = FakeNLI(), FakeNLI()
    on = make_cfg("contradiction.strip_reporting_frames=true").contradiction
    off = make_cfg("contradiction.strip_reporting_frames=false").contradiction
    s_on, f_on = detect_contradictions(claims, embedder, on_nli, on)
    s_off, _ = detect_contradictions(claims, embedder, off_nli, off)
    assert s_on == s_off                                     # identical candidate selection
    assert all(not p.startswith("We also note") for pair in on_nli.calls for p in pair)
    assert any(p.startswith("We also note") for pair in off_nli.calls for p in pair)
    assert f_on and f_on[0].claim_a.text.startswith("We also note")   # the claim itself is reported unchanged


def test_frame_stripping_is_on_by_default(cfg):
    assert cfg.contradiction.strip_reporting_frames is True
    assert nli_text("Jane Doe, CFO: We found that revenue fell.", cfg.contradiction) == "Revenue fell."
