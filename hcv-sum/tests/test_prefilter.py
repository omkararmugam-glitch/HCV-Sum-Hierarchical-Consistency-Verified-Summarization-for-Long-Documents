"""Stage 3 pre-filters: the similarity threshold and the optional entity-overlap gate.

The pre-filter decides which pairs ever reach the NLI model, so these tests pin both that it cuts
work and that it cuts the RIGHT work -- an over-eager filter silently destroys recall.
"""

import numpy as np
import pytest

from hcv_sum.contradiction import detect_contradictions, filter_by_shared_entity, select_pairs
from hcv_sum.text_utils import entity_tokens

from conftest import FakeNLI, make_cfg

RELATED = ["Sensor revenue fell because of the Monterrey plant shutdown.",
           "Sensor revenue fell not because of the Monterrey plant shutdown."]
UNRELATED = "Resin lead times improved to nine weeks."


# ------------------------------------------------------------------ similarity threshold

# The fake embedder ignores negation, so the two RELATED sentences score cosine 1.0 with each
# other: only a threshold above 1.0 excludes everything.
@pytest.mark.parametrize("threshold,expected_pairs", [(0.0, 3), (0.5, 1), (1.01, 0)])
def test_similarity_threshold_controls_how_many_pairs_reach_nli(embedder, threshold, expected_pairs):
    nli = FakeNLI()
    cfg = make_cfg(f"contradiction.pair_min_similarity={threshold}",
                   "contradiction.candidate_top_k=0").contradiction
    claims = [[RELATED[0]], [RELATED[1]], [UNRELATED]]
    stats, _ = detect_contradictions(claims, embedder, nli, cfg)
    assert stats.checked == expected_pairs
    assert stats.total == 3                       # three cross-section pairs exist either way
    assert len(nli.calls) == expected_pairs * 2   # NLI is only called for pairs that survive


def test_raising_the_threshold_never_increases_nli_work(embedder):
    claims = [[RELATED[0]], [RELATED[1]], [UNRELATED], ["Freight costs fell eleven percent."]]
    counts = []
    for threshold in (0.0, 0.2, 0.4, 0.6, 0.8):
        cfg = make_cfg(f"contradiction.pair_min_similarity={threshold}",
                       "contradiction.candidate_top_k=0").contradiction
        stats, _ = detect_contradictions(claims, embedder, FakeNLI(), cfg)
        counts.append(stats.checked)
    assert counts == sorted(counts, reverse=True), f"work must fall monotonically: {counts}"


def test_prefilter_keeps_the_contradicting_pair_at_the_default_threshold(embedder, cfg):
    """The default floor must not cost recall on a pair we know is a contradiction."""
    stats, found = detect_contradictions([[RELATED[0]], [RELATED[1]], [UNRELATED]],
                                         embedder, FakeNLI(), cfg.contradiction)
    assert len(found) == 1
    assert {found[0].claim_a.text, found[0].claim_b.text} == set(RELATED)
    assert stats.skipped_similarity > 0, "the unrelated pairs should have been filtered out"


def test_skipped_counts_add_up_to_the_total(embedder):
    """Every pair must be accounted for: compared, or skipped for a stated reason."""
    claims = [[f"Division {i} revenue grew four percent this quarter."] for i in range(12)]
    cfg = make_cfg("contradiction.candidate_top_k=3", "contradiction.max_pairs=8",
                   "contradiction.pair_min_similarity=0.1").contradiction
    stats, _ = detect_contradictions(claims, embedder, FakeNLI(), cfg)
    accounted = (stats.checked + stats.skipped_similarity + stats.skipped_top_k
                 + stats.skipped_budget + stats.skipped_entity + stats.skipped_comparative)
    assert accounted == stats.total


# ------------------------------------------------------------------ entity gate

def test_entity_tokens_extracts_names_acronyms_and_figures():
    assert entity_tokens("Revenue at the Monterrey plant fell 18% to $97 million.") >= {"monterrey", "18%"}
    assert "the" not in entity_tokens("The Monterrey plant closed.")
    assert entity_tokens("RACN encodes every note.") >= {"racn"}
    # A lower-case sentence with spelled-out numbers yields nothing -- the gate's blind spot.
    assert entity_tokens("roughly sixty enterprise accounts are still served from the legacy environment") == set()


def test_entity_gate_drops_pairs_with_no_shared_entity():
    rows = ["Revenue at the Monterrey plant fell.", "Freight costs in Osaka rose."]
    cols = ["The Monterrey shutdown lasted three weeks.", "Resin lead times improved."]
    kept, skipped = filter_by_shared_entity([[0, 1], [0, 1]], rows, cols, symmetric=False)
    assert kept == [[0], []]        # only row 0 x col 0 share "monterrey"
    assert skipped == 3


def test_entity_gate_reduces_nli_calls(embedder):
    claims = [["Revenue at the Monterrey plant fell eighteen percent."],
              ["The Monterrey shutdown lasted three weeks."],
              ["Freight costs in Osaka rose eleven percent."]]
    off = make_cfg("contradiction.require_shared_entity=false", "contradiction.pair_min_similarity=0.0",
                   "contradiction.candidate_top_k=0").contradiction
    on = make_cfg("contradiction.require_shared_entity=true", "contradiction.pair_min_similarity=0.0",
                  "contradiction.candidate_top_k=0").contradiction
    stats_off, _ = detect_contradictions(claims, embedder, FakeNLI(), off)
    stats_on, _ = detect_contradictions(claims, embedder, FakeNLI(), on)
    assert stats_on.checked < stats_off.checked
    assert stats_on.skipped_entity > 0


def test_entity_gate_is_off_by_default_because_it_drops_real_contradictions(cfg, embedder):
    """Regression guard for a measured finding, not a preference.

    Sample 03's planted contradiction pairs a sentence containing "Helix Core" with one that is all
    lower case and spells its number out. They share no entity token, so the gate would drop the
    pair and the catch rate falls from 2/3 to 1/3 on the sample set.
    """
    assert cfg.contradiction.require_shared_entity is False

    a = "Every enterprise customer is now running on the new Helix Core platform."
    b = "roughly sixty enterprise accounts are still being served from the legacy environment"
    kept, skipped = filter_by_shared_entity([[0]], [a], [b], symmetric=False)
    assert kept == [[]] and skipped == 1, "documents the exact blind spot that keeps this off by default"


def test_select_pairs_is_unaffected_when_entity_gate_is_off():
    sims = np.full((3, 3), 0.9, dtype=np.float32)
    cfg = make_cfg("contradiction.require_shared_entity=false", "contradiction.candidate_top_k=0",
                   "contradiction.max_pairs=0", "contradiction.pair_min_similarity=0.5").contradiction
    pairs, top_k, budget = select_pairs(sims, [[1, 2], [0, 2], [0, 1]], cfg, symmetric=True)
    assert len(pairs) == 3 and (top_k, budget) == (0, 0)
