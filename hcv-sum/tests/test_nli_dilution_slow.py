"""Pin the mechanism behind the missed explicit numeric contradiction in sample 05 (FINDINGS.md 12.1).

These are MODEL-BEHAVIOUR tests, not pipeline tests: they record how the configured NLI model
(DeBERTa-v3-base-mnli-fever-anli) responds, so that a model swap, a new aggregation rule or a
"fix" that changes this behaviour fails loudly and someone has to update FINDINGS.md 12.1 with why.
Measured values (scripts/probe_dilution.py, outputs/probe_dilution.txt) are in the comments; the
assertion thresholds leave a wide margin around them.

What they establish:
- the conflict itself is easy: the bare pair scores ~0.99 in both directions;
- as written in the paper it scores 0.15, and max-aggregation would not rescue it (0.23);
- LENGTH is not the cause: restating the same fact to 30 tokens, or adding the SAME clause to both
  sides, keeps it ~0.98-0.996;
- a QUALIFIER ON THE QUANTITY in the hypothesis ("eighteen months OF SENSOR READINGS") collapses that
  direction (0.016), while the same words as a separate clause do not (0.916), and the same qualifier
  on both sides restores it (0.993);
- a reporting frame in the hypothesis ("We also note that ...") collapses that direction (0.009 on a
  synthetic pair) while the other direction stays ~1.0 -- the mean then sits on the 0.5 threshold.
"""

import pytest

from hcv_sum.config import load_config
from hcv_sum.models import CONTRADICTION, ModelRegistry

pytestmark = pytest.mark.slow

A_DOC = ("The automotive dataset spans eighteen months of sensor readings from a stamping line, including two "
         "hundred and forty individual sensors.")
B_DOC = ("We also note that the automotive dataset, spanning twenty-four months of continuous stamping line "
         "operation, provided a particularly stable basis for evaluating long-term drift in sensor correlation "
         "patterns, since longer observation windows make gradual equipment aging easier to distinguish from "
         "short-term noise.")
A_BARE, B_BARE = "The automotive dataset spans eighteen months.", "The automotive dataset spans twenty-four months."
CLAUSE = "longer observation windows make gradual equipment aging easier to distinguish from short-term noise"


@pytest.fixture(scope="module")
def contradiction():
    nli = ModelRegistry(load_config()).nli

    def score(a: str, b: str) -> tuple[float, float]:
        probs = nli.predict([(a, b), (b, a)])
        return float(probs[0, CONTRADICTION]), float(probs[1, CONTRADICTION])
    return score


def test_bare_numeric_conflict_is_easy(contradiction):
    ab, ba = contradiction(A_BARE, B_BARE)          # measured 0.996 / 0.991
    assert ab > 0.9 and ba > 0.9


def test_as_written_in_the_document_it_is_missed_in_both_directions(contradiction):
    ab, ba = contradiction(A_DOC, B_DOC)            # measured 0.231 / 0.072, mean 0.151
    assert (ab + ba) / 2 < 0.35
    assert max(ab, ba) < 0.5, "max-aggregation would not rescue this pair either"


def test_length_alone_does_not_suppress(contradiction):
    padded = ("The automotive dataset spans twenty-four months, which is to say that the automotive dataset spans a "
              "period of twenty-four months in total.")
    ab, ba = contradiction(A_BARE, padded)          # measured 0.971 / 0.997
    assert min(ab, ba) > 0.85
    shared_a = f"The automotive dataset spans eighteen months, and {CLAUSE}."
    shared_b = f"The automotive dataset spans twenty-four months, and {CLAUSE}."
    ab, ba = contradiction(shared_a, shared_b)      # measured 0.997 / 0.995 at 26/28 tokens
    assert min(ab, ba) > 0.85


def test_qualifier_on_the_quantity_collapses_the_hypothesis_direction(contradiction):
    qualified = "The automotive dataset spans eighteen months of sensor readings from a stamping line."
    _, ba = contradiction(qualified, B_BARE)        # A is the HYPOTHESIS in B->A; measured 0.010
    assert ba < 0.2
    as_clause = "The automotive dataset spans eighteen months and contains sensor readings from a stamping line."
    _, ba = contradiction(as_clause, B_BARE)        # same words, not attached to the number; measured 0.916
    assert ba > 0.7
    both = "The automotive dataset spans twenty-four months of sensor readings from a stamping line."
    ab, ba = contradiction(qualified, both)         # matched qualifier on both sides; measured 0.996 / 0.993
    assert min(ab, ba) > 0.85


def test_reporting_frame_collapses_the_hypothesis_direction_only(contradiction):
    ab, ba = contradiction("The clinical pilot enrolled 120 patients.",
                           "We also note that the clinical pilot enrolled 85 patients.")
    assert ab < 0.2                                  # framed sentence is the hypothesis; measured 0.009
    assert ba > 0.9                                  # framed sentence is the premise;   measured 0.999


def test_removing_the_frame_from_the_document_sentence_restores_one_direction(contradiction):
    no_frame = ("The automotive dataset, spanning twenty-four months of continuous stamping line operation, provided "
                "a particularly stable basis for evaluating long-term drift in sensor correlation patterns.")
    ab, _ = contradiction(A_DOC, no_frame)           # measured 0.949 (0.203 with 'We also note that')
    assert ab > 0.7
