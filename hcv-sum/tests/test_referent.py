"""Grounded referent re-check (src/hcv_sum/referent.py). Not wired into the pipeline; validated in
FINDINGS.md 13 against data/eval/REFERENT_CHECK_PREREGISTRATION.md."""

from hcv_sum.referent import ground_claim, referent_recheck
from hcv_sum.types import ClaimRef, Contradiction

from conftest import FakeNLI, build_index, make_cfg

BASELINES = ["We compare against four baselines.", "The second is an autoencoder that treats each sensor independently.",
             "The third is a static graph network that uses a fixed adjacency matrix from the wiring diagram."]
INTRO = ["In this work we introduce a framework that learns a dynamic sensor graph."]


def test_generated_claim_grounds_to_its_source_sentence_with_antecedent_and_title(embedder):
    index = build_index([("Introduction", INTRO), ("4.2 Baselines", BASELINES)], embedder)
    g = ground_claim(ClaimRef(1, 0, "The network uses a fixed adjacency matrix from the wiring diagram."), index, embedder)
    assert g.sentence == BASELINES[2]
    assert g.antecedent == BASELINES[1]
    assert g.as_premise().startswith("[4.2 Baselines] The second is")


def test_source_side_claim_grounds_to_itself(embedder):
    index = build_index([("Introduction", INTRO), ("4.2 Baselines", BASELINES)], embedder)
    g = ground_claim(ClaimRef(1, 0, BASELINES[0]), index, embedder, is_source=True)
    assert g.sentence == BASELINES[0] and g.antecedent == "" and g.similarity == 1.0


def test_context_goes_on_the_premise_side_only(embedder):
    index = build_index([("Introduction", INTRO), ("4.2 Baselines", BASELINES)], embedder)
    flag = Contradiction(ClaimRef(0, 0, INTRO[0]),
                         ClaimRef(1, 0, "The network uses a fixed adjacency matrix from the wiring diagram."), 0.9, 0.9, 0.9)
    nli = FakeNLI()
    referent_recheck([flag], index, embedder, nli, make_cfg().contradiction)
    (p1, h1), (p2, h2) = nli.calls
    assert p1.startswith("[Introduction]") and h1 == flag.claim_b.text
    assert p2.startswith("[4.2 Baselines]") and h2 == flag.claim_a.text


def test_confirmation_uses_the_detectors_own_threshold(embedder):
    index = build_index([("A", ["Sensor revenue fell because of the plant shutdown."]),
                         ("B", ["Sensor revenue fell not because of the plant shutdown."])], embedder)
    flag = Contradiction(ClaimRef(0, 0, "Sensor revenue fell because of the plant shutdown."),
                         ClaimRef(1, 0, "Sensor revenue fell not because of the plant shutdown."), 0.9, 0.9, 0.9)
    table = {}
    nli = FakeNLI(table)
    [r] = referent_recheck([flag], index, embedder, nli, make_cfg().contradiction)
    assert r["confirmed"] and r["score"] >= 0.5          # a real contradiction survives its own context
    table.update({call: [0.05, 0.05, 0.90] for call in nli.calls})
    [r] = referent_recheck([flag], index, embedder, FakeNLI(table), make_cfg().contradiction)
    assert not r["confirmed"]


def test_nothing_is_wired_into_the_pipeline_by_default(cfg):
    import hcv_sum.contradiction as contradiction
    import hcv_sum.pipeline as pipeline
    assert "referent" not in contradiction.__dict__ and "referent_recheck" not in pipeline.__dict__
    assert not hasattr(cfg.contradiction, "referent_check")
