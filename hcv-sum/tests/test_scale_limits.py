"""Scale controls: candidate_top_k, max_pairs and max_sections.

These bound the pipeline's cost on long documents. Each one skips real work, so each test also
pins WHAT is skipped and checks the skip is reported rather than hidden.
"""

import numpy as np

from hcv_sum.contradiction import detect_contradictions, select_pairs
from hcv_sum.segmentation import minimum_segment_size, segment_document

from conftest import FakeEmbedder, FakeNLI, build_index, make_cfg


def eligible_all(rows, cols, same_section=None):
    return [[j for j in range(cols) if same_section is None or same_section(i, j)] for i in range(rows)]


def test_similarity_floor_skips_unrelated_pairs():
    sims = np.array([[1.0, 0.9, 0.1], [0.9, 1.0, 0.05], [0.1, 0.05, 1.0]], dtype=np.float32)
    cfg = make_cfg("contradiction.pair_min_similarity=0.5", "contradiction.candidate_top_k=0",
                   "contradiction.max_pairs=0").contradiction
    pairs, top_k, budget = select_pairs(sims, eligible_all(3, 3, lambda i, j: i != j), cfg, symmetric=True)
    assert sorted(pairs) == [(0, 1)]
    assert (top_k, budget) == (0, 0)


def test_candidate_top_k_keeps_the_most_similar_and_reports_the_rest():
    # row 0 is similar to 1, 2 and 3; with top_k=1 only the best survives.
    sims = np.array([[1.0, 0.9, 0.8, 0.7]], dtype=np.float32)
    cfg = make_cfg("contradiction.pair_min_similarity=0.5", "contradiction.candidate_top_k=1",
                   "contradiction.max_pairs=0").contradiction
    pairs, top_k, budget = select_pairs(sims, [[1, 2, 3]], cfg, symmetric=False)
    assert pairs == [(0, 1)], "the top-k cut must keep the HIGHEST similarity candidate"
    assert top_k == 2 and budget == 0


def test_max_pairs_budget_keeps_the_most_similar_pairs():
    sims = np.array([[1.0, 0.9, 0.8, 0.7]], dtype=np.float32)
    cfg = make_cfg("contradiction.pair_min_similarity=0.5", "contradiction.candidate_top_k=0",
                   "contradiction.max_pairs=2").contradiction
    pairs, top_k, budget = select_pairs(sims, [[1, 2, 3]], cfg, symmetric=False)
    assert sorted(pairs) == [(0, 1), (0, 2)]
    assert budget == 1 and top_k == 0


def test_symmetric_mode_compares_each_pair_once():
    sims = np.full((4, 4), 0.9, dtype=np.float32)
    cfg = make_cfg("contradiction.pair_min_similarity=0.5", "contradiction.candidate_top_k=0",
                   "contradiction.max_pairs=0").contradiction
    pairs, _, _ = select_pairs(sims, eligible_all(4, 4, lambda i, j: i != j), cfg, symmetric=True)
    assert len(pairs) == 6 == len({tuple(sorted(p)) for p in pairs})    # C(4,2), no duplicates


def test_top_k_bounds_nli_work_on_a_wide_document(embedder, nli):
    """The point of the limit: comparisons grow linearly with claims, not quadratically."""
    claims = [[f"Division {i} revenue grew four percent in the quarter."] for i in range(40)]
    unlimited = make_cfg("contradiction.candidate_top_k=0", "contradiction.pair_min_similarity=0.0").contradiction
    limited = make_cfg("contradiction.candidate_top_k=5", "contradiction.pair_min_similarity=0.0").contradiction
    stats_all, _ = detect_contradictions(claims, embedder, FakeNLI(), unlimited)
    stats_cap, _ = detect_contradictions(claims, embedder, FakeNLI(), limited)
    assert stats_all.checked == 40 * 39 // 2          # every cross-section pair
    assert stats_cap.checked <= 40 * 5                # bounded by claims x k
    assert stats_cap.skipped_top_k > 0                # and the skip is reported
    assert stats_cap.nli_calls == stats_cap.checked * 2


def test_budget_is_reported_not_hidden(embedder):
    claims = [[f"Division {i} revenue grew four percent in the quarter."] for i in range(30)]
    cfg = make_cfg("contradiction.max_pairs=10", "contradiction.candidate_top_k=0",
                   "contradiction.pair_min_similarity=0.0").contradiction
    stats, _ = detect_contradictions(claims, embedder, FakeNLI(), cfg)
    assert stats.checked == 10
    assert stats.skipped_budget == 30 * 29 // 2 - 10
    assert stats.total == 30 * 29 // 2


# ------------------------------------------------------------------ section cap

def test_minimum_segment_size_grows_with_document_length():
    cfg = make_cfg("segmentation.min_segment_sentences=4", "segmentation.max_sections=250").segmentation
    assert minimum_segment_size(100, cfg) == 4          # short document: config value applies
    assert minimum_segment_size(6000, cfg) == 24        # 6000/250, so at most 250 sections
    uncapped = make_cfg("segmentation.max_sections=0").segmentation
    assert minimum_segment_size(6000, uncapped) == uncapped.min_segment_sentences


def test_max_sections_caps_unstructured_segmentation():
    embedder = FakeEmbedder()
    topics = [f"Topic {i} concerns widget {i} and its distinct supply chain characteristics." for i in range(60)]
    doc = " ".join(f"{t} {t} {t}" for t in topics)      # 180 sentences, 60 distinct topics
    words = lambda t: len(t.split())                     # noqa: E731 - test stub

    capped = segment_document(doc, embedder, words, make_cfg("segmentation.max_sections=5",
                                                             "segmentation.min_segment_sentences=2").segmentation)
    assert len(capped) <= 5
    uncapped = segment_document(doc, embedder, words, make_cfg("segmentation.max_sections=0",
                                                               "segmentation.min_segment_sentences=2").segmentation)
    assert len(uncapped) > len(capped), "without the cap this document fragments into many sections"


def test_default_config_scale_limits_are_set(cfg):
    assert cfg.contradiction.candidate_top_k > 0, "an unbounded default would be O(n^2) on long documents"
    assert cfg.contradiction.max_pairs > 0
    assert cfg.segmentation.max_sections > 0
    assert cfg.summarization.batch_size >= 1


# ------------------------------------------------------------------ resolution budget

def test_resolution_budget_leaves_extra_flags_unscored_but_visible(embedder, nli):
    """Resolution cost scales with the number of FLAGS, so it needs its own budget.

    Unscored pairs must still appear in the report, and nothing may be removed on their account.
    """
    from hcv_sum.contradiction import check_section_summaries
    from hcv_sum.types import SectionSummary

    claims = [[f"Division {i} revenue grew four percent this quarter."] for i in range(12)]
    claims += [[f"Division {i} revenue grew four percent this quarter not at all."] for i in range(12)]
    spec = [(f"S{i}", list(c)) for i, c in enumerate(claims)]
    index = build_index(spec, embedder)
    summaries = [SectionSummary(i, "", " ".join(c), list(c), [], True, False) for i, c in enumerate(claims)]

    cfg = make_cfg("contradiction.max_resolutions=3", "contradiction.max_resolutions_per_section=0",
                   "contradiction.pair_min_similarity=0.0").contradiction
    report = check_section_summaries(summaries, index, embedder, nli, cfg)
    assert len(report.contradictions) > 3, "this setup should flag more pairs than the budget allows"
    assert report.unscored > 0
    resolved = [c for c in report.contradictions if c.resolution != "unscored"]
    unscored = [c for c in report.contradictions if c.resolution == "unscored"]
    assert len(resolved) == 3
    assert all("max_resolutions" in c.reason for c in unscored)
    # nothing removed for an unscored pair: those claims are all still present
    kept = {s for sec in report.corrected_sentences for s in sec}
    for c in unscored:
        assert c.claim_a.text in kept and c.claim_b.text in kept
    # unscored pairs are the LEAST contradictory ones (highest scores are resolved first)
    assert min(c.score for c in resolved) >= max(c.score for c in unscored)


def test_resolution_budget_off_by_default_on_small_documents(embedder, nli, cfg):
    from hcv_sum.contradiction import check_section_summaries
    from hcv_sum.types import SectionSummary

    claims = [["Sensor revenue fell because of the plant shutdown."],
              ["Sensor revenue fell not because of the plant shutdown."]]
    spec = [("A", list(claims[0])), ("B", list(claims[1]))]
    index = build_index(spec, embedder)
    summaries = [SectionSummary(i, "", " ".join(c), list(c), [], True, False) for i, c in enumerate(claims)]
    report = check_section_summaries(summaries, index, embedder, nli, cfg.contradiction)
    assert report.unscored == 0, "a normal document must never hit the budget"
