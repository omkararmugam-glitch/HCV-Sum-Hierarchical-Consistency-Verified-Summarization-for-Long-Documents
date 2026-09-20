"""The memory-bounded Stage 3 selection (pair_selection.py) must pick exactly what the dense path picks.

Small documents never reach it (contradiction.dense_pair_limit), so these tests force it on and
compare with the dense code on the same inputs."""

import random

import numpy as np
import pytest

from hcv_sum.contradiction import (detect_contradictions, filter_by_shared_content_word, filter_by_shared_entity,
                                   filter_comparative_pairs, resolution_budget, select_pairs, use_chunked)
from hcv_sum.pair_selection import select_pairs_chunked

from conftest import make_cfg

WORDS = ("revenue margin plant shutdown Monterrey sensor demand europe automotive contract term renewal "
         "notice payment invoice MORPHO XIMAGE inflation labor market rates policy committee").split()


def random_texts(n, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        words = rng.sample(WORDS, rng.randint(4, 8))
        if rng.random() < 0.15:
            words.insert(0, "Unlike")          # comparative framing
        out.append(" ".join(words).capitalize() + ".")
    return out


def dense(row_emb, col_emb, row_sec, col_sec, row_texts, col_texts, cfg, symmetric):
    """The dense path exactly as detect_contradictions/detect_one_sided run it."""
    sims = row_emb @ col_emb.T
    eligible = [[j for j in range(len(col_emb)) if col_sec[j] != row_sec[i]] for i in range(len(row_emb))]
    total = (sum(1 for i in range(len(row_emb)) for j in range(i + 1, len(row_emb)) if row_sec[i] != row_sec[j])
             if symmetric else sum(len(c) for c in eligible))
    counts = {"comparative": 0, "entity": 0, "topic": 0}
    if cfg.skip_comparative_framing:
        eligible, counts["comparative"] = filter_comparative_pairs(eligible, row_texts, col_texts, symmetric)
    if cfg.require_shared_entity:
        eligible, counts["entity"] = filter_by_shared_entity(eligible, row_texts, col_texts, symmetric)
    if cfg.require_shared_content_word:
        eligible, counts["topic"] = filter_by_shared_content_word(eligible, row_texts, col_texts, symmetric)
    checked, top_k, budget = select_pairs(sims, eligible, cfg, symmetric)
    return {"total": total, "checked": checked, "skipped_top_k": top_k, "skipped_budget": budget,
            "skipped_comparative": counts["comparative"], "skipped_entity": counts["entity"],
            "skipped_topic": counts["topic"]}


def random_unit(n, dim, rng):
    x = rng.standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


CONFIGS = [
    ("contradiction.candidate_top_k=5", "contradiction.max_pairs=0"),
    ("contradiction.candidate_top_k=3", "contradiction.max_pairs=150"),       # budget binds
    ("contradiction.candidate_top_k=0", "contradiction.pair_min_similarity=0.1"),
    ("contradiction.skip_comparative_framing=true", "contradiction.require_shared_content_word=true",
     "contradiction.require_shared_entity=true", "contradiction.candidate_top_k=4"),
]


@pytest.mark.parametrize("overrides", CONFIGS)
@pytest.mark.parametrize("symmetric", [True, False])
def test_chunked_selection_matches_dense(overrides, symmetric):
    rng = np.random.default_rng(3)
    n, m = 300, (300 if symmetric else 700)      # > BLOCK_ROWS, so several blocks
    row_emb = random_unit(n, 16, rng)
    col_emb = row_emb if symmetric else random_unit(m, 16, rng)
    row_sec = rng.integers(0, 25, n)
    col_sec = row_sec if symmetric else rng.integers(0, 25, m)
    row_texts = random_texts(n, 1)
    col_texts = row_texts if symmetric else random_texts(m, 2)
    cfg = make_cfg("contradiction.pair_min_similarity=0.0", *overrides).contradiction
    expected = dense(row_emb, col_emb, row_sec, col_sec, row_texts, col_texts, cfg, symmetric)
    got = select_pairs_chunked(row_emb, col_emb, row_sec, col_sec, row_texts, col_texts, cfg, symmetric)
    assert got["checked"] == expected["checked"]
    for key in ("total", "skipped_top_k", "skipped_budget", "skipped_comparative", "skipped_entity", "skipped_topic"):
        assert got[key] == expected[key], key


def test_detect_contradictions_same_result_either_path(embedder, nli):
    """End to end through Stage 3a with the fake models: identical stats and flags."""
    texts = random_texts(120, 5)
    claims = [texts[i:i + 3] for i in range(0, 120, 3)]
    claims[4].append("Sensor revenue fell because of the plant shutdown.")
    claims[20].append("Sensor revenue fell not because of the plant shutdown.")
    dense_cfg = make_cfg("contradiction.dense_pair_limit=0").contradiction
    chunk_cfg = make_cfg("contradiction.dense_pair_limit=1").contradiction
    assert not use_chunked(122, 122, dense_cfg) and use_chunked(122, 122, chunk_cfg)
    s1, f1 = detect_contradictions(claims, embedder, nli, dense_cfg)
    s2, f2 = detect_contradictions(claims, embedder, nli, chunk_cfg)
    assert s1 == s2
    assert [(c.claim_a, c.claim_b, c.score) for c in f1] == [(c.claim_a, c.claim_b, c.score) for c in f2]
    assert f1, "the planted pair should be flagged"


def test_small_documents_stay_on_the_dense_path(cfg):
    """The 4 samples have at most a few hundred claims and source sentences."""
    assert not use_chunked(400, 1500, cfg.contradiction)
    assert use_chunked(25_000, 25_000, cfg.contradiction)


def test_resolution_budget_scales_with_sections(cfg):
    c = cfg.contradiction
    assert resolution_budget(c, 5) == c.max_resolutions == 200          # small documents: unchanged
    assert resolution_budget(c, 100) == 200
    assert resolution_budget(c, 1000) == 2000                           # 2 per section
    assert resolution_budget(make_cfg("contradiction.max_resolutions=0").contradiction, 1000) == 0  # unlimited


def test_diagnostic_ceiling_is_applied_and_reported(embedder, nli, caplog):
    from hcv_sum.contradiction import diagnose_source_sections
    from conftest import build_index

    texts = random_texts(80, 9)
    spec = [(f"S{i}", texts[i:i + 4]) for i in range(0, 80, 4)]
    index = build_index(spec, embedder)
    cfg = make_cfg("contradiction.diagnostic_max_pairs=25", "contradiction.pair_min_similarity=0.0").contradiction
    with caplog.at_level("WARNING"):
        report = diagnose_source_sections(index, embedder, nli, cfg)
    assert report.pair_stats.checked == 25
    assert report.pair_stats.skipped_budget > 0
    assert "diagnostic_max_pairs=25" in caplog.text and "incomplete" in caplog.text
