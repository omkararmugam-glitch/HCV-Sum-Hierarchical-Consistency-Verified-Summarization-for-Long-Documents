"""Memory-bounded Stage 3 candidate selection for large documents.

Stage 3's dense path (``contradiction.select_pairs`` and the ``filter_*`` helpers) materialises a full
row-by-column cosine matrix and a Python list of every cross-section candidate before any pre-filter
runs. For a summary of a few hundred claims that is nothing. For the source-level diagnostic on a
500-page document (~25,000 source sentences) it is a 2.5 GB matrix plus ~600 million Python list
entries, i.e. the run dies before the first NLI call.

This module computes the same selection in row blocks: each block's similarities are computed,
filtered, reduced to the block's top-k candidates and discarded. Peak memory is
O(block_rows x columns + rows x candidate_top_k) instead of O(rows x columns).

It is used only when rows x columns exceeds ``contradiction.dense_pair_limit``; below that the
original dense code runs unchanged, so small documents are unaffected by construction.

Equivalence with the dense path (tests/test_pair_selection.py checks it on random data):
- the same pairs are selected, in the same order, with the same skip counts, EXCEPT that
- for a symmetric comparison the dense path counts an unordered pair as "above the floor" if EITHER
  orientation passes; here only the upper triangle is counted. The two differ only when floating-
  point asymmetry of a matrix product puts sims[i, j] and sims[j, i] on opposite sides of the floor.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

from .config import ContradictionConfig
from .text_utils import comparative_framing, content_tokens, entity_tokens

BLOCK_ROWS = 256


def _token_coords(token_sets: "list[set[str]]", vocab: dict[str, int]) -> "tuple[list[int], list[int]]":
    rows, cols = [], []
    for i, tokens in enumerate(token_sets):
        for t in tokens:
            rows.append(i)
            cols.append(vocab.setdefault(t, len(vocab)))
    return rows, cols


def _incidence(row_sets: "list[set[str]]", col_sets: "list[set[str]]"):
    vocab: dict[str, int] = {}
    r_rows, r_cols = _token_coords(row_sets, vocab)
    c_rows, c_cols = _token_coords(col_sets, vocab)
    shape_v = max(len(vocab), 1)
    r = sparse.csr_matrix((np.ones(len(r_rows), dtype=np.int32), (r_rows, r_cols)), shape=(len(row_sets), shape_v))
    c = sparse.csr_matrix((np.ones(len(c_rows), dtype=np.int32), (c_rows, c_cols)), shape=(len(col_sets), shape_v))
    return r, c.T.tocsc()


def select_pairs_chunked(row_emb: np.ndarray, col_emb: np.ndarray, row_sec: np.ndarray, col_sec: np.ndarray,
                         row_texts: "list[str]", col_texts: "list[str]", cfg: ContradictionConfig,
                         symmetric: bool) -> dict:
    """Return the selection and its accounting, mirroring the dense path's PairStats fields."""
    n, m = len(row_emb), len(col_emb)
    row_sec, col_sec = np.asarray(row_sec), np.asarray(col_sec)
    n_ids = int(max(row_sec.max(initial=0), col_sec.max(initial=0))) + 1
    if symmetric:
        per_section = np.bincount(row_sec, minlength=n_ids)
        total = n * (n - 1) // 2 - int((per_section * (per_section - 1) // 2).sum())
    else:
        total = int(n * m - np.bincount(col_sec, minlength=n_ids)[row_sec].sum())

    row_comp = col_comp = None
    if cfg.skip_comparative_framing:
        row_comp = np.array([comparative_framing(t) is not None for t in row_texts])
        col_comp = row_comp if symmetric else np.array([comparative_framing(t) is not None for t in col_texts])
    entity_mats = topic_mats = None
    if cfg.require_shared_entity:
        rs = [entity_tokens(t) for t in row_texts]
        entity_mats = _incidence(rs, rs if symmetric else [entity_tokens(t) for t in col_texts])
    if cfg.require_shared_content_word:
        rs = [content_tokens(t) for t in row_texts]
        topic_mats = _incidence(rs, rs if symmetric else [content_tokens(t) for t in col_texts])

    skipped = {"comparative": 0, "entity": 0, "topic": 0}
    above_floor = 0
    best: dict[tuple[int, int], tuple[float, int, int]] = {}
    within_top_k: list[tuple[float, int, int]] = []
    k = cfg.candidate_top_k

    for start in range(0, n, BLOCK_ROWS):
        stop = min(start + BLOCK_ROWS, n)
        sims = row_emb[start:stop] @ col_emb.T
        eligible = row_sec[start:stop, None] != col_sec[None, :]
        # Pre-filters in the dense path's order, each counted on what survived the previous one.
        if row_comp is not None:
            ok = ~(row_comp[start:stop, None] | col_comp[None, :])
            skipped["comparative"] += int((eligible & ~ok).sum())
            eligible &= ok
        for name, mats in (("entity", entity_mats), ("topic", topic_mats)):
            if mats is not None:
                ok = (mats[0][start:stop] @ mats[1]).toarray() > 0
                skipped[name] += int((eligible & ~ok).sum())
                eligible &= ok
        candidate = eligible & (sims >= cfg.pair_min_similarity)
        if symmetric:
            upper = np.arange(m)[None, :] > np.arange(start, stop)[:, None]
            above_floor += int((candidate & upper).sum())
        else:
            above_floor += int(candidate.sum())
        for r in range(stop - start):
            i = start + r
            idx = np.flatnonzero(candidate[r])
            if idx.size == 0:
                continue
            vals = sims[r, idx]
            if k > 0 and idx.size > k:
                v = np.partition(vals, idx.size - k)[idx.size - k]      # k-th largest value
                greater = idx[vals > v]
                ties = idx[vals == v][: k - greater.size]               # dense path breaks ties by column order
                chosen = np.concatenate([greater, ties])
                chosen_vals = sims[r, chosen]
                order = np.lexsort((chosen, -chosen_vals))              # dense path: sorted by -sim, stable
                chosen, chosen_vals = chosen[order], chosen_vals[order]
            else:
                chosen, chosen_vals = idx, vals
            for j, s in zip(chosen.tolist(), chosen_vals.tolist()):
                if symmetric:
                    key = (i, j) if i < j else (j, i)
                    if s > best.get(key, (-2.0, 0, 0))[0]:
                        best[key] = (s, i, j)
                else:
                    within_top_k.append((s, i, j))

    scored = list(best.values()) if symmetric else within_top_k
    skipped_top_k = above_floor - len(scored)
    skipped_budget = 0
    if cfg.max_pairs > 0 and len(scored) > cfg.max_pairs:
        scored.sort(key=lambda t: -t[0])
        skipped_budget = len(scored) - cfg.max_pairs
        scored = scored[: cfg.max_pairs]
    halve = (lambda x: x // 2) if symmetric else (lambda x: x)
    return {
        "total": total,
        "checked": [(i, j) for _, i, j in scored],
        "skipped_top_k": max(skipped_top_k, 0),
        "skipped_budget": skipped_budget,
        "skipped_entity": halve(skipped["entity"]),
        "skipped_comparative": halve(skipped["comparative"]),
        "skipped_topic": halve(skipped["topic"]),
    }
