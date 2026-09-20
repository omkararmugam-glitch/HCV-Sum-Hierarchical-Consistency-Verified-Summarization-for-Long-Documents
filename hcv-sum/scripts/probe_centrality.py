"""IDEAS.md idea 1, offline check: do contradiction-carrying sentences rank high by graph centrality?

Usage:  python scripts/probe_centrality.py        (writes outputs/probe_centrality.txt)

Backbone protection (a cheap stand-in for GloSA-sum's protection pool) would re-attach the most central
source sentences. It can only help recall if the sentences that carry known contradictions ARE central.
Bar stated in IDEAS.md before this ran: they need to rank in about the top 20% (80th percentile or above).

Sentences: the Stage 1 source sentences of the saved runs (outputs/cur_small/03..., outputs/new_data_05/05...),
embedded with the pipeline's embedder. Graphs (several, so the answer does not hinge on one choice):
  kNN-k   each sentence linked to its k most similar sentences (k = 5, 10, 20), symmetrised, weight = cosine
  dense   every pair with cosine >= 0.1 (LexRank-style)
Centrality: weighted PageRank (hubs), and betweenness with distance = 1 - cosine (bridges).
Percentile = share of the document's sentences with a LOWER score (100 = most central).
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

TARGETS = {
    "03_subtle_contradiction": ("cur_small", [
        ("planted A: 'every enterprise customer ... Helix Core'", "Every enterprise customer is now running on the new Helix Core platform"),
        ("planted B: 'roughly sixty enterprise accounts ...'", "roughly sixty enterprise accounts are still being served from the legacy environment"),
    ]),
    "05_medium_paper_graphfault": ("new_data_05", [
        ("C2 A: abstract 'different equipment vendors'", "transfer effectively across plants with different equipment vendors"),
        ("C3 A: 5.3 'despite differing equipment vendors'", "despite differing equipment vendors and process types"),
        ("C2/C3 B: 6 'only evaluated ... similar vendors'", "only evaluated between plants using similar underlying equipment vendors"),
        ("C2 B': 6 'not yet evaluated ... different equipment'", "We have not yet evaluated transfer between plants using substantially different"),
        ("C1 A: 4.1 'eighteen months'", "The automotive dataset spans eighteen months"),
        ("C1 B: 6 'twenty-four months'", "automotive dataset, spanning twenty-four months"),
    ]),
}


def graphs(emb: np.ndarray) -> dict[str, nx.Graph]:
    from hcv_sum.centrality import dense_graph, knn_graph   # shared with multi-document salience
    out = {f"kNN-{k}": knn_graph(emb, k) for k in (5, 10, 20)}
    out["dense>=0.1"] = dense_graph(emb, 0.1)
    return out


def percentile(scores: dict[int, float], node: int) -> float:
    vals = np.array(list(scores.values()))
    return 100.0 * float((vals < scores[node]).mean())


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from hcv_sum.cli import _quiet_third_party
    _quiet_third_party()
    sys.stdout.reconfigure(encoding="utf-8")
    import json
    from hcv_sum.config import load_config
    from hcv_sum.models import ModelRegistry

    emb_model = ModelRegistry(load_config()).embedder
    lines = []
    summary = []
    for stem, (folder, targets) in TARGETS.items():
        d = json.loads((ROOT / "outputs" / folder / f"{stem}.json").read_text(encoding="utf-8"))
        sents = [s for sec in d["sections"] for s in sec["sentences"]]
        emb = emb_model.encode(sents)
        gs = graphs(emb)
        cent = {}
        from hcv_sum.centrality import centrality
        for name, g in gs.items():
            cent[(name, "PageRank")] = centrality(g, "pagerank")
            cent[(name, "betweenness")] = centrality(g, "betweenness")
        lines.append(f"\n=== {stem}: {len(sents)} source sentences ===")
        header = "".join(f"{f'{g} {m[:4]}':>16}" for g, m in cent)
        lines.append(f"  {'sentence (percentile, 100 = most central)':<52}{header}")
        for label, key in targets:
            node = next(i for i, s in enumerate(sents) if key in s)
            pcts = [percentile(scores, node) for scores in cent.values()]
            summary.append((stem, label, max(pcts), np.median(pcts)))
            lines.append(f"  {label:<52}" + "".join(f"{p:>16.0f}" for p in pcts))
    lines.append("\n=== against the bar (top ~20%, i.e. percentile >= 80, in the graph where the sentence does best) ===")
    for stem, label, best, med in summary:
        lines.append(f"  {stem[:2]} {label:<52} best {best:>4.0f}  median {med:>4.0f}  -> "
                     f"{'WOULD be protected' if best >= 80 else 'not protected'}")
    text = "\n".join(lines)
    print(text)
    (ROOT / "outputs" / "probe_centrality.txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
