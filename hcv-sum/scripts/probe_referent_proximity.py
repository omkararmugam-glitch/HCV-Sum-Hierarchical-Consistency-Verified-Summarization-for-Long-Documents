"""Can graph proximity separate TRUE contradiction pairs from FALSE flags on sample 05? (measurement only)

Usage:  python scripts/probe_referent_proximity.py     (needs outputs/flags_05_frames_on.json)

A graph-based "are these two claims about the same thing" check would drop pairs that are far apart in
the document's sentence graph. It is only worth building if true pairs are measurably CLOSER than false
ones. Measured on sample 05's current flags (reporting-frame stripping on).

Each claim is grounded to a source sentence (the claim itself if it is a source sentence, otherwise the
most similar source sentence of its own section -- the same grounding as src/hcv_sum/referent.py), then
the two grounding sentences are compared on the kNN-10 sentence graph of the whole document:
  cosine        direct embedding similarity of the two claims (what Stage 3 already filters on)
  hops          unweighted shortest-path length
  path_dist     weighted shortest path, distance = 1 - cosine
  nbr_jaccard   overlap of the two sentences' 10-nearest-neighbour sets ("same neighbourhood")
For each measure: the true pairs' values, the false flags' distribution, and the best case for a filter
-- a threshold set to keep EVERY true pair -- and how many false flags it would then remove.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
K = 10
# STRICT truth: a flag is true only if its two claims ground to the planted sentences themselves. (A loose
# cosine >= 0.6 match mislabelled a false pair -- semiconductor "twelve months" vs automotive "twenty-four
# months" -- as C1.) Discussion has two limitation sentences; both contradict the abstract's C2 claim.
STRICT = {
    "C1": (["The automotive dataset spans eighteen months"], ["automotive dataset, spanning twenty-four months"]),
    "C2": (["transfer effectively across plants with different equipment vendors"],
           ["only evaluated between plants using similar underlying", "We have not yet evaluated transfer between plants"]),
    "C3": (["despite differing equipment vendors and process types"],
           ["only evaluated between plants using similar underlying", "We have not yet evaluated transfer between plants"]),
}
BORDERLINE = ["learned sensor representations generalize meaningfully across manufacturing sites"]
# C3's two sides are both compressed out of the summaries, so it only exists as a source-level pair.
C3 = ("despite differing equipment vendors and process types", "only evaluated between plants using similar underlying equipment vendors")


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from hcv_sum.cli import _quiet_third_party
    _quiet_third_party()
    sys.stdout.reconfigure(encoding="utf-8")
    from analyze_false_flags import RULES
    from hcv_sum.config import load_config
    from hcv_sum.models import ModelRegistry

    emb_model = ModelRegistry(load_config()).embedder
    run = json.loads((ROOT / "outputs" / "new_data_05" / "05_medium_paper_graphfault.json").read_text(encoding="utf-8"))
    flags = json.loads((ROOT / "outputs" / "flags_05_frames_on.json").read_text(encoding="utf-8"))

    sections = run["sections"]
    sents = [s for sec in sections for s in sec["sentences"]]
    offset = {sec["index"]: sec["sentence_offset"] for sec in sections}
    emb = emb_model.encode(sents)
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)
    nbrs = [set(np.argsort(-sims[i])[:K].tolist()) for i in range(len(sents))]
    g = nx.Graph()
    g.add_nodes_from(range(len(sents)))
    for i in range(len(sents)):
        for j in nbrs[i]:
            w = float(sims[i, j])
            g.add_edge(i, j, dist=max(1.0 - w, 1e-6))
    hops = dict(nx.all_pairs_shortest_path_length(g))
    wdist = dict(nx.all_pairs_dijkstra_path_length(g, weight="dist"))

    def ground(claim: dict, is_source: bool) -> int:
        sec = next(s for s in sections if s["index"] == claim["section_index"])
        ids = list(range(offset[sec["index"]], offset[sec["index"]] + len(sec["sentences"])))
        if is_source:
            return offset[sec["index"]] + claim["sentence_index"]
        e = emb_model.encode([claim["text"]])[0]
        return ids[int(np.argmax(emb[ids] @ e))]

    def measures(i: int, j: int, ta: str, tb: str) -> dict:
        ea, eb = emb_model.encode([ta, tb])
        return {"cosine": float(ea @ eb), "hops": hops[i].get(j, np.inf), "path_dist": wdist[i].get(j, np.inf),
                "nbr_jaccard": len(nbrs[i] & nbrs[j]) / len(nbrs[i] | nbrs[j])}

    def ids(keys):
        return {k for k, s in enumerate(sents) if any(key in s for key in keys)}
    strict = {name: (ids(a), ids(b)) for name, (a, b) in STRICT.items()}
    borderline = ids(BORDERLINE)

    rows = []
    for f in flags:
        i = ground(f["claim_a"], False)
        j = ground(f["claim_b"], f["kind"] == "one_sided")
        label = next((name for name, (sa, sb) in strict.items()
                      if (i in sa and j in sb) or (j in sa and i in sb)), None)
        if label is None and (i in borderline or j in borderline):
            continue                                   # borderline pair: in neither set
        pattern = None
        if label is None:
            pattern = next(name for name, rule, _ in RULES if rule(f["claim_a"]["text"], f["claim_b"]["text"]))[:1]
        rows.append({"label": label, "pattern": pattern, **measures(i, j, f["claim_a"]["text"], f["claim_b"]["text"])})
    i3, j3 = (next(k for k, s in enumerate(sents) if key in s) for key in C3)
    if not any(r["label"] == "C3" for r in rows):
        rows.append({"label": "C3 (source pair)", "pattern": None, **measures(i3, j3, sents[i3], sents[j3])})

    true = [r for r in rows if r["label"]]
    false = [r for r in rows if not r["label"]]
    lines = [f"sample 05: {len(true)} true pairs ({', '.join(r['label'] for r in true)}), {len(false)} false flags; "
             f"kNN-{K} graph over {len(sents)} source sentences"]
    # direction: higher cosine / jaccard = closer; lower hops / path_dist = closer
    closer_is_higher = {"cosine": True, "hops": False, "path_dist": False, "nbr_jaccard": True}
    lines.append(f"\n  {'measure':<12}{'true pairs':<38}{'false flags: min / q25 / median / q75 / max':<48}"
                 f"{'keep-all-true filter removes':>30}")
    for m, higher in closer_is_higher.items():
        tv = [r[m] for r in true]
        fv = np.array([r[m] for r in false], dtype=float)
        cut = min(tv) if higher else max(tv)                      # loosest threshold that keeps every true pair
        removed = int((fv < cut).sum()) if higher else int((fv > cut).sum())
        removed_a = sum(1 for r in false if r["pattern"] == "A" and ((r[m] < cut) if higher else (r[m] > cut)))
        n_a = sum(1 for r in false if r["pattern"] == "A")
        q = np.percentile(fv, [0, 25, 50, 75, 100])
        lines.append(f"  {m:<12}{', '.join(f'{v:.2f}' for v in tv):<38}{' / '.join(f'{v:.2f}' for v in q):<48}"
                     f"{removed:>6} / {len(false)} false  ({removed_a}/{n_a} of pattern A)")
    lines.append("\n  true pairs, percentile of each within the false-flag distribution (50 = typical false flag;"
                 " 100 = closer than every false flag):")
    for r in true:
        cells = []
        for m, higher in closer_is_higher.items():
            fv = np.array([x[m] for x in false], dtype=float)
            pct = 100 * ((fv < r[m]).mean() if higher else (fv > r[m]).mean())
            cells.append(f"{m} {r[m]:.2f} (p{pct:.0f})")
        lines.append(f"    {r['label']:<18}" + "   ".join(cells))
    text = "\n".join(lines)
    print(text)
    (ROOT / "outputs" / "probe_referent_proximity.txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
