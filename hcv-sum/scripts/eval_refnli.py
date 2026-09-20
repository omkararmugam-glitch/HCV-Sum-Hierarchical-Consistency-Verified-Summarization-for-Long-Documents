"""Evaluate our NLI model and Stage 3 decision rule on RefNLI (Chen et al., NAACL Findings 2025).

Usage:  python scripts/eval_refnli.py [--data data/external/refnli.jsonl]

RefNLI has 1,143 (premise, hypothesis) pairs where the premise was retrieved from Wikipedia and need
not refer to the same context as the hypothesis. Labels: entailment, contradiction, neutral (no
support or conflict under ANY reading), ambiguous (depends on which referent you assume).

The question this answers: how often does OUR contradiction rule fire on pairs that are not
contradictions, especially pairs that refer to different contexts? Four documents could not answer
that; 1,143 pairs can. Evaluation only -- no pipeline code is changed or called beyond the models.

The rule evaluated is exactly Stage 3's: P(contradiction) in both directions, combined with
``contradiction.direction_aggregation``, flagged at ``contradiction.threshold``. It is reported
twice: on every pair (the raw model), and after the pre-filters the pipeline applies before NLI ever
runs (similarity floor, comparative framing, non-claim filter), which is the realistic figure.

CAVEATS, printed with the results:
- Our NLI model was trained on FEVER-NLI, and most RefNLI pairs come from FEVER. Results are split by
  source (FEVER vs VitaminC) so contamination can be seen rather than averaged away.
- RefNLI pairs are Wikipedia sentence vs claim, not two sentences from the same document. The
  distribution differs from Stage 3's input, so this measures the failure mode, not our exact rate.

The data file is downloaded from the authors' repository and is not redistributed here (the
repository states no license).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "external" / "refnli.jsonl"
SOURCE_URL = "https://raw.githubusercontent.com/refnli-authors/refnli/main/refnli.jsonl"
LABELS = ("contradiction", "entailment", "neutral", "ambiguous")

# Figures reported by Chen et al. (2025) for comparison (Table 3, T5-Large trained on five NLI datasets).
PAPER_T5_CONTRADICTION_PRECISION = 15.76
PAPER_T5_CONTRADICTION_RECALL = 92.42


def load(path: Path) -> list[dict]:
    if not path.exists():
        print(f"downloading RefNLI from {SOURCE_URL}")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE_URL, path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def jaccard(a: str, b: str, stopwords) -> float:
    ta = {w for w in re.findall(r"[a-z0-9]+", a.lower()) if w not in stopwords and len(w) > 2}
    tb = {w for w in re.findall(r"[a-z0-9]+", b.lower()) if w not in stopwords and len(w) > 2}
    return len(ta & tb) / max(len(ta | tb), 1)


def report(title: str, rows: list[dict], flagged: np.ndarray) -> None:
    by_label = defaultdict(lambda: [0, 0])          # label -> [flagged, total]
    for r, f in zip(rows, flagged):
        by_label[r["label"]][0] += int(f)
        by_label[r["label"]][1] += 1
    tp = by_label["contradiction"][0]
    fp = sum(v[0] for k, v in by_label.items() if k != "contradiction")
    gold = by_label["contradiction"][1]
    precision = 100 * tp / max(tp + fp, 1)
    recall = 100 * tp / max(gold, 1)
    print(f"\n  {title}")
    print(f"    {'gold label':<14}{'flagged as contradiction':>28}")
    for label in LABELS:
        f, n = by_label[label]
        if n:
            print(f"    {label:<14}{f:>12} / {n:<6} = {100 * f / n:5.1f}%"
                  + ("   <- recall" if label == "contradiction" else "   <- false contradiction rate"))
    print(f"    contradiction precision {precision:5.1f}%   recall {recall:5.1f}%   "
          f"({tp} true, {fp} false flags)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from hcv_sum.cli import _quiet_third_party
    _quiet_third_party()
    sys.stdout.reconfigure(encoding="utf-8")
    from hcv_sum.config import load_config
    from hcv_sum.contradiction import AGGREGATIONS, is_claim
    from hcv_sum.models import CONTRADICTION, ModelRegistry
    from hcv_sum.text_utils import STOPWORDS, comparative_framing, content_tokens, entity_tokens

    rows = load(args.data)
    cfg = load_config()
    c = cfg.contradiction
    reg = ModelRegistry(cfg)

    premises = [r["premise"] for r in rows]
    hypotheses = [r["hypothesis"] for r in rows]
    print(f"RefNLI: {len(rows)} pairs, labels {dict(Counter(r['label'] for r in rows))}")
    print(f"model: {cfg.models.nli} | rule: {c.direction_aggregation} of both directions >= {c.threshold}")

    forward = reg.nli.predict(list(zip(premises, hypotheses)))[:, CONTRADICTION]
    backward = reg.nli.predict(list(zip(hypotheses, premises)))[:, CONTRADICTION]
    score = AGGREGATIONS[c.direction_aggregation](forward, backward)
    raw_flag = score >= c.threshold

    emb_p = reg.embedder.encode(premises)
    emb_h = reg.embedder.encode(hypotheses)
    cosine = np.einsum("ij,ij->i", emb_p, emb_h)

    # What the pipeline would let through to NLI in the first place.
    passes_floor = cosine >= c.pair_min_similarity
    passes_claims = np.array([is_claim(p, c) and is_claim(h, c) for p, h in zip(premises, hypotheses)])
    passes_framing = np.array([not (comparative_framing(p) or comparative_framing(h))
                               for p, h in zip(premises, hypotheses)]) if c.skip_comparative_framing \
        else np.ones(len(rows), dtype=bool)
    reaches_nli = passes_floor & passes_claims & passes_framing
    pipeline_flag = raw_flag & reaches_nli

    print("\n=== 1. Raw NLI rule on every pair ===")
    report("all sources", rows, raw_flag)

    print("\n=== 2. As the pipeline would run it (after its own pre-filters) ===")
    print(f"  pairs reaching NLI: {int(reaches_nli.sum())} of {len(rows)} "
          f"(similarity floor {c.pair_min_similarity} drops {int((~passes_floor).sum())}, "
          f"non-claim filter {int((~passes_claims).sum())}, comparative framing {int((~passes_framing).sum())}; "
          "overlaps counted in each)")
    report("all sources, pipeline pre-filters applied", rows, pipeline_flag)
    report("similarity floor ALONE", rows, raw_flag & passes_floor)
    report("comparative-framing filter ALONE", rows, raw_flag & passes_framing)

    print("\n=== 3. Split by source (our NLI model was trained on FEVER-NLI) ===")
    for family in ("fever", "vc"):
        idx = [i for i, r in enumerate(rows) if r["source"].startswith(family)]
        name = "FEVER (possible training overlap)" if family == "fever" else "VitaminC (not in our model's training data)"
        report(f"{name}: {len(idx)} pairs, raw rule", [rows[i] for i in idx], raw_flag[idx])

    print("\n=== 4. Candidate reference gates, measured here instead of on 4 documents ===")
    shared_entity = np.array([bool(entity_tokens(p) & entity_tokens(h)) for p, h in zip(premises, hypotheses)])
    overlap = np.array([jaccard(p, h, STOPWORDS) > 0.15 for p, h in zip(premises, hypotheses)])
    report("raw rule + shared-entity gate (require_shared_entity)", rows, raw_flag & shared_entity)
    report("raw rule + word-overlap gate (Jaccard > 0.15, the RefNLI authors' heuristic)", rows, raw_flag & overlap)
    shared_content = np.array([bool(content_tokens(p) & content_tokens(h)) for p, h in zip(premises, hypotheses)])
    report("raw rule + shared-content-word gate (require_shared_content_word)", rows, raw_flag & shared_content)
    report("pipeline pre-filters + shared-content-word gate", rows, pipeline_flag & shared_content)

    print("\n=== Reference point from the paper ===")
    print(f"  Chen et al. (2025), T5-Large on five NLI datasets: contradiction precision "
          f"{PAPER_T5_CONTRADICTION_PRECISION}%, recall {PAPER_T5_CONTRADICTION_RECALL}%")
    print("  Not directly comparable: different model, and the paper's exact evaluation protocol "
          "was not reproduced here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
