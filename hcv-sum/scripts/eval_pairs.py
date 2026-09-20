"""Evaluate Stage 3's contradiction rule and pre-filters on the frozen labelled pair set.

Usage:  python scripts/eval_pairs.py

The label files are verified against the hashes recorded in data/eval/MANIFEST.md before anything
runs; if a label has changed since it was frozen, the script stops. See MANIFEST.md for how the set
was built and its limitations.

Each pair is scored exactly as Stage 3 scores a candidate: P(contradiction) in both directions,
combined with contradiction.direction_aggregation, flagged at contradiction.threshold. Five filter
configurations are compared, all using the same NLI scores:

  raw           NLI rule alone, every pair
  floor+claim   the pipeline's similarity floor and non-claim filter only
  comparison    floor+claim + comparison-wording filter (the default until FINDINGS 10.3)
  entity        floor+claim + shared-entity gate
  topic         floor+claim + shared-content-word gate (FINDINGS 11)
  default       whatever config/default.yaml currently switches on, built from the config flags

Note that "floor+claim" uses the CURRENT non-claim filter, so it includes require_verb_or_number
when that is on.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data" / "eval"
FROZEN = {
    "candidates.jsonl": "7baea4243e822a292dce07a47191ef90a5e000fc6cdae9f357037497754adc13",
    "natural_labels.json": "108eb267970c5d613cec02cb035002aeaa74a5b942f1e95e408d14d3bc8ffcc0",
    "constructed_pairs.jsonl": "2d64a9e043de6191f01a5a2beca3ae5a2f4dec661f9eae12fea640e19b6153da",
}


def verify_frozen() -> None:
    for name, expected in FROZEN.items():
        actual = hashlib.sha256((EVAL / name).read_bytes()).hexdigest()
        if actual != expected:
            sys.exit(f"REFUSING TO RUN: {name} changed since it was frozen (sha256 {actual[:12]}..., "
                     f"expected {expected[:12]}...). Record any correction in a new file; see MANIFEST.md.")


def load_pairs() -> list[dict]:
    labels = json.loads((EVAL / "natural_labels.json").read_text(encoding="utf-8"))
    pairs = []
    for line in (EVAL / "candidates.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            label, category, referent, rationale = labels[row["id"]]
            pairs.append({**row, "label": label, "category": category, "referent": referent,
                          "rationale": rationale})
    for line in (EVAL / "constructed_pairs.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            pairs.append({**json.loads(line), "origin": "constructed"})
    return pairs


def metrics(gold: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, float, float, float]:
    tp = int((gold & pred).sum())
    fp = int((~gold & pred).sum())
    fn = int((gold & ~pred).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return tp, fp, fn, precision, recall, f1


def main() -> int:
    verify_frozen()
    sys.path.insert(0, str(ROOT / "src"))
    from hcv_sum.cli import _quiet_third_party
    _quiet_third_party()
    sys.stdout.reconfigure(encoding="utf-8")
    from hcv_sum.config import load_config
    from hcv_sum.contradiction import AGGREGATIONS, is_claim
    from hcv_sum.models import CONTRADICTION, ModelRegistry
    from hcv_sum.text_utils import comparative_framing, content_tokens, entity_tokens

    pairs = load_pairs()
    cfg = load_config()
    c = cfg.contradiction
    reg = ModelRegistry(cfg)
    a = [p["a"] for p in pairs]
    b = [p["b"] for p in pairs]

    forward = reg.nli.predict(list(zip(a, b)))[:, CONTRADICTION]
    backward = reg.nli.predict(list(zip(b, a)))[:, CONTRADICTION]
    score = AGGREGATIONS[c.direction_aggregation](forward, backward)
    nli_flag = score >= c.threshold
    ea, eb = reg.embedder.encode(a), reg.embedder.encode(b)
    cosine = np.einsum("ij,ij->i", ea, eb)

    gold = np.array([p["label"] == "contradiction" for p in pairs])
    floor = cosine >= c.pair_min_similarity
    claim = np.array([is_claim(x, c) and is_claim(y, c) for x, y in zip(a, b)])
    no_comparison = np.array([not (comparative_framing(x) or comparative_framing(y)) for x, y in zip(a, b)])
    shared_entity = np.array([bool(entity_tokens(x) & entity_tokens(y)) for x, y in zip(a, b)])
    shared_content = np.array([bool(content_tokens(x) & content_tokens(y)) for x, y in zip(a, b)])
    base = nli_flag & floor & claim
    default = base.copy()
    if c.skip_comparative_framing:
        default &= no_comparison
    if c.require_shared_entity:
        default &= shared_entity
    if c.require_shared_content_word:
        default &= shared_content

    configs = {
        "raw": nli_flag,
        "floor+claim": base,
        "comparison": base & no_comparison,
        "entity": base & shared_entity,
        "topic": base & shared_content,
        "default": default,
    }

    n_con, n_not = int(gold.sum()), int((~gold).sum())
    print(f"frozen pair set: {len(pairs)} pairs ({n_con} contradictions, {n_not} non-contradictions); hashes verified")
    print(f"model {cfg.models.nli} | rule: {c.direction_aggregation} of both directions >= {c.threshold}\n")

    print("=== 1. Overall ===")
    print(f"  {'configuration':<13}{'TP':>4}{'FP':>5}{'FN':>5}{'precision':>11}{'recall':>8}{'F1':>7}")
    for name, pred in configs.items():
        tp, fp, fn, pr, rc, f1 = metrics(gold, pred)
        print(f"  {name:<13}{tp:>4}{fp:>5}{fn:>5}{pr:>10.0%}{rc:>9.0%}{f1:>7.2f}")

    print("\n=== 2. By document type (default configuration, then topic gate) ===")
    print(f"  {'type':<12}{'contradictions caught':>24}{'false flags / natural pairs':>30}")
    for kind in ("research", "financial", "contract", "transcript"):
        idx = np.array([p["doc_type"] == kind for p in pairs])
        for name in ("default", "topic"):
            tp, fp, fn, *_ = metrics(gold[idx], configs[name][idx])
            print(f"  {kind:<12}{name:<12}{tp:>4} / {tp + fn:<5}{fp:>18} / {int((~gold[idx]).sum())}")

    print("\n=== 3. Recall by contradiction category ===")
    cats = sorted({p["category"] for p, g in zip(pairs, gold) if g})
    print(f"  {'category':<22}{'n':>3}" + "".join(f"{n:>13}" for n in ("raw", "default", "entity", "topic")))
    for cat in cats:
        idx = np.array([p["category"] == cat and g for p, g in zip(pairs, gold)])
        row = "".join(f"{int(configs[n][idx].sum()):>9} / {int(idx.sum()):<2}" for n in ("raw", "default", "entity", "topic"))
        print(f"  {cat:<22}{int(idx.sum()):>3}{row}")

    print("\n=== 4. THE ENTITY-GATE QUESTION: recall by what the contradiction is about ===")
    for ref in ("named", "common"):
        idx = np.array([p["referent"] == ref and g for p, g in zip(pairs, gold)])
        share = int((shared_entity & idx).sum())
        print(f"  {ref:<7} ({int(idx.sum())} contradictions): share an entity token {share}/{int(idx.sum())} | "
              f"caught raw {int(configs['raw'][idx].sum())}, default {int(configs['default'][idx].sum())}, "
              f"with entity gate {int(configs['entity'][idx].sum())} | share a content word "
              f"{int((shared_content & idx).sum())}/{int(idx.sum())}, with topic gate {int(configs['topic'][idx].sum())}")

    print("\n=== 5. False flags on natural pairs, by kind of look-alike (raw -> default -> entity -> topic) ===")
    kinds = sorted({p["category"] for p, g in zip(pairs, gold) if not g})
    for kind in kinds:
        idx = np.array([p["category"] == kind and not g for p, g in zip(pairs, gold)])
        counts = " -> ".join(str(int(configs[n][idx].sum())) for n in ("raw", "default", "entity", "topic"))
        print(f"  {kind:<20} {int(idx.sum()):>3} pairs, flagged {counts}")

    print("\n=== 6. Every error of the default configuration ===")
    pred = configs["default"]
    for p, g, f, s, fw, bw, cs in zip(pairs, gold, pred, score, forward, backward, cosine):
        if g != f:
            kind = "MISSED" if g else "FALSE FLAG"
            why = []
            if g and not (s >= c.threshold):
                why.append(f"NLI {s:.2f} < {c.threshold}")
            if g and s >= c.threshold:
                why.append("filtered out before NLI")
            print(f"  [{kind}] {p['id']} ({p['category']}) mean {s:.2f} ({fw:.2f}/{bw:.2f}) cos {cs:.2f} {'; '.join(why)}")
            print(f"     A: {p['a'][:130]}")
            print(f"     B: {p['b'][:130]}")

    out = ROOT / "outputs" / "eval_pairs.json"
    out.write_text(json.dumps([{**{k: p[k] for k in ("id", "doc_type", "origin", "label", "category", "referent")},
                                "score": float(s), "forward": float(fw), "backward": float(bw),
                                "cosine": float(cs), "shared_entity": bool(se), "shared_content_word": bool(shared_content[i]),
                                **{f"flag_{n}": bool(v[i]) for n, v in configs.items()}}
                               for i, (p, s, fw, bw, cs, se) in enumerate(zip(pairs, score, forward, backward,
                                                                               cosine, shared_entity))],
                              indent=1), encoding="utf-8")
    print(f"\nper-pair scores written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
