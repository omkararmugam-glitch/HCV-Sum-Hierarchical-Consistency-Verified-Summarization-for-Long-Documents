"""Why did sample 05 produce 95 flags? Pair count vs per-pair rate, and what the false flags are.

Usage:  python scripts/analyze_false_flags.py

Inputs: outputs/new_data_05/ (sample 05) and outputs/cur_small/ (samples 01-04), both run with the
same, current default config and no source diagnostic.

1. Per document: pairs compared (3a + 3b), flags, true flags, false flags, false flags per 100 pairs.
2. The same rate within cosine-similarity bands, so a document with more near-duplicate-looking pairs
   is not mistaken for one with a higher per-pair error rate. Needs the list of COMPARED pairs, which
   the run does not store: it is re-derived by re-running Stage 3's selection with an NLI stub that
   records what it is asked (selection is deterministic; nothing is re-scored).
3. Sample 05's false flags assigned to patterns by EXPLICIT rules (printed below, so the counts can be
   audited), with what each existing pre-filter would do to each pattern and to the true flags.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MATCH = 0.60          # same tolerance as score_samples.py
BANDS = [(0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 1.01)]

# Pattern rules for sample 05, applied in this order (first match wins).
RULES = [
    ("A referent lost in Stage 2 compression",
     lambda a, b: any(t.startswith("The network uses a fixed") for t in (a, b)),
     "one side is Stage 2's summary 'The network uses a fixed, manually specified sensor adjacency matrix' -- "
     "the source says 'The THIRD [baseline] is a static graph attention network that uses ...'"),
    ("B method vs its own ablation",
     lambda a, b: any(t.startswith("Removing the") for t in (a, b)),
     "one side is an ablation result ('Removing the ... module ... reduces lead time')"),
    ("C method vs a baseline / prior work",
     lambda a, b: any(re.match(r"(The (first|second|third|fourth) is|We compare GraphFault against|These methods|"
                               r"Statistical process control|Autoencoder-based|Traditional threshold|"
                               r"Graph neural networks have gained|However, most prior)", t) for t in (a, b)),
     "one side describes a baseline or prior work, the other the proposed method"),
    ("D different dataset / quantity, same numeric structure",
     lambda a, b: all(re.search(r"\b(automotive|semiconductor|chemical|dataset|hours|months|sensors|percent)\b", t)
                      for t in (a, b)),
     "both sides state a figure about a dataset, metric or setting, but about DIFFERENT ones"),
    ("E other: different parts of the method or results",
     lambda a, b: True, "two statements about different aspects that do not conflict"),
]


class RecordingNLI:
    def __init__(self):
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs):
        self.pairs.extend(pairs)
        out = np.zeros((len(pairs), 3), dtype=np.float32)
        out[:, 2] = 1.0
        return out


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from hcv_sum.cli import _quiet_third_party
    _quiet_third_party()
    sys.stdout.reconfigure(encoding="utf-8")
    from hcv_sum.config import load_config
    from hcv_sum.contradiction import detect_contradictions, detect_one_sided, is_claim
    from hcv_sum.models import ModelRegistry
    from hcv_sum.scoring import DocumentIndex
    from hcv_sum.text_utils import comparative_framing, content_tokens, entity_tokens, strip_speaker_label
    from hcv_sum.types import Section

    cfg = load_config()
    emb = ModelRegistry(cfg).embedder
    truth = json.loads((ROOT / "data" / "samples" / "ground_truth.json").read_text(encoding="utf-8"))
    runs = sorted((ROOT / "outputs" / "cur_small").glob("0[1-4]_*.json")) + \
        [ROOT / "outputs" / "new_data_05" / "05_medium_paper_graphfault.json"]

    def is_true(flag, doc_stem, src):
        planted = truth.get(doc_stem, {}).get("contradictions", [])
        if not planted:
            return False
        fa, fb = emb.encode([flag["claim_a"]["text"], flag["claim_b"]["text"]])
        for ka, kb in planted:
            a = next((s for s in src if ka in s), None)
            b = next((s for s in src if kb in s), None)
            if a is None or b is None:
                continue
            ea, eb = emb.encode([a, b])
            if max(min(fa @ ea, fb @ eb), min(fa @ eb, fb @ ea)) >= MATCH:
                return True
        return False

    print("=== 1. Pair count vs per-pair false-flag rate (current default config, same for all documents) ===")
    print(f"  {'document':<32}{'sections':>9}{'claims':>7}{'pairs 3a+3b':>13}{'flags':>7}{'true':>6}{'false':>7}"
          f"{'false / 100 pairs':>19}")
    band_stats: dict[str, dict] = {}
    all_flags: dict[str, list] = {}
    for path in runs:
        d = json.loads(path.read_text(encoding="utf-8"))
        stem = path.stem
        src = [s for sec in d["sections"] for s in sec["sentences"]]
        flags = d["contradictions"]["contradictions"] + d["contradictions"]["one_sided"]
        labels = [is_true(f, stem, src) for f in flags]
        s = d["stats"]
        pairs = s["stage3a_pairs"]["compared"] + s["stage3b_pairs"]["compared"]
        fp = sum(not x for x in labels)
        print(f"  {stem:<32}{s['sections']:>9}{s['summary_claims']:>7}{pairs:>13}{len(flags):>7}{sum(labels):>6}"
              f"{fp:>7}{100 * fp / max(pairs, 1):>19.1f}")
        all_flags[stem] = list(zip(flags, labels))

        # re-derive the compared pairs (selection only) to get their cosine distribution
        sections = [Section(x["index"], x["title"], x["sentences"], x["sentence_offset"], x["origin"])
                    for x in d["sections"]]
        index = DocumentIndex.build(sections, emb)
        claims = [ss["sentences"] for ss in d["section_summaries"]]
        rec = RecordingNLI()
        detect_contradictions(claims, emb, rec, cfg.contradiction)
        n3a = len(rec.pairs) // 2
        compared = rec.pairs[:n3a]
        rec2 = RecordingNLI()
        detect_one_sided(claims, set(), index, emb, rec2, cfg.contradiction)
        compared += rec2.pairs[: len(rec2.pairs) // 2]
        cos = np.array([float(x @ y) for x, y in zip(emb.encode([a for a, _ in compared]),
                                                     emb.encode([b for _, b in compared]))]) if compared else np.zeros(0)
        fcos = np.array([float(x @ y) for x, y in zip(
            emb.encode([strip_speaker_label(f["claim_a"]["text"]) for f, _ in all_flags[stem]]),
            emb.encode([strip_speaker_label(f["claim_b"]["text"]) for f, _ in all_flags[stem]]))]) \
            if flags else np.zeros(0)
        fpmask = np.array([not t for _, t in all_flags[stem]], dtype=bool)
        band_stats[stem] = {"cos": cos, "fcos": fcos[fpmask] if flags else fcos, "reselected": len(compared),
                            "reported": pairs}

    print("\n  (re-derived compared pairs vs the run's own count: "
          + ", ".join(f"{k[:2]} {v['reselected']}/{v['reported']}" for k, v in band_stats.items()) + ")")

    print("\n=== 2. False flags per 100 compared pairs, WITHIN cosine bands ===")
    header = "".join(f"{f'{lo:.1f}-{min(hi, 1):.1f}':>16}" for lo, hi in BANDS)
    print(f"  {'document':<32}{header}")
    pooled = defaultdict(lambda: [0, 0])
    for stem, st in band_stats.items():
        cells = []
        for lo, hi in BANDS:
            n = int(((st["cos"] >= lo) & (st["cos"] < hi)).sum())
            f = int(((st["fcos"] >= lo) & (st["fcos"] < hi)).sum())
            key = "05" if stem.startswith("05") else "01-04"
            pooled[(key, lo)][0] += f
            pooled[(key, lo)][1] += n
            cells.append(f"{f:>3}/{n:<4}={100 * f / n if n else 0:>4.0f}%" if n else f"{'-':>16}")
        print(f"  {stem:<32}" + "".join(f"{c:>16}" for c in cells))
    for key in ("01-04", "05"):
        cells = []
        for lo, _ in BANDS:
            f, n = pooled[(key, lo)]
            cells.append(f"{f:>3}/{n:<4}={100 * f / n if n else 0:>4.0f}%" if n else f"{'-':>16}")
        print(f"  {'POOLED ' + key:<32}" + "".join(f"{c:>16}" for c in cells))
    for key, stems in (("01-04", [k for k in band_stats if not k.startswith("05")]), ("05", [k for k in band_stats if k.startswith("05")])):
        cos = np.concatenate([band_stats[k]["cos"] for k in stems])
        print(f"  share of compared pairs with cosine >= 0.5, {key}: {(cos >= 0.5).mean():.0%} (median cosine {np.median(cos):.2f})")

    print("\n=== 3. Sample 05's flags by pattern (rules applied in order, first match wins) ===")
    for name, _, desc in RULES:
        print(f"  {name}: {desc}")
    flags05 = all_flags["05_medium_paper_graphfault"]
    groups: dict[str, list] = defaultdict(list)
    for f, is_t in flags05:
        a, b = f["claim_a"]["text"], f["claim_b"]["text"]
        if is_t:
            groups["TRUE (C2)"].append(f)
            continue
        for name, rule, _ in RULES:
            if rule(a, b):
                groups[name].append(f)
                break
    print(f"\n  {'pattern':<52}{'flags':>6}{'mean P':>8}{'comparison':>12}{'entity':>8}{'topic':>7}"
          f"   (flags that each EXISTING gate would remove)")
    for name in ["TRUE (C2)"] + [r[0] for r in RULES]:
        fl = groups.get(name, [])
        if not fl:
            continue
        comp = sum(bool(comparative_framing(f["claim_a"]["text"]) or comparative_framing(f["claim_b"]["text"])) for f in fl)
        ent = sum(not (entity_tokens(f["claim_a"]["text"]) & entity_tokens(f["claim_b"]["text"])) for f in fl)
        top = sum(not (content_tokens(f["claim_a"]["text"]) & content_tokens(f["claim_b"]["text"])) for f in fl)
        print(f"  {name:<52}{len(fl):>6}{np.mean([f['score'] for f in fl]):>8.2f}{comp:>12}{ent:>8}{top:>7}")
    print("\n  Examples (2-3 per pattern, highest score first):")
    for name in [r[0] for r in RULES]:
        fl = sorted(groups.get(name, []), key=lambda f: -f["score"])
        if not fl:
            continue
        print(f"\n  [{name}]")
        for f in fl[:3]:
            print(f"    P={f['score']:.3f} ({f['score_ab']:.2f}/{f['score_ba']:.2f}) {f['kind']}")
            print(f"      A: {f['claim_a']['text']}")
            print(f"      B: {f['claim_b']['text']}")
    claims05 = [x for x in json.loads(runs[-1].read_text(encoding="utf-8"))["section_summaries"]]
    print(f"\n  non-claims skipped in 05: {sum(not is_claim(t, cfg.contradiction) for c in claims05 for t in c['sentences'])}")
    counts = Counter(name for name, fl in groups.items() for _ in fl)
    (ROOT / "outputs" / "false_flags_05.json").write_text(json.dumps(
        {name: [{"a": f["claim_a"]["text"], "b": f["claim_b"]["text"], "score": f["score"], "kind": f["kind"]}
                for f in fl] for name, fl in groups.items()}, indent=1), encoding="utf-8")
    print(f"\n  totals: {dict(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
