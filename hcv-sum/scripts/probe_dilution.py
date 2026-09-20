"""Controlled probe: why does an EXPLICIT numeric contradiction score low when wrapped in extra text?

Usage:  python scripts/probe_dilution.py            (writes outputs/probe_dilution.txt and .json)

Found on sample 05: "automotive dataset spans eighteen months" (Section 4.1) vs "the automotive dataset,
spanning twenty-four months of continuous stamping line operation, provided ... since longer
observation windows ..." (Discussion) scores 0.151, although the bare conflict scores 0.99.

Four candidate mechanisms, each with a variant designed to separate it from the others:

  LENGTH        more tokens, whatever they say, hide the conflict.
                -> 'redundant pad': the same fact restated to add tokens but NO new proposition.
  UNSHARED      a proposition present in one sentence but not the other pushes the model to
  CONTENT       'neutral' (NLI is trained so that unverifiable hypothesis content = neutral).
                -> 'clause on B only' / 'clause on A only' vs 'SAME clause on both sides'
                   (both longer, but the non-number content is matched, so nothing is unverifiable).
  BACKGROUNDING the number sits in a participle/relative clause, not the main assertion, so the
                sentence "is about" something else.
                -> 'backgrounded' vs 'asserted' with the same words.
  DIRECTION     the pipeline averages P(contradiction) over both directions (mean aggregation);
                one direction may collapse while the other holds.
                -> every variant reports both directions and the mean.

Each variant changes ONE thing relative to the bare pair or to the previous rung.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CLAUSE = "longer observation windows make gradual equipment aging easier to distinguish from short-term noise"

PAPER = [
    # (label, A, B) -- A = Section 4.1 side, B = Discussion side
    ("1 as written in the document",
     "The automotive dataset spans eighteen months of sensor readings from a stamping line, including two hundred and forty individual sensors.",
     "We also note that the automotive dataset, spanning twenty-four months of continuous stamping line operation, provided a particularly stable basis for evaluating long-term drift in sensor correlation patterns, since longer observation windows make gradual equipment aging easier to distinguish from short-term noise."),
    ("2 B: trailing 'since longer observation windows...' removed",
     "The automotive dataset spans eighteen months of sensor readings from a stamping line, including two hundred and forty individual sensors.",
     "We also note that the automotive dataset, spanning twenty-four months of continuous stamping line operation, provided a particularly stable basis for evaluating long-term drift in sensor correlation patterns."),
    ("3 B: leading 'We also note that' removed",
     "The automotive dataset spans eighteen months of sensor readings from a stamping line, including two hundred and forty individual sensors.",
     "The automotive dataset, spanning twenty-four months of continuous stamping line operation, provided a particularly stable basis for evaluating long-term drift in sensor correlation patterns."),
    ("4 B: number moved into the main clause, 'provided...' predicate dropped",
     "The automotive dataset spans eighteen months of sensor readings from a stamping line, including two hundred and forty individual sensors.",
     "The automotive dataset spans twenty-four months of continuous stamping line operation."),
    ("5 B: bare",
     "The automotive dataset spans eighteen months of sensor readings from a stamping line, including two hundred and forty individual sensors.",
     "The automotive dataset spans twenty-four months."),
    ("6 A: ', including two hundred and forty individual sensors' removed (B bare)",
     "The automotive dataset spans eighteen months of sensor readings from a stamping line.",
     "The automotive dataset spans twenty-four months."),
    ("7 both bare",
     "The automotive dataset spans eighteen months.",
     "The automotive dataset spans twenty-four months."),
]

BARE_A, BARE_B = "The automotive dataset spans eighteen months.", "The automotive dataset spans twenty-four months."
PAPER_CONTROLS = [
    ("C1 bare (reference)", BARE_A, BARE_B),
    ("C2 LENGTH: redundant pad on B (no new proposition)", BARE_A,
     "The automotive dataset spans twenty-four months, which is to say that the automotive dataset spans a period of twenty-four months in total."),
    ("C3 UNSHARED: clause on B only", BARE_A, f"The automotive dataset spans twenty-four months, and {CLAUSE}."),
    ("C4 UNSHARED: clause on A only", f"The automotive dataset spans eighteen months, and {CLAUSE}.", BARE_B),
    ("C5 SHARED: same clause on both sides", f"The automotive dataset spans eighteen months, and {CLAUSE}.",
     f"The automotive dataset spans twenty-four months, and {CLAUSE}."),
    ("C6 BACKGROUNDED number, B only", BARE_A,
     "The automotive dataset, spanning twenty-four months, provided a stable basis for evaluating drift."),
    ("C7 ASSERTED number, same words, B only", BARE_A,
     "The automotive dataset spans twenty-four months and provided a stable basis for evaluating drift."),
    ("C8 BACKGROUNDED number, both sides (matched)",
     "The automotive dataset, spanning eighteen months, provided a stable basis for evaluating drift.",
     "The automotive dataset, spanning twenty-four months, provided a stable basis for evaluating drift."),
]

# Is it content ATTACHED TO THE QUANTITY ("months of sensor readings") that breaks the comparison, or any
# unshared content? Hypothesis side is A in the B->A direction, so read that column.
PAPER_QUALIFIER = [
    ("Q0 bare", BARE_A, BARE_B),
    ("Q1 A: qualifier on the quantity 'of sensor readings'",
     "The automotive dataset spans eighteen months of sensor readings.", BARE_B),
    ("Q2 A: 'of sensor readings from a stamping line'",
     "The automotive dataset spans eighteen months of sensor readings from a stamping line.", BARE_B),
    ("Q3 A: same content as a SEPARATE clause (not on the quantity)",
     "The automotive dataset spans eighteen months and contains sensor readings from a stamping line.", BARE_B),
    ("Q4 both: SAME qualifier on the quantity",
     "The automotive dataset spans eighteen months of sensor readings from a stamping line.",
     "The automotive dataset spans twenty-four months of sensor readings from a stamping line."),
    ("Q5 DIFFERENT qualifiers ('of sensor readings' vs 'of continuous operation')",
     "The automotive dataset spans eighteen months of sensor readings.",
     "The automotive dataset spans twenty-four months of continuous operation."),
    ("Q6 B: introducing frame 'We also note that' only", BARE_A,
     "We also note that the automotive dataset spans twenty-four months."),
]

# (name, subject, A predicate, B predicate, qualifier attached to the quantity, same content as a clause)
QUALIFIER_FACTS = [
    ("pilot", "The clinical pilot", "enrolled 120 patients", "enrolled 85 patients",
     "from the northern clinics", "recruited them from the northern clinics"),
    ("warehouse", "The regional warehouse", "holds 4,000 pallets", "holds 3,200 pallets",
     "of dry goods", "stores dry goods"),
    ("contract", "The service contract", "runs for three years", "runs for five years",
     "of on-site support", "covers on-site support"),
]

# Synthetic facts: (name, subject, A value predicate, B value predicate)
FACTS = [
    ("pilot", "The clinical pilot", "enrolled 120 patients", "enrolled 85 patients"),
    ("warehouse", "The regional warehouse", "holds 4,000 pallets", "holds 3,200 pallets"),
    ("contract", "The service contract", "runs for three years", "runs for five years"),
]
FRAMES = {
    "causal": ("Because regional demand fell sharply after the merger announcement, {s_lower} {p}.",
               "because regional demand fell sharply after the merger announcement"),
    "hedge": ("We believe, based on preliminary conversations with the site coordinators, that {s_lower} {p}.",
              "we believe based on preliminary conversations with the site coordinators"),
    "parenthetical": ("{s} (a figure that surprised several members of the review board) {p}.",
                      "a figure that surprised several members of the review board"),
    "trailing": ("{s} {p}, which made later scheduling considerably easier for the operations team.",
                 "which made later scheduling considerably easier for the operations team"),
}


def synthetic_variants(subject: str, pa: str, pb: str) -> list[tuple[str, str, str]]:
    s_lower = subject[0].lower() + subject[1:]
    bare_a, bare_b = f"{subject} {pa}.", f"{subject} {pb}."
    out = [("bare", bare_a, bare_b),
           ("LENGTH redundant pad on B", bare_a,
            f"{subject} {pb}, which is to say that {s_lower} {pb} in total, as recorded.")]
    for name, (template, _) in FRAMES.items():
        fa = template.format(s=subject, s_lower=s_lower, p=pa)
        fb = template.format(s=subject, s_lower=s_lower, p=pb)
        out += [(f"{name}: frame on B only", bare_a, fb),
                (f"{name}: frame on A only", fa, bare_b),
                (f"{name}: SAME frame on both", fa, fb)]
    out += [("BACKGROUNDED number, B only", bare_a,
             f"{subject}, which {pb}, was praised by the review board for its careful design."),
            ("ASSERTED number, same words, B only", bare_a,
             f"{subject} {pb} and was praised by the review board for its careful design.")]
    return out


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from hcv_sum.cli import _quiet_third_party
    _quiet_third_party()
    sys.stdout.reconfigure(encoding="utf-8")
    from hcv_sum.config import load_config
    from hcv_sum.models import CONTRADICTION, ENTAILMENT, NEUTRAL, ModelRegistry

    cfg = load_config()
    reg = ModelRegistry(cfg)
    tok = reg.nli.model.tokenizer
    threshold = cfg.contradiction.threshold
    lines, rows = [], []

    def score_block(title: str, variants: list[tuple[str, str, str]]) -> None:
        pairs = [(a, b) for _, a, b in variants] + [(b, a) for _, a, b in variants]
        probs = reg.nli.predict(pairs)
        n = len(variants)
        lines.append(f"\n=== {title} ===")
        lines.append(f"  {'variant':<66}{'tokens A/B':>11}{'A->B':>7}{'B->A':>7}{'mean':>7}{'max':>6}   "
                     f"B->A label (C/E/N)")
        for k, (label, a, b) in enumerate(variants):
            ab, ba = probs[k], probs[n + k]
            mean = (ab[CONTRADICTION] + ba[CONTRADICTION]) / 2
            ta, tb = len(tok(a)["input_ids"]), len(tok(b)["input_ids"])
            flag = "FLAG" if mean >= threshold else "    "
            lines.append(f"  {label:<66}{ta:>5}/{tb:<5}{ab[CONTRADICTION]:>7.3f}{ba[CONTRADICTION]:>7.3f}"
                         f"{mean:>7.3f}{max(ab[CONTRADICTION], ba[CONTRADICTION]):>6.2f} {flag} "
                         f"{ba[CONTRADICTION]:.2f}/{ba[ENTAILMENT]:.2f}/{ba[NEUTRAL]:.2f}")
            rows.append({"block": title, "variant": label, "a": a, "b": b, "tokens_a": ta, "tokens_b": tb,
                         "ab": [float(x) for x in ab], "ba": [float(x) for x in ba], "mean": float(mean)})

    score_block("Paper pair (sample 05, C1): removing one piece at a time", PAPER)
    score_block("Paper pair: controls on the bare pair (one mechanism each)", PAPER_CONTROLS)
    score_block("Paper pair: QUALIFIER on the number vs unshared content elsewhere (hypothesis side = A in B->A)",
                PAPER_QUALIFIER)
    for name, subject, pa, pb, qualifier, separate in QUALIFIER_FACTS:
        score_block(f"Synthetic qualifier test: {name}", [
            ("bare", f"{subject} {pa}.", f"{subject} {pb}."),
            ("A: qualifier ON the quantity", f"{subject} {pa} {qualifier}.", f"{subject} {pb}."),
            ("A: same content as a SEPARATE clause", f"{subject} {pa} and {separate}.", f"{subject} {pb}."),
            ("both: SAME qualifier on the quantity", f"{subject} {pa} {qualifier}.", f"{subject} {pb} {qualifier}."),
            ("B: introducing frame 'We also note that'", f"{subject} {pa}.",
             f"We also note that {subject[0].lower() + subject[1:]} {pb}."),
        ])
    for name, subject, pa, pb in FACTS:
        score_block(f"Synthetic: {name} ({pa} vs {pb})", synthetic_variants(subject, pa, pb))

    lines.append(f"\nmodel {cfg.models.nli}; columns = P(contradiction); mean >= {threshold} is flagged by "
                 f"Stage 3. Last column: full label distribution for B as premise, A as hypothesis.")
    text = "\n".join(lines)
    print(text)
    out = ROOT / "outputs" / "probe_dilution"
    (out.with_suffix(".txt")).write_text(text, encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
