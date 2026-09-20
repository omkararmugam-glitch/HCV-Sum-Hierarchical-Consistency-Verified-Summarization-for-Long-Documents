"""Generate a large SYNTHETIC document of a known size, with planted contradictions.

Usage:
  python scripts/make_benchmark_document.py --sections 30 --sentences 11 --out data/bench/medium.md

The text is templated prose, not real writing: it exists to measure cost and to check that Stage 3
still finds planted contradictions at scale. Summary QUALITY must never be judged on it -- use a
real document for that. Roughly 450 words per printed page, reported on generation.

Alongside the document it writes ``<name>_truth.json`` listing each planted contradiction as the
two distinctive substrings, in the same shape as data/samples/ground_truth.json.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TOPICS = [
    ("Regional Sales", "sales in the {region} region", ["grew {pct} percent", "declined {pct} percent",
                                                        "held flat", "recovered by {pct} percent"]),
    ("Manufacturing", "output at the {region} plant", ["rose to {num} thousand units",
                                                       "fell to {num} thousand units",
                                                       "was constrained by tooling shortages"]),
    ("Procurement", "lead times for {material}", ["improved to {num} weeks", "lengthened to {num} weeks",
                                                  "remained at {num} weeks"]),
    ("Workforce", "headcount in the {region} organisation", ["increased by {num}", "decreased by {num}",
                                                             "was unchanged"]),
    ("Compliance", "the {region} audit", ["closed with no findings", "raised {num} minor findings",
                                          "remains open pending documentation"]),
    ("Logistics", "freight cost per shipment from {region}", ["fell {pct} percent", "rose {pct} percent",
                                                              "was renegotiated"]),
    ("Research", "the {material} qualification programme", ["met its milestone", "slipped by {num} weeks",
                                                            "was expanded to a second supplier"]),
    ("Capital Projects", "the {region} expansion", ["was approved at {num} million dollars",
                                                    "was deferred to next year",
                                                    "completed commissioning"]),
]
REGIONS = ["Monterrey", "Ohio", "Frankfurt", "Osaka", "Bengaluru", "Lyon", "Gdansk", "Tucson",
           "Hamilton", "Rotterdam", "Chennai", "Leipzig"]
MATERIALS = ["specialty resin", "ceramic substrate", "copper foil", "optical adhesive",
             "lithium separator", "aluminium extrusion"]

# Each planted contradiction is (earlier sentence, later sentence): the later one explicitly denies
# the causal link the earlier one asserts, mirroring sample 01.
CONTRADICTION_TEMPLATES = [
    ("Because the {region} line was idle for three weeks, backlog orders went unfilled and "
     "{material} revenue fell {pct} percent in the period.",
     "The fall in {material} revenue was unrelated to the {region} line being idle: "
     "safety stock covered every customer order, and no backlog orders went unfilled."),
    ("All {region} customers were migrated to the new platform before the quarter closed, and the "
     "legacy environment was retired.",
     "About forty {region} customers are still served from the legacy environment while their data "
     "transfers complete."),
    ("The {region} facility operated at full capacity throughout the period with no unplanned downtime.",
     "Unplanned downtime at the {region} facility totalled eleven days during the period."),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sections", type=int, default=30)
    ap.add_argument("--sentences", type=int, default=11, help="sentences per section")
    ap.add_argument("--contradictions", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    def fill(template: str) -> str:
        return template.format(region=rng.choice(REGIONS), material=rng.choice(MATERIALS),
                               pct=rng.randint(2, 40), num=rng.randint(2, 900))

    sections: list[list[str]] = []
    titles: list[str] = []
    for i in range(args.sections):
        name, subject, verbs = TOPICS[i % len(TOPICS)]
        titles.append(f"{i + 1}. {name} Review {i + 1}")
        body = []
        for _ in range(args.sentences):
            sentence = fill(f"In the period under review, {subject} {rng.choice(verbs)}.")
            body.append(sentence[0].upper() + sentence[1:])
        sections.append(body)

    # Plant contradictions in well-separated sections so they always cross a section boundary.
    truth = []
    if args.sections >= 4:
        for k in range(min(args.contradictions, len(CONTRADICTION_TEMPLATES))):
            early_t, late_t = CONTRADICTION_TEMPLATES[k]
            region, material = rng.choice(REGIONS), rng.choice(MATERIALS)
            pct, num = rng.randint(5, 30), rng.randint(20, 400)
            early = early_t.format(region=region, material=material, pct=pct, num=num)
            late = late_t.format(region=region, material=material, pct=pct, num=num)
            a = 1 + k * 2
            b = args.sections - 2 - k * 2
            sections[a].insert(len(sections[a]) // 2, early)
            sections[b].insert(len(sections[b]) // 2, late)
            truth.append([early[:70], late[:70]])

    lines = ["# Synthetic Benchmark Document", "",
             "Generated by scripts/make_benchmark_document.py -- templated prose for cost measurement.", ""]
    for title, body in zip(titles, sections):
        lines += [f"## {title}", "", " ".join(body), ""]
    text = "\n".join(lines)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    truth_path = args.out.with_name(f"{args.out.stem}_truth.json")
    truth_path.write_text(json.dumps({args.out.stem: {"contradictions": truth}}, indent=2), encoding="utf-8")

    words = len(text.split())
    print(f"wrote {args.out}")
    print(f"  {args.sections} sections, {sum(len(b) for b in sections)} sentences, {words} words "
          f"(~{words / 450:.0f} printed pages), {len(truth)} planted contradictions")
    print(f"  ground truth: {truth_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
