"""Grounded referent re-check for Stage 3 flags (design: data/eval/REFERENT_CHECK_PREREGISTRATION.md).

NOT wired into the pipeline: nothing here runs by default. It exists so the check can be validated
(scripts/eval_referent_check.py, FINDINGS.md section 13) before any decision to use it.

Problem it targets: Stage 2 compression keeps a faithful predicate and drops what it is about --
"The third is a static graph attention network that uses a fixed ... adjacency matrix" becomes
"The network uses a fixed ... adjacency matrix" -- and Stage 3's NLI then reads "The network" as the
document's own method. 18 of sample 05's 92 false flags came from that one sentence.

Approach: the author already said what each claim is about. Each claim is put back into the
referential context it came from (section title, the preceding source sentence, and the source
sentence it was derived from), and the SAME contradiction detector is asked again with that context as
the PREMISE. No token, entity or grammatical-subject comparison is used. Context goes on the premise
side only: FINDINGS 12.1 measured that extra premise content leaves true contradictions intact, while
extra hypothesis content can suppress them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ContradictionConfig
from .models import CONTRADICTION, Embedder, NLIModel
from .scoring import DocumentIndex
from .types import ClaimRef, Contradiction


@dataclass
class Grounding:
    section_title: str
    antecedent: str            # source sentence before the grounding sentence ("" if it opens the section)
    sentence: str              # source sentence the claim was derived from (or the claim itself)
    similarity: float          # cosine between the claim and its grounding sentence

    def as_premise(self) -> str:
        title = f"[{self.section_title}] " if self.section_title else ""
        return f"{title}{self.antecedent + ' ' if self.antecedent else ''}{self.sentence}".strip()


def ground_claim(claim: ClaimRef, index: DocumentIndex, embedder: Embedder, is_source: bool = False) -> Grounding:
    """Map a claim to the source sentence it came from, plus the referential context before it.

    ``is_source``: the claim is itself a source sentence (Stage 3b's sibling side, or the source
    diagnostic), so it grounds to itself.
    """
    section = index.sections[claim.section_index]
    title = section.title or ""
    ids = list(section.sentence_indices)
    if is_source:
        local = claim.sentence_index
        sim = 1.0
    else:
        sims = index.embeddings[ids] @ embedder.encode([claim.text])[0]
        local = int(np.argmax(sims))
        sim = float(sims[local])
    antecedent = section.sentences[local - 1] if local > 0 else ""
    return Grounding(title, antecedent, section.sentences[local], sim)


def referent_recheck(contradictions: list[Contradiction], index: DocumentIndex, embedder: Embedder,
                     nli: NLIModel, cfg: ContradictionConfig) -> list[dict]:
    """Re-score each flag with each claim's grounding context as the premise.

    Returns, per flag: the two groundings, both re-check directions, the mean, and whether the flag is
    ``confirmed`` (mean >= cfg.threshold, the detector's own rule). Nothing is modified.
    """
    from .contradiction import AGGREGATIONS, nli_text   # local import: contradiction imports nothing from here

    if not contradictions:
        return []
    groundings = []
    for c in contradictions:
        ga = ground_claim(c.claim_a, index, embedder, is_source=c.kind == "diagnostic")
        gb = ground_claim(c.claim_b, index, embedder, is_source=c.kind in ("one_sided", "diagnostic"))
        groundings.append((ga, gb))
    pairs = []
    for c, (ga, gb) in zip(contradictions, groundings):
        pairs.append((ga.as_premise(), nli_text(c.claim_b.text, cfg)))
        pairs.append((gb.as_premise(), nli_text(c.claim_a.text, cfg)))
    probs = nli.predict(pairs)[:, CONTRADICTION]
    out = []
    for k, (c, (ga, gb)) in enumerate(zip(contradictions, groundings)):
        p1, p2 = float(probs[2 * k]), float(probs[2 * k + 1])
        score = float(AGGREGATIONS[cfg.direction_aggregation](np.array([p1]), np.array([p2]))[0])
        out.append({"grounding_a": ga, "grounding_b": gb, "p_a_context": p1, "p_b_context": p2,
                    "score": score, "confirmed": score >= cfg.threshold})
    return out
