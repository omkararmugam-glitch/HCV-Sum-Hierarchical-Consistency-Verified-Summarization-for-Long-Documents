# Referent check: design and pass criteria, fixed before measurement

Written 2026-09-19, before the check was run on any of the validation data below. Results are reported
in FINDINGS.md section 13 against exactly these criteria. No default is changed on the basis of this
check until the results have been reviewed.

## Problem

18 of sample 05's 92 false flags come from ONE Stage 2 summary sentence, "The network uses a fixed,
manually specified sensor adjacency matrix based on the equipment's wiring diagram.", whose source is
"The third is a static graph attention network that uses a fixed, manually specified sensor adjacency
matrix based on the equipment's physical wiring diagram." (Section 4.2 Baselines). Compression kept a
faithful predicate and dropped what it is about. Stage 3's NLI then reads "The network" as the paper's
own method.

Two approaches are ruled out by earlier measurements and are NOT used: the shared-entity-token gate
(removes all three true C2 flags on sample 05) and any grammatical-subject comparison (the true C2
pair has different subjects: "the learned sensor embeddings" vs "we").

## Design: grounded referent re-check

The author already said what each claim is about; compression is what lost it. So instead of
comparing surface tokens or subjects between the two claims, each claim is put back into the
referential context it came from, and the SAME contradiction detector is asked again.

1. **Referent grounding (the Stage 2 part).** For a summary claim X from section s, its grounding
   context g(X) is: the title path of section s, then the source sentence immediately before X's
   grounding sentence (the antecedent window; omitted when the grounding sentence opens the section),
   then the grounding sentence itself -- the source sentence of section s with the highest embedding
   cosine to X. For a claim that IS a source sentence (the sibling side of Stage 3b, or the source
   diagnostic), the grounding sentence is the claim itself. The summary text is not edited; the
   grounding is computed from the document index at check time.
2. **Re-check.** For a flagged pair (A, B):
   P1 = P(contradiction | premise = g(A), hypothesis = nli_text(B)),
   P2 = P(contradiction | premise = g(B), hypothesis = nli_text(A)),
   re-check score = mean(P1, P2), using the pipeline's own aggregation and threshold (0.5).
   Context is added on the PREMISE side only, because FINDINGS 12.1 measured that extra premise
   content leaves true contradictions intact while extra hypothesis content can suppress them.
3. A flag is "referent-confirmed" if the re-check score >= threshold; otherwise "referent-rejected".
   The judgement of whether two claims are about the same thing is made by the NLI model reading the
   author's own referring expressions, with no token, entity or subject overlap rule.

Cost: 2 NLI calls per FLAG (not per compared pair).

## Validation sets

(a) MUST KEEP -- true flags:
    - sample 05: every flag matching C2 (default path); C3 on the source diagnostic; C1 if the default
      path flags it after reporting-frame stripping;
    - samples 01 and 03: their true flags;
    - the two real documents of the (interrupted) planted-contradiction run that completed
      (outputs/planted_gate_off: research paper, FOMC minutes): every flag matching a planted pair.
(b) SHOULD REMOVE -- the 18 flags of pattern A ("referent lost in Stage 2 compression") on sample 05.
    Also reported, not part of the criterion: sample 05's other false-flag patterns (B-E).
(c) EXTERNAL -- RefNLI cannot be used: its records hold only a premise sentence and a claim, with no
    document context, so there is nothing to ground against. Used instead: all flags on the four real
    documents (outputs/real_docs_v2_gate_off; published, presumably consistent, so nearly all false),
    which played no part in this design.

## Pass criteria (all must hold for the check to be proposed as a default)

1. (a): 100% of true flags confirmed.
2. (b): at least 50% of the 18 pattern-A flags rejected.
3. (c): at least 25% of real-document flags rejected, while (a)'s planted real-document flags are kept.

If any criterion fails, the result is reported as is, and no threshold or context rule is adjusted
on these same sets to make it pass.
