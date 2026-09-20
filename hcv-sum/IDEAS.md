# Ideas and scoping notes (not implemented)

Design notes for possible future work. Nothing here changes the pipeline; each idea says what it would
take, how it would be tested, and whether it is worth doing given the current priorities (FINDINGS.md
sections 12-13: precision, i.e. false flags, is the main open problem).

---

## Idea 1: protect the document's semantic backbone (a cheap approximation of GloSA-sum)

### Motivation

GloSA-sum (Zhang et al., ICLR 2026; FINDINGS 9.6) protects a document-wide structural backbone from
deletion. Our only protection, `summarization.protect_cross_referenced`, is local: it re-attaches a
source sentence verbatim only if another section's Stage 2 retrieval reached for it and the generated
summary dropped it. It was added because compression was deleting BOTH sides of sample 01's planted
contradiction. The same failure still occurs: on sample 05, C3 is caught only by the opt-in source
diagnostic, because both of its sentences were compressed out of their section summaries (FINDINGS 12).

### Proposal

A non-topological approximation using what the pipeline already computes:

1. **Graph.** Nodes are source sentences (`DocumentIndex.embeddings` already holds them). Edges are
   cosine similarities, sparsified to each sentence's k nearest neighbours (k ~ 10) above a floor
   (~0.3), so the graph is O(n k) rather than O(n^2). The block-wise selection in `pair_selection.py`
   already computes top-k neighbours without the full matrix.
2. **Centrality.** Weighted PageRank on that graph. This is not new: PageRank on a sentence-similarity
   graph is LexRank (Erkan & Radev, 2004) / TextRank (Mihalcea & Tarau, 2004), a standard extractive
   baseline. The idea is to use it as a protection signal, not as the summarizer. A second variant is
   worth testing because it is closer to what topology captures: **betweenness centrality** marks
   sentences that BRIDGE otherwise weakly connected groups of sentences, whereas PageRank marks hubs
   inside a dense group. GloSA-sum's persistent homology is about connectivity structure across scales,
   which bridges approximate better than hubs do.
3. **Protection.** The top p% of sentences by centrality (or the top 1 per section) join the
   cross-referenced sentences as candidates for verbatim re-attachment, through the SAME mechanism and
   the same "already covered by the summary" check (`protection_coverage_similarity`).
4. **Config.** `summarization.protect_central: off | pagerank | betweenness`, `protect_central_share`.
   Off by default, so the four samples would not change.

Cost: building the kNN graph reuses the chunked similarity code (seconds at 25,000 sentences);
PageRank is a few dozen sparse matrix-vector products; betweenness is costlier (O(n E) exact) and
would need its sampled approximation on long documents. No model calls either way.

### How to test it

Before writing any pipeline code, a one-hour offline measurement settles whether it can help at all:
rank every source sentence of samples 01, 03 and 05 by PageRank and by betweenness, and report where
the sentences carrying the planted contradictions fall (C1-C3 on sample 05; one pair each on 01 and
03). If they are not in the top ~20%, protecting the "backbone" would not protect them, and the idea
has no bearing on the recall problem it is meant to address.

Only if they rank high: prototype behind the config switch and measure, on all five samples, (a)
contradictions caught, (b) false flags, and (c) summary length.

### Status (2026-09-19): investigated and deprioritized

The offline check below was run (FINDINGS.md 13.4). The Limitations sentence behind C2 and C3 ranks in
the 3rd-25th percentile in every graph; the other contradiction sentences are middling, and no single
graph/measure puts them in the top 20% consistently. Not prototyped. Kept here for the record.

### Estimate (written before the check): document as future work; do not prototype now

1. **It works against the current priority.** Every protected sentence becomes a Stage 3 claim. FINDINGS
   12.2 measured that false flags grow with the number of compared pairs at a roughly constant rate
   (4.5-5.7 per 100 pairs), so more claims means proportionally more false flags. Recall-oriented changes
   should wait until the referent check (FINDINGS 13.2) or something like it has brought the per-pair
   false-flag rate down.
2. **The expected payoff is doubtful.** A prediction, to be checked by the offline measurement: sentences
   that carry contradictions are often NOT central. On sample 05, C2 and C3 both involve a Limitations
   sentence ("only evaluated between plants using similar underlying equipment vendors"), which is
   peripheral to the paper's main theme. Centrality favours the repeated main-theme sentences that the
   summarizer already keeps.
3. **The offline measurement is cheap and decides it.** It is worth running whenever recall comes back
   into focus; the prototype is not, unless that measurement comes out positive.

---

## Idea 2: multiple related documents (scoping only)

### What exists

The pipeline handles exactly ONE document. Its structure is document -> sections -> sentences:
`DocumentIndex` indexes one document, Stage 2 retrieves context from other sections of the same
document, and Stage 3 compares claims across sections of that document.

### What a multi-document version would need

Use cases: several earnings calls from one company, a contract and its amendments, several related
papers.

1. **A level above sections.** Documents become the top of the hierarchy: corpus -> document -> section
   -> sentence. Every section and claim needs a document id, and the evidence panel has to cite
   document + section.
2. **Cross-document interaction.** Liu & Lapata (2019; FINDINGS 9.6) do this with learned
   inter-paragraph attention across documents, after a learned ranker keeps only the most salient
   paragraphs. That needs a trained model and a GPU. The CPU-feasible analogue within our design is to
   extend Stage 2's retrieval to sections of OTHER documents, plus a salience ranker to bound input
   size (as their ranker does), since the total number of paragraphs grows with every added document.
3. **Stage 3 generalises almost directly.** Sibling-contradiction detection compares claims from
   different sections; a multi-document version compares claims from different documents (and from
   different sections within each). The candidate selection is already "rows vs columns from a
   different group", and the memory-bounded selection in `pair_selection.py` already handles large
   row x column products. Resolution also carries over: "which claim does its own source support
   more" becomes "its own document".
4. **But the referent problem gets worse, and must be solved first.** Across documents, most apparent
   conflicts are about different referents: Q1 "revenue grew 5%" vs Q2 "revenue fell 2%" is two
   periods, not a contradiction; an original contract clause vs its amendment is a change, not an
   inconsistency. This is the reference-determinacy failure (FINDINGS 9.3, 12.2) at a larger scale, and
   it would multiply the false flags that already dominate single-document runs. A multi-document
   version needs explicit anchoring (document date, period, version) and a working referent check
   before its Stage 3 output could be trusted.
5. **Supersession needs its own label.** When a later document legitimately updates an earlier one
   (an amendment, a revised guidance figure), the right output is "superseded by document X", not
   "contradiction". Stage 3 has no such label today.

### Status

Scoping note only. Not planned for implementation now: items 4-5 depend on the precision work that is
still open for single documents.
