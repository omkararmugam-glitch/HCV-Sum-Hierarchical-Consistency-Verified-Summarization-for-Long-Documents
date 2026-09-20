# HCV-Sum: what works, what does not, and what changed

Numbers come from runs on the five sample documents (`data/samples/`), four public real documents
(`data/external/eval_docs/`), a frozen labelled pair set (`data/eval/`) and RefNLI, each reproducible
with the script named where the result is reported. Raw per-stage reports and JSON are in `outputs/`.
Section 0 summarises the whole research phase; sections 1-13 are the detailed record, in order.

## 0. Research-phase summary (read this first)

This section ties the whole research phase together. The numbered sections below it are the detailed
record, in the order the work happened; every claim here points to the section with the measurement.

### Capability statement

HCV-Sum is a CPU-only, five-stage pipeline that segments a document, summarises each section with
context retrieved from other sections, checks summary claims against each other and against sibling
sections' source text for contradictions, merges the result, and cites every final sentence to its
source. It reliably catches explicit, directly stated conflicts between two sentences: on a frozen
100-pair set of real-document sentences it catches 24 of 36 constructed contradictions, including every
direct numeric conflict (5/5), with 2 false flags on 64 real pairs, and on the five sample documents its
default path catches 4 of 6 planted contradictions (one of them, sample 05's C1, only at 0.523, just over
the 0.5 threshold). It never deletes content on its own judgement: flags are kept and marked for review.
It misses contradictions that need arithmetic (0 of 7 derived-numeric pairs; sample 04's contract term),
causal denials (0 of 2), conflicts whose numbers are attached to differently phrased quantities or wrapped
in reporting frames, and conflicts whose sentences the summarizer compressed away. Precision is the binding
limit: false flags occur at a roughly constant 4.5-6 per 100 compared sentence pairs, so they grow with
document length -- 122 across the five samples (99 on a four-page paper) and about 610 on four real 6-19
page documents, nearly all false -- because the NLI model cannot tell whether two sentences are about the
same thing (a proposed method vs its ablation or a baseline, one dataset vs another, or a referent the
summarizer erased). No similarity- or proximity-based signal we tested fixes this: NLI scores and
aggregation rules, shared entity or content-word tokens, and graph centrality or proximity all place true
contradictions inside the false-flag distribution. The one untested candidate, a grounded re-check that
asks the detector again with each claim's source context, is designed and pre-registered but unvalidated.
The flag list is therefore a review queue, not a verdict. Runtime is about one minute per page on this
CPU; the architecture now bounds memory and reports every cap it hits, but it has not been run on a
document longer than 19 pages, and the check that the scaling changes left small-document outputs
byte-identical was not completed.

### The research narrative

**1. The detector is good at explicit opposition and blind to anything it must work out.** Stage 3
scores sentence pairs with an NLI cross-encoder. On explicit conflicts it is strong: every direct numeric
conflict in the labelled pair set (5/5) and 13 of 14 explicit oppositions under the current defaults
(10.1; `outputs/eval_pairs.txt`). Its first hard limit
is derivation. Sample 04's contract says the term is 24 months from 1 January 2026 and elsewhere that it
expires 31 December 2026; the pair is compared and scores 0.02, rising step by step to 0.999 only as the
end date is spelled out for it (8.1). On real text this generalised: 0 of 7 derived-numeric
contradictions (hours to days, subtraction, reading a ratio) and 0 of 2 causal denials (10.1),
consistent with Mahendra et al. (2025) on numeric NLI (9.1).

**2. Even explicit conflicts are lost when the numbers are phrased differently.** On the medium paper
(sample 05), "the automotive dataset spans eighteen months of sensor readings..." against "We also note
that the automotive dataset, spanning twenty-four months..., provided..." scored 0.15 although the bare
conflict scores 0.99 (12.1). A controlled ladder of variants ruled out length (padding to 30 tokens, or
adding the same clause to both sides, keeps it at 0.98 or higher) and generic extra content (4 of 24
one-sided framings weakened one direction; none flipped the result). Two specific mechanisms remained,
both acting only on the hypothesis side of the NLI pair: a qualifier attached to the quantity ("months
OF SENSOR READINGS" reads as a different quantity; 0.01-0.04, reproduced on 2 of 3 synthetic facts) and a
reporting frame ("We also note that..."; 0.005-0.63, reproduced on 3 of 3). Each collapses one direction,
leaving the averaged score at about 0.5, a coin flip; the paper's pair had one on each side, so both
collapsed. This is a different limitation from derivation: nothing needs working out. It is pinned by six
model-behaviour tests (`tests/test_nli_dilution_slow.py`).

**3. Summarisation removes evidence before the detector sees it.** Stage 3a compares summary claims, so
it needs both sides of a conflict to survive compression. Early on it caught nothing in normal runs for
that reason; cross-reference protection (re-attaching sentences that another section's retrieval reached
for) fixed sample 01 (8.2), and the one-sided check (3b, summary claim vs sibling source) catches cases
where one side survives (sample 03). Sample 05's C3 is still missed on the default path because both of its
sentences were compressed away; only the opt-in source-level diagnostic finds it (0.97) (12). The
summariser also creates problems of its own: invented speaker attributions in transcripts ("President
Obama says", with no Obama in the document; 11.3) and referents erased by compression ("The third
[baseline] is a static graph network that uses..." became "The network uses...", which alone produced 18
false flags; 12.2).

**4. False flags are a scale problem with a reference cause.** Four synthetic samples gave 14 false
flags, so precision looked adequate. Real documents said otherwise: about 610 flags on four published,
presumably consistent documents, a hand-checked sample of which was entirely false (10.5), and 92 of 95
flags on the four-page paper (12.2). The per-pair rate barely differs between document types (4.5 per 100
compared pairs on the small samples, 5.7 on the paper, and matching rates within every similarity band);
what grows is the number of pairs. Classifying the paper's false flags showed the cause: the proposed
method against its own ablations (23), against baselines and prior work (26), different datasets with the
same numeric structure (21), and the erased referent above (18). All are reference failures: the model
assumes both sentences describe the same thing, the failure Chen et al. (2025) document for NLI datasets
(9.3), which we reproduced at scale on RefNLI (16.8% contradiction precision, 86.4% recall; 9.5).

**5. Every surface-level fix we tried either did nothing or removed true contradictions.** Each was
measured, and each is closed:
- *Comparison-wording filter* ("baseline", "unlike", ...): removed 5 false flags on the samples, 7 of 282
  on RefNLI and 0 on the real pair set, and cost a 0.99-scoring true contradiction; switched off (10.3).
  Only 13 of the paper's 92 false flags contain such wording (12.3).
- *Shared-entity gate*: -45% false flags on RefNLI, where referents are proper names, but on real
  documents it halved recall and removed no false flags (10.2); on the paper it would delete all three
  true C2 flags, because the true pair shares no entity token (12.3). A grammatical-subject match fails the
  same way: the true C2 pair has different subjects ("the learned sensor embeddings" vs "we").
- *Shared-content-word gate*: passed its pre-set criteria only narrowly (-11% false flags on RefNLI, -28%
  on real documents, mostly one transcript) and was left off (11.4).
- *Thresholds and aggregation*: true and false scores overlap (8.3); max-aggregation raises false flags
  more than catches (10.4).
- *Centrality-based "backbone" protection* (a cheap stand-in for GloSA-sum's protection pool, 9.6): the
  sentence both C2 and C3 depend on ranks in the 3rd-25th centrality percentile in every graph tested;
  no single graph lifts the contradiction sentences consistently. Deprioritised (13.4).
- *Graph proximity as a referent check*: across four closeness measures, every true pair sits inside the
  false-flag range; the weakest is at the false flags' median. Closed (13.5).
The common thread: the property that separates a true contradiction from a false flag is whether the two
claims are about the same thing, and that is not visible in scores, tokens, or embedding geometry.

**6. What changed in the pipeline as a result, and what it cost.**
- Deletion switched off (`action: flag`): on real documents 14 of 18 automatic removals had deleted
  correct content, all triggered by false flags (10.5, 11.1).
- Non-claim filter: fragments with neither a verb nor a number (reporter introductions) are no longer
  compared; transcript flags 467 -> 410 (11.2).
- Invented-attribution guard in Stage 2 (11.3); not re-measured on a full run.
- Reporting-frame stripping before NLI (13.1): recovers the paper's C1 (0.523) at the cost of 7 new false
  flags, 5 of them the un-framed sentence now flagged against other datasets' figures. Net +1 true, +7
  false across the five samples. Kept on.
- Context-anchored merging (Ou & Lapata's merge-step technique, 14-15): implemented and measured twice.
  With DistilBART, coverage was unchanged; with an instruction-tuned merge model (flan-t5-base) the context
  was still not used beyond occasional copying, coverage did not improve reliably, and the model introduced
  more contradictions while merging. Closed; left off.

### Corrections made during the phase

Four reported results were wrong when first written and were corrected in place:
1. The tally of wrongful deletions on real documents was 15 of 18; one removed claim ("seven of them
   wrote down three or more cuts") was a real misstatement by the summariser, so the tally is 14 (10.5).
2. Sample 05's C2 was first reported as caught at 0.519, ranked 43rd; the evaluation had stopped at an
   earlier, looser match. It is flagged twice, the higher at 0.91 (12).
3. Frame stripping was first reported as catching C1 at 0.84 and adding 2 true / 6 false flags. The
   evaluation matcher (cosine >= 0.6 to the planted sentences) had accepted a false pair as C1; pair by
   pair, C1 scores 0.523 and the change adds 1 true / 7 false (13.1).
4. The graph-proximity analysis first counted that same false pair and a borderline pair as true; it was
   re-run with strict labels (13.5).
Three of these trace to one cause: matching flags to ground truth by embedding similarity rather than by
identity of the source sentences. Any further evaluation should label a flag true only if its two claims
ground to the planted sentences themselves.

### Left unfinished

- **Grounded re-check** (13.2): designed, pre-registered, coded and unit-tested; not validated, not wired
  into the pipeline.
- **Scaling verification**: pre-flight estimate, Stage 2 checkpoint/resume, memory-bounded Stage 3
  selection, run-limits reporting and `--brief` are implemented and tested with fake models; the
  real-model check that small-document outputs are byte-identical before and after, the medium-document
  scale report, the full-size stress run and SCALING.md were not completed.
- **Planted contradictions in real documents**: stopped after 2 of 4 documents (11.4).
- **Sample 05 ground truth**: labelled by us before any run; not yet confirmed by the document's supplier.
- The stage-4 comparison on the paper showed abstractive merging citing only 2-4 of 20 sections while
  capped extractive merging kept all 20, and context-anchored merging did not change that (14); the
  large-document default (abstractive) has not been revisited in light of it.
- Context-anchored merging was measured on sample 05 only; the 31-section synthetic benchmark and samples
  01-04 were not run (14).
- Three extensions were built, unit-tested and measured (sections 15-17). Context-anchored merging with
  an instruction-tuned merge model (flan-t5-base) is a **negative, closed** result (15). Rule-based
  dialogue-to-description preprocessing had a bug in its first measurement, was fixed and re-measured, and
  is kept off by default (16). The unsupervised multi-document mode works as designed on its 3-document
  synthetic test case, including its predicted failure on timeless facts (17).

### Where the details are

| Topic | Section |
|---|---|
| Stage-by-stage verdict on the first samples | 2 |
| Scaling costs and limits | 5 |
| Why sample 04 is missed; 3a vs 3b; early false positives | 8 |
| Related work: numeric NLI, implicit inference, reference determinacy, RefNLI, long-document architectures | 9 |
| Real-document pair set, entity gate, comparison filter, deletions on real documents | 10 |
| Claim filter, attribution guard, topic gate | 11 |
| Qualifier/framing dilution; false flags scale with pairs | 12 |
| Frame stripping, grounded re-check, centrality and proximity checks | 13 |
| Context-anchored merging (Ou & Lapata at the merge step) | 14 |
| Extensions: instruction-tuned merge model (negative), dialogue preprocessing, multi-document mode | 15-17 |


## 1. Environment and compatibility issues found

| Issue | Effect | Resolution |
|---|---|---|
| Python 3.14 is the machine default | PyTorch wheels exist (2.14) but the HF tokenizer stack is least tested there | Built the venv on **Python 3.11.9** |
| Project folder is inside OneDrive | a torch venv is ~2 GB / tens of thousands of files that OneDrive would sync and can lock mid-install | venv lives at `C:\Users\Omkar\.venvs\hcv-sum`, outside OneDrive |
| **transformers 5.x loads checkpoints in their stored dtype** | `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` ships fp16 → ran at **170 ms/pair on CPU instead of 27 ms** (6x slower), identical scores | all models are now loaded with `dtype=torch.float32` (`models.py`) |
| flan-t5 has no `max_position_embeddings` | input silently truncated at 512 tokens while the config allowed 1024 | input budget capped at `summarizer.model_max_input` |
| DistilBART-CNN detokenization | `"12 percent ."`, `"( slightly )"` | `clean_generated()` |
| Generation length limit cuts the last sentence | `"Freight costs fell 11% year over"` entered summaries and the evidence panel | `drop_incomplete_tail()` in Stages 2 and 4 |
| ...and when the cut sentence was the section's **only** summary sentence | a truncated legal clause (`"...giving rise to the"`) reached the final contract summary | fall back to the section's first source sentence |
| HF symlink warning on Windows | cache uses more disk; harmless | silenced in the CLI |
| **torch 2.14 made TorchScript deprecation a visible `FutureWarning`** | transformers' DeBERTa-v2 module applies `@torch.jit.script` to six helpers at import time, so loading our NLI model printed it; no released transformers avoids it (`main` still has all six) | a message- **and** module-scoped filter for that one warning (`silence_torchscript_deprecation()` in `cli.py`, mirrored in `pyproject.toml`); all other warnings stay visible, enforced by `tests/test_warnings.py` |

Versions: Python 3.11.9, torch 2.14.0+cpu, transformers 5.17.0, sentence-transformers 6.0.1, nltk 3.10.3
(`requirements-lock.txt`). Full pipeline: **40-70 s per document on CPU**, plus ~10-30 s for the optional
source diagnostic. Timings in some run JSONs are inflated where runs overlapped on this 16-thread CPU.

## 2. Stage-by-stage verdict

### Stage 1 — segmentation: works
- Markdown, numbered and ALL-CAPS headings are detected; a document title with no body folds into the
  first section's title. Samples 01/02/04 segmented on their real structure (5/6/8 sections).
- The transcript (no headings) fell back to embedding segmentation and produced 5 coherent parts
  (intro, CEO, CFO, operator hand-off, Q&A).
- **Weakness:** one real boundary was missed — the CEO's migration remarks and product launch stayed in
  one section (similarity dip depth 0.08 vs cutoff 0.25). The cutoff is relative
  (`mean + 0.5*std`), so it always finds *some* boundaries even in a single-topic document.

### Stage 2 — context-anchored summarization: works, but this is the pipeline's weakest link
- Retrieval does what it should: for the Manufacturing section of sample 01 the top context sentence
  (sim 0.71) is exactly the Revenue section's denial sentence.
- **Compression drops claims.** Measured retention (share of source sentences entailed by their section
  summary) is **0.31**. DistilBART copies the leading 2-3 sentences of a section, so claims in position 3+
  disappear — including the planted contradiction's second half.
- **Appended context leaks into the summary.** For short sections DistilBART summarized the *context*
  instead of the section. The leak guard (a summary sentence matched better by the context than by its own
  section) removes those; in sample 01 two sections fell back to their own lead sentence.
- Summarizer comparison (`scripts/compare_summarizers.py`, all sections of all samples):

| Summarizer / context | Retention | Faithfulness | Leaks dropped | s/section |
|---|---|---|---|---|
| **distilbart / append (chosen)** | **0.31** | **0.93** | 7 | 4.6 |
| distilbart / none | 0.27 | 0.88 | – | 4.1 |
| flan-t5-base / instruct | 0.19 | 0.69 | 15 | 3.6 |
| flan-t5-large / instruct | 0.15 | 0.70 | 14 | 9.1 |

  flan-t5 ignored "do not summarize the background" and hallucinated freely ("results for the fourth
  quarter of 2008, as reported by the Associated Press" for a supply-chain section). **Honest conclusion:
  context anchoring cannot be said to work here.** Retention 0.31 vs 0.27 is within noise for ~25 sections,
  and a CNN/DM model cannot be told that context is only background. Real anchoring needs a model
  fine-tuned with a context field, or an instruction-following LLM.

### Stage 3 — sibling contradiction detection: the logic works; precision is the problem
Two sub-checks:
- **3a summary vs summary** (as specified).
- **3b summary vs sibling section SOURCE** (added after run 2, see §3).

Evidence that 3a is correct, isolated from Stage 2 (`outputs/ablation_nocompress/`, compression disabled):
the planted pair is ranked **first at P(contradiction)=0.996** (0.999 / 0.992) and resolved as
**UNRESOLVED** because both claims are entailed by their own sections at 0.99 — the honest outcome for a
self-contradictory document, exactly as predicted in `GROUND_TRUTH.md`.

Detection results with the pipeline's real compression:

| Sample | Planted | 3a (summaries) | 3b (one-sided) | Source diagnostic | False positives (3a / 3b / source) |
|---|---|---|---|---|---|
| 01 explicit | 1 | miss (claim dropped by Stage 2) | miss | **found, P=1.00** | 0 / 2 / 7 |
| 02 none | 0 | – | – | – | 1 / 4 / 11 |
| 03 implicit | 1 | miss (CFO clause dropped) | **found, P=0.61** | **found, P=0.60** | 0 / 0 / 2 |
| 04 date arithmetic | 1 | miss | miss | miss | 2 / 5 / 8 |

> **Superseded by section 8.** The table above is the state before cross-reference protection and the
> comparative-framing filter; section 8 has the current numbers and the full investigation.

- **Recall** depends almost entirely on whether Stage 2 keeps both sides. Where it does (ablation), 3a
  finds the planted pair. Where only one side survives, 3b finds it (sample 03). Where neither survives
  (01), only the source diagnostic sees it.
- **The contract's contradiction is missed by design of the method**: "24-month term from January 1, 2026"
  vs "expires December 31, 2026" needs date arithmetic across three sections. Sentence-pair NLI cannot do it.
- **Precision is the honest weak point**, worst on legal text (5 one-sided false flags on the contract).
  Every false positive resolved to `unresolved`, so nothing was deleted from a consistent document —
  but an analyst would see spurious flags. Note the no-contradiction sample still produced 1 + 4 of them.

### Stage 4 — merge: works, default changed to extractive
Measured section coverage of the final summary (via Stage 5 citations):

| Mode | Sections cited (01/02/03/04) | Words | Unsupported |
|---|---|---|---|
| abstractive | 3/5, 3/6, 2/5, 3/8 | 47-73 | 0 |
| abstractive, fan-in 2 | 3/5, 2/6, 2/5, 2/8 | 34-46 | 1 (hallucinated) |
| **extractive (new default)** | **5/5, 6/6, 5/5, 8/8** | 128-215 | 0 |

Abstractive fusion with DistilBART is effectively lead-bias truncation: it silently discards the later half
of every document, which defeats resolving contradictions before merging. Extractive costs fluency
(concatenated claims, no connectives) and length.

Two things extractive merging carries through unchanged:
- **Stage 2 corruptions.** `"twenty-four (24) months"` became **"twenty-24 months"**, and Stage 5's NLI
  rated that sentence *supported* at 0.87. That is what the novel-token flag (below) exists to catch.
- **Boilerplate.** The transcript's final summary still contains *"Operator: Thank you."* and
  *"At this time all participants are in a listen-only mode."* Stage 3 ignores such sentences (the
  non-claim filter), but nothing removes them from the summary itself. Reusing the same filter at merge
  time is the obvious next step; it was not done here because it was not measured.

### Stage 5 — provenance: works
- Every final sentence gets a citation (global source sentence ids + section), a similarity, an entailment
  score and a status. Multi-sentence windows correctly cite fused sentences (e.g. "cites source
  sentences 11,12").
- Similarity alone is explicitly not trusted: a negated near-copy of a source sentence scores similarity
  0.73 and still comes out `unsupported` with a `contradicted_by_source` flag.
- The audit trail survives: sample 03's final summary sentence *"Every enterprise customer is now running
  on the new Helix Core platform."* is `SUPPORTED` (true of its own section) **and** carries a `DISPUTED`
  flag plus the Stage 3b note naming the CFO sentence that contradicts it. This is the motivating failure
  case from the project brief, caught end to end.
- A **lexical backstop** flags summary tokens that occur nowhere in the source (any token containing a
  digit, or at least 5 characters). This is what catches `"twenty-24"`, which entailment rated 0.87
  supported. Legitimate paraphrase also triggers it, so it is a flag for a human, not an error.
- **Weakness:** no calibration. Thresholds (0.70 / 0.35) are guesses; NLI entailment is generous with
  garbled numbers, so "supported" alone is weaker evidence than it sounds — read it together with the
  flags.

## 3. What changed from the original plan, and why

| # | Change | Why |
|---|---|---|
| 1 | **NLI model**: `cross-encoder/nli-deberta-v3-small` → `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` | The small model flagged 22 source pairs on sample 01 (1 true) and 45 on the contradiction-free sample 02. Chosen on `data/probes/contradiction_pairs.jsonl` (38 pairs, independent of the samples): precision 1.00 vs 0.81. |
| 2 | **Direction aggregation**: max → **mean** (`max`/`min`/`mean` configurable) | My original max was a logic error: contradiction is symmetric, entailment is not, so one-directional scores were inflating flags. Pure `min` loses causal denials where one side carries extra content; `mean` keeps precision 1.00 at recall 0.87 on the probe. |
| 3 | **Added Stage 3b, the one-sided check** | Run 2 showed the real-world failure: compression keeps one side of a contradiction, so 3a sees nothing and Stage 5 calls the survivor "supported". 3b is what catches sample 03. |
| 4 | **Merge default**: abstractive → **extractive** | Coverage measurements above. |
| 5 | **Added the non-claim filter** (speaker labels stripped, `min_claim_words`) | Transcript pleasantries ("Thank you, operator." vs "Thanks for taking my question.") scored P=1.00 contradiction under NLI's same-situation assumption. |
| 6 | **Added the context-leak guard** and the incomplete-sentence drop | Both were observed corrupting summaries in run 1. |
| 7 | **float32 loading** | 6x CPU speedup, see §1. |
| 8 | Duplicate-claim checks use **NLI, not similarity** | My first version used embeddings, which ignore negation — it would have disabled the resurrection check in exactly the case it exists for (caught by a unit test). |
| 9 | Added `scripts/eval_nli_probe.py`, `compare_summarizers.py`, `digest_run.py`, `data/probes/`, `ground_truth.json` | Needed to make model choices on evidence rather than on the evaluation documents. |
| 10 | Added the **novel-token flag** in Stage 5 | NLI accepted the corrupted `"twenty-24 months"` at 0.87 entailment. A lexical check catches what entailment does not. |
| 11 | Section summaries fall back to their first source sentence when the only generated sentence is truncated | A cut-off legal clause reached the final summary of sample 04. |

Threshold honesty: `contradiction.threshold` is still **0.5**, its original value. On the probe, 0.5/0.6/0.7
are indistinguishable (precision 1.00, recall 0.87), so there was no evidence-based reason to move it.
Raising it to 0.7 would have suppressed sample 04's false positives — and also hidden sample 03's true
contradiction (0.61). The probe set was written *after* seeing run 1's failure types, so it is independent
of the sample text but not blind to the failure categories.

## 4. What to trust and what not to

**Trust:** segmentation; provenance citations and their traceability; the Stage 3 resolution rule
(better-supported claim wins, otherwise `unresolved` — it never guessed on a genuinely inconsistent
document); the audit trail from a detected contradiction to the final summary sentence.

**Do not trust yet:**
1. **Stage 3 precision** — 1-5 false flags per document; worst on legal text. Fix: calibrate on a labelled
   set (AggreFact / SummaC benchmark), or require a shared entity/event anchor rather than plain cosine.
2. **Claim retention in Stage 2 (0.31)** — the biggest limiter on Stage 3 recall. Fix: a summarizer that
   can be instructed, or claim-aware selection that forces cross-section-salient sentences into summaries.
3. **Context anchoring** — no measurable benefit with DistilBART, and it actively leaks on short sections.
4. **Numeric and arithmetic contradictions** — not detected at all (sample 04).
5. **"supported" as a guarantee** — thresholds are uncalibrated and NLI accepted the corrupted
   "twenty-24 months" (now caught by the novel-token flag, but only because the corruption was lexical).
6. **Scaling** — Stage 3 is O(n²) NLI calls. Fine for these documents (~30-100 pairs); a 100-page filing
   needs blocking or a cheaper first-pass filter.
7. **Boilerplate in the output** — the extractive summary of a transcript keeps operator chatter.

## 5. Scaling: measured costs and what a long document needs

Measured throughput on this CPU (16 threads, float32): **57 NLI calls/sec** (batch size makes no
difference above 16), **99 sentences/sec** embedded, **2.1 s per section** summarized (batched;
4.6 s unbatched). Every pair is scored in both directions, so one pair = two NLI calls.

### Complexity of Stage 3, as written

| Check | Pairs before limits | For 50 sections (~130 claims) | For 200 sections (~500 claims) |
|---|---|---|---|
| 3a summary vs summary | C(claims, 2) minus same-section | ~8,300 pairs = 16,600 NLI calls (~5 min) | ~125,000 pairs = 250,000 calls (~73 min) |
| 3b summary vs sibling source | claims x source sentences | ~130 x 1,600 = 208,000 pairs | ~500 x 6,500 = 3.2M pairs |
| source diagnostic (`--diagnose-sources`) | C(source sentences, 2) | ~1.3M pairs | ~21M pairs |

The similarity floor removes 60-85% of those in practice, but the growth is still quadratic, and
resolution adds NLI support scoring for **every flagged pair** -- a cost driven by the flag count,
not by document length.

### Stage 2 on very long documents: safe, but lossy in a reportable way

Truncation is safe, not silent: the input budget is `min(max_input_tokens, model_max_input)`,
context is dropped before section text, and a section truncated on its own sets `input_truncated`,
which the report prints as `[INPUT TRUNCATED]`. Two real risks remain, both now bounded:
- A section longer than `max_section_tokens` (900) is re-split at semantic valleys first, so
  truncation only happens for a *single sentence* longer than the budget.
- The embedding fallback used to be unbounded: a 6,000-sentence document with
  `min_segment_sentences=4` could produce 1,500 sections, i.e. 1,500 summarizer calls (~52 min).
  `segmentation.max_sections` (250) now grows the minimum segment length instead.

### What was added to make long documents tractable

| Control | Default | Effect |
|---|---|---|
| `contradiction.candidate_top_k` | 30 | each claim is compared only with its k most similar cross-section candidates: pairs become O(claims x k), not O(claims^2) |
| `contradiction.max_pairs` | 40,000 | hard ceiling per check, highest-similarity pairs first |
| `contradiction.max_resolutions` | 200 | caps support scoring; extra flags are reported as `unscored` and nothing is removed for them |
| `segmentation.max_sections` | 250 | bounds the number of summarizer calls |
| `summarization.batch_size` | 4 | 1.97x faster generation, byte-identical output (batches share one length budget, so only sections with the *same* budget are batched) |

**The trade-off, stated plainly:** `candidate_top_k` means a contradiction whose partner is not among
a claim's 30 nearest neighbours is never looked for. Evidence that this is survivable: in all four
planted contradictions measured, the true partner ranked **#1** by cosine (0.54-0.71). That is four
data points, and an adversarial document could bury a partner further down. `max_resolutions` never
deletes anything -- it only stops deciding. All skips are counted in the run stats, so a run always
reports what it did not look at.

### Measured: 15-page synthetic benchmark (31 sections, 485 sentences)

`python scripts/make_benchmark_document.py --sections 30 --sentences 16 --out data/bench/medium_15pages.md`

| Configuration | Wall clock | Stage 2 | Stage 3 | NLI pairs |
|---|---|---|---|---|
| before these controls | 348 s | 63 s | 252 s | 10,694 |
| defaults (top_k 30, max_resolutions 200) | **265 s** | 62 s | 169 s | 8,845 |
| large-document preset (top_k 10, floor 0.35) | 293 s* | 64 s | 197 s | 5,674 |

\* measured before `max_resolutions` existed; resolution dominated that run.

On the four small samples the controls are inert: section summaries, final summaries and flag counts
are **identical** to the pre-optimisation run, and sample 03's planted contradiction is still found.

### Projection to larger documents (structured, ~30 sentences/page)

| Size | Sections | Stage 2 | 3a | 3b | Stage 5 | Total |
|---|---|---|---|---|---|---|
| 15 pages | 31 | 62 s | 36 s | 79 s | 14 s | **4.4 min (measured)** |
| 200 pages | ~400 | ~14 min | ~9 min | ~18 min | ~5 min | **~45 min (projected)** |
| 200 pages, `candidate_top_k=10` | ~400 | ~14 min | ~3 min | ~6 min | ~5 min | **~28 min (projected)** |
| 500 pages | ~1000 | ~35 min | ~7 min | ~15 min | ~12 min | **~70 min (projected)** |

### Pre-filter effectiveness, measured on the 4 samples (Stage 3a + 3b combined)

| Configuration | Pairs possible | Compared | NLI calls | vs no filter | Flags | Catch rate (3a/3b) |
|---|---|---|---|---|---|---|
| no pre-filter (floor 0.0, top_k off) | 796 | 763 | 1,526 | - | 23 | 1/3 |
| **default (floor 0.20 + candidate_top_k 30)** | 796 | 391 | **782** | **-49%** | 14 | **1/3 (unchanged)** |
| default + `require_shared_entity` | 796 | 58 | 116 | -92% | 5 | **0/3 (breaks it)** |

The default pre-filter halves NLI work, removes 9 false flags and changes the catch rate not at all.
The entity gate is far cheaper still and even cleaner on false positives, but it drops the one
contradiction Stage 3b catches, because sample 03's contradicting sentence is all lower case with a
spelled-out number and therefore has no entity tokens. That is why it ships **off** by default, with
a regression test recording the blind spot.

### Recursive batched merge, measured with the real models

`max_group_sections=5` fuses at most 5 blocks per summarizer call and repeats:

| Document | Sections | Rounds | Groups per round | Generations | Contradictions introduced by merging |
|---|---|---|---|---|---|
| 04_contract_term | 8 | 2 | 2 -> 1 | 3 | 0 |
| medium_15pages | 31 | **3** | 7 -> 2 -> 1 | 10 | **8 (all in round 1)** |

Contradiction checking stays at the **leaf level** for resolution, and each round is additionally
checked for contradictions *between its own outputs*. The reasoning: Stage 3 already compares every
pair of leaf claims, so a conflict present in the source is resolved before merging begins. What it
cannot see is a conflict the merger invents. Those are detected but never resolved, because
resolution scores a claim against its own section's source and a merged block spans several
sections, so no honest support score exists. On the 31-section document that check fired 8 times in
round 1 -- real evidence that abstractive fusion introduces conflicts that leaf-level checking misses.

### Two problems at scale that are NOT about speed

1. **Precision collapses on repetitive text.** On the 15-page benchmark the NLI model flagged
   **1,646 pairs** where 2 were planted. The benchmark is adversarial (hundreds of near-identical
   templated sentences differing only by a number or a place name, exactly the failure mode in
   section 2), but a real 200-page filing has plenty of repetitive numeric prose too. Expect to
   spend the review effort on false flags, not on the real ones.
2. **The extractive summary grows linearly with the document.** 485 source sentences produced a
   68-sentence, 726-word "summary" (14% compression). At 200 pages that is ~900 sentences, which is
   not a summary. Long documents need either abstractive merging (which loses section coverage, see
   section 2) or a second selection pass over the kept claims. Neither is implemented.

Also measured at 15 pages: **both planted contradictions were missed**. One had neither side survive
Stage 2; the other survived in shortened form and scored 0.48 against the 0.50 threshold (0.02/0.94
across directions -- the `mean` aggregation trade-off from section 3). Top-k was NOT the cause: that
partner ranked #1. Recall at scale is limited by Stage 2 compression and the threshold, not by the
new limits.

## 6. Bugs found and fixed in the hardening pass

Found by static analysis (pyflakes), a robustness sweep over 14 degenerate documents, and an
invariant check run against the real outputs. All are now covered by regression tests.

| # | Bug | Symptom | Fix |
|---|---|---|---|
| B1 | **Extractive merge re-split its own output** | the kept sentences were joined into one string and re-split, so sentences with no terminal punctuation were fused: two claims became one, and one provenance row disappeared | keep the sentence list as-is |
| B2 | **Stage 2 passthrough re-split the section text** | same class of bug for short sections: bullet lists and transcript lines collapsed into a single "sentence" (a 4-bullet list became 1 claim) | reuse the section's own sentence list |
| B3 | **UTF-8 BOM was not stripped** | a BOM (routine on Windows) became part of the first heading, so Stage 1 missed the document's first section | read with `utf-8-sig` |
| B4 | **Non-UTF-8 input crashed** | raw `UnicodeDecodeError` traceback | clean message, exit code 2 |
| B5 | **Fallback replacements were logged as "dropped"** | the report printed a sentence that was actually KEPT under `DROPPED`, which would mislead an auditor | separate `notes` field on each section summary |
| B6 | **Stage 3 flag notes overwrote each other** | a claim contradicted by two different siblings showed only one of them (`dict` assignment / `setdefault`) | notes accumulate, joined with ` \| ` |
| B7 | **Provenance re-encoded every claim for every final sentence** | O(final sentences x contradictions) embedder calls, pure waste | embed each claim once per run |
| B8 | pyflakes findings | unused imports, an f-string with no placeholders, an unused accumulator | removed |

Robustness sweep (empty file, whitespace only, single sentence, headings with no body, 120-sentence
single section, one 400-clause sentence, unicode/smart quotes, identical sentences, only pleasantries,
bullets only, CRLF, no terminal punctuation anywhere): **0 crashes**. Empty inputs raise one clean
`ValueError` that the CLI turns into an error message.

`tests/test_warnings.py` guards the one warning filter in the project: it proves that an unrelated
`FutureWarning`, the same message from a different module, and other categories from torch all still
surface, so the filter can never silently become a blanket ignore. The full suite (117 tests) runs with a
clean warnings summary; running it with `-W always::FutureWarning` brings the 7 upstream TorchScript
notices straight back, which is how the filter was verified to be the thing silencing them.

The invariant check is now a permanent test (`check_invariants` in `tests/test_end_to_end.py`): section
offsets are contiguous, every citation's sentence ids resolve to the cited text and stay inside the cited
section, every Stage 3 claim reference resolves to the exact summary/source sentence it names, provenance
rows match final sentences 1:1, and no rejected claim reaches the summary.

## 7. Test suite

`pytest -m "not slow"`: **165 tests, ~5 s**, all with deterministic fake models (a word-overlap embedder, a
rule-based NLI, a lead-extracting summarizer). They cover each stage's logic, the config loader, the CLI and
an end-to-end smoke run. They deliberately prove nothing about model quality.

`pytest -m slow`: **18 tests, ~3 min on this CPU** (was ~18 min before the provenance encoding fix B7),
with the real models on the sample documents. These pin
actual observed behaviour, including three known limitations that are asserted *as* limitations:
`test_planted_pair_is_missed_at_summary_level_because_stage2_drops_it`,
`test_abstractive_merge_loses_section_coverage` and `test_contradiction_free_document_stays_nearly_clean`
(an upper bound on false positives, not a claim of zero). If Stage 2 improves, the first test is meant to
fail and be rewritten.

## 8. Contradiction mechanism: what actually works (investigation of three failures)

### 8.1 Why sample 04's contradiction is missed: NLI cannot do date arithmetic

Full trace of the planted pair through the pipeline (`outputs/cli_json/04_contract_term.json`):

| Step | Finding |
|---|---|
| Segmentation | "...initial term of twenty-four (24) months" landed in section 2 (TERM AND RENEWAL); "...expires at the end of the initial term on December 31, 2026" in section 7 (TERMINATION). **Different sections, so siblings, eligible for comparison.** |
| Stage 2 | **Both survived** into their section summaries (the first garbled to "twenty-24 months", the second verbatim). |
| Stage 3 pre-filters | Cosine 0.68, well above the 0.20 floor; top-k skipped nothing. **The pair was compared** in 3a and 3b. |
| NLI | **P(contradiction) = 0.03 / 0.01, mean 0.021** against a 0.50 threshold. |

So it is not segmentation, not summarization, not a pre-filter and not the threshold. Isolating the cause
by removing one reasoning step at a time:

| Variant of the same pair | Mean P(contradiction) | Result |
|---|---|---|
| as written: needs "Effective Date = Jan 1 2026" from another section, then +24 months | 0.021 | missed |
| effective date inlined, still needs +24 months | 0.151 | missed |
| end date stated explicitly ("ending on December 31, 2027") | **0.905** | caught |
| same unit, different number ("24 months" vs "12 months") | 0.999 | caught |
| control: plain semantic opposition | 0.999 | caught |

**Finding:** the NLI model detects a numeric conflict when the two values are directly comparable, and
fails almost completely (0.02) when one value must be *derived*. Sentence-pair NLI does semantic
opposition, not arithmetic. This matches the one arithmetic case in the independent probe set (a lease
"running three years from January 2024" vs "expires January 2025": 0.11 / 0.05). It is a limitation of
the method, not a bug, and it is pinned by two slow tests
(`test_nli_cannot_detect_a_contradiction_that_needs_date_arithmetic` and its explicit-date control).
Fixing it needs a different component -- extracting dates and durations as structured values and
checking them arithmetically -- which is not attempted here.

### 8.2 Stage 3a had never worked in a normal run

Audit of every recorded run: Stage 3a had caught a planted contradiction **exactly once**, in the
no-compression ablation (`outputs/ablation_nocompress`, sample 01, P=0.996). In every normal run it
caught nothing. In samples 01 and 03 the cause was Stage 2 compressing away one or both sides; in
sample 04 both sides survived and NLI missed it (section 8.1).

Option A turned out to be feasible without over-engineering, so it was implemented as
**cross-reference protection**. Stage 2 already retrieves, for every section, the sentences elsewhere in
the document most related to it. A sentence that some other section reached for is load-bearing for
cross-section comparison, so if compression drops it, it is re-attached verbatim. In sample 01 both
halves of the planted contradiction had been retrieved as each other's context (cosine 0.71). With
protection both survive, and **Stage 3a flags the pair at P=0.996 in the default pipeline**, its first
catch in a normal run.

Option B's honesty requirement applies anyway, and the code and README now say it: **3a and 3b are
complementary, and neither is the primary mechanism.** Each catches exactly one of the three planted
contradictions, and each catches one the other structurally cannot: 3a needs both sides present in the
summaries, while 3b sees a side that compression dropped because it compares against the source.

### 8.3 False positives: the comparative-framing cause fixed, the rest diagnosed

Four of the five false positives on the contradiction-free document paired a description of the proposed
system with "The baseline is the same six-layer classifier fine-tuned without retrieval". The two
sentences describe **two different systems on purpose**; NLI reads them as conflicting because its
training data assumes both sentences describe one situation. That is a model limitation, but it has a
visible textual cause the pipeline was ignoring: explicit comparative framing ("the baseline", "unlike",
"compared with", "faster than" and similar). Pairs where either sentence is framed that way are now
skipped (`contradiction.skip_comparative_framing`), and the skips are counted in the run stats.

> Section 9.4 reconsiders this filter in light of the reference-determinacy literature: it removes
> one surface cue of a deeper cause, rather than the cause itself.

### 8.4 Before / after on all four documents (2x2 ablation of the two fixes)

`--diagnose-sources` in every run; scored with `scripts/score_samples.py`.

| Configuration | Default-path catch (3a or 3b) | 3a | 3b | False positives, all docs | FP on clean doc |
|---|---|---|---|---|---|
| before (neither fix) | 1/3 | 0/3 | 1/3 | 13 | 5 |
| comparative filter only | 1/3 | 0/3 | 1/3 | **8** | **2** |
| protection only | **2/3** | 1/3 | 1/3 | 23 | 9 |
| **both (new default)** | **2/3** | 1/3 | 1/3 | 14 | 4 |

The exact trade-off: the comparative filter is a clean win (5 fewer false positives, no recall cost).
Protection doubles the default-path catch rate (1/3 to 2/3) but adds sentences, therefore pairs,
therefore false positives (6 more, net, with the filter on). **Both are on by default** because catching
contradictions is the purpose of the stage. A precision-first run can set
`summarization.protect_cross_referenced=false`: 8 false positives, 1 of 3 caught.

The false positives that remain on the clean document are **not** comparative framing, and no text
signal separates them from true positives:
- "At inference time, the **five** most similar training notes are retrieved" vs "All models are trained
  with **five** random seeds": a shared number with different referents (0.531).
- "We propose RACN, a model that retrieves..." vs "RACN encodes every training note...": two steps of
  one method read as rival claims (0.525).
- Two involving protected sentences (0.507 and 0.630).

They are strongly asymmetric across the two NLI directions (for example 1.00 / 0.06), but so is sample 03's
**true** positive (0.22 / 1.00), so asymmetry cannot filter them either.

### 8.5 Current capability, stated modestly

On four hand-written sample documents (three containing one planted cross-section contradiction each,
one containing none), the default pipeline flags 2 of the 3 planted contradictions and raises 14 false
flags in total, 4 of them on the contradiction-free document. It resolves none of the flags
automatically: in every case both claims are fully supported by their own sections, so all are reported
as `unresolved` for a human to judge, and nothing is removed from the summary. It detects direct
semantic opposition between claims in different sections, including cases where summarization dropped
one side, and it cannot detect contradictions that require arithmetic or combining facts from more than
two sentences. Four documents and three contradictions are far too few to estimate catch or false
positive rates that would hold on real documents; these numbers describe this test set only.

## 9. Connection to prior research

Three 2025 papers bear on the findings in section 8. Each was checked against its published abstract
and full text before being cited here; where a paper's result only partly matches ours, the difference
is stated rather than smoothed over.

### 9.1 Numeric reasoning in NLI -- Mahendra et al. (2025)

Mahendra, R., Spina, D., Cavedon, L., & Verspoor, K. (2025). *Evaluating Numeracy of Language Models as a
Natural Language Inference Task.* Findings of NAACL 2025, pp. 8351-8376.
[aclanthology.org/2025.findings-naacl.467](https://aclanthology.org/2025.findings-naacl.467/)

The paper frames numeracy as NLI and evaluates 49 models, including models fine-tuned on NLI datasets
(the same class as our DeBERTa-v3 cross-encoder). It finds that models "achieve fair to good accuracy for
NLI entailment cases" but "still struggle to predict contradiction and neutral cases", and that ability in
one numeracy skill (arithmetic, number comparison, normalization) does not carry over to the others.
Our sample 04 miss is a concrete instance: the contradiction needs *arithmetic* (+24 months) and
*normalization* (a month count against a calendar date), two of the paper's three named skills, and our
controlled ladder shows the score recovering only as each is removed (0.021 as written, 0.151 with the
effective date inlined, **0.905** with the end date stated, **0.999** for "24 months" vs "12 months").

**Where our evidence differs:** the paper reports contradiction as the weak class *in general*, whereas
our model caught the direct numeric contradiction ("24 months" vs "12 months") at 0.999. That is one
example on one model, so it does not contradict the paper; it does mean our results support the paper's
finding for *derived* values specifically, not a general claim that numeric contradiction is always missed.

### 9.2 Implicit versus explicit inference -- Havaldar et al. (2025)

Havaldar, S., et al. (2025). *Entailed Between the Lines: Incorporating Implication into NLI.*
ACL 2025 (long papers). [aclanthology.org/2025.acl-long.1552](https://aclanthology.org/2025.acl-long.1552/)

The paper introduces the INLI dataset, which separates *explicit* from *implied* entailment, and reports
that models trained on standard NLI data are at chance on implied entailment: T5 models trained on SNLI,
MNLI and WANLI score 0.500, 0.528 and 0.525, against 0.94 human agreement, until fine-tuned on INLI.
Our results show the same explicit-to-implicit cliff, on the contradiction side. The derived-date case
goes from 0.021 to 0.905 once the implicit value is made explicit. Sample 03 is an implied contradiction
("every enterprise customer is now running on the new platform" versus "roughly sixty enterprise accounts
are still being served from the legacy environment"), and it is our weakest true positive: 0.61, just
over the 0.50 threshold, and strongly one-directional (0.22 / 1.00).

**Where our evidence differs:** "implied" in this paper means pragmatic and world-knowledge implication,
explicitly not arithmetic; it studies entailment, not contradiction; and its fine-tuned models are T5, not
DeBERTa. So it corroborates the general pattern (explicitly stated relations are handled, implied ones are
not) rather than our date-arithmetic mechanism specifically. Sample 03 is the closer match to what the
paper actually measures.

### 9.3 Reference determinacy -- Chen et al. (2025)

Chen, S., Malaviya, C., Fabrikant, A., Taitelbaum, H., Schuster, T., Buthpitiya, S., & Roth, D. (2025).
*On Reference (In-)Determinacy in Natural Language Inference.* Findings of NAACL 2025.
[aclanthology.org/2025.findings-naacl.450](https://aclanthology.org/2025.findings-naacl.450/)

NLI annotation assumes the premise and hypothesis "refer to the same context". The paper's RefNLI
benchmark (1,143 pairs) shows that when they do not, "finetuned NLI models and few-shot prompted LLMs both
fail to recognize context mismatch, leading to over 80% false contradiction"; a fine-tuned T5-Large reaches
15.8% contradiction precision at 92.4% recall. That is exactly our clean-document false positives:
"The baseline is the same six-layer classifier fine-tuned without retrieval" scored 0.51-0.71 as
contradicting descriptions of RACN, although the two sentences describe *different systems*. Our NLI
model is trained on MNLI, FEVER and ANLI, standard NLI data of the kind the paper argues embeds the
same-context assumption. The paper's models are T5 and LLMs, not ours, but the failure mode matches.

### 9.4 Is the comparison-wording filter treating a symptom?

Yes. The paper's framing places the cause in the model's assumption that both sentences share a context;
explicit comparison wording ("the baseline", "unlike", "compared with") is only one visible cue that they
do not. The two false positives that survive the filter show the limit: "five retrieved notes" versus
"five random seeds" is a reference mismatch with no comparison word, and "RACN retrieves..." versus "RACN
encodes..." is not a reference mismatch at all but two compatible facts about the same system.

A general shared-context gate would target the cause more directly. The question is whether a version we
can build now would work, so both simple forms were measured on the pairs that matter:

| Pair | Actual | Shared-entity gate | Word-overlap filter (RefNLI's, Jaccard <= 0.15) |
|---|---|---|---|
| "five retrieved notes" vs "five random seeds" | false positive | blocks | discards |
| "RACN retrieves..." vs "RACN encodes..." | false positive | keeps | discards |
| Sample 01 planted pair | **true positive** | **blocks** | keeps |
| Sample 03 planted pair (implicit) | **true positive** | **blocks** | **discards** |

The shared-entity gate (already implemented as `require_shared_entity`, off by default) blocks **both**
true positives and only one of two false positives. The word-overlap filter the RefNLI authors tested
removes both false positives but also discards the implicit sample 03 contradiction, and the authors
themselves report only "minor improvements" from it.

**The structural reason:** an implicit contradiction is, nearly by definition, one where the two sentences
share little surface wording (Jaccard 0.07 for sample 03). Any *lexical* same-reference test therefore
discards exactly the contradictions that are hardest to find and most worth finding. A principled gate
would need to resolve reference semantically -- to decide that "sixty enterprise accounts" falls inside
"every enterprise customer" -- which is itself an inference problem of the kind sections 9.1 and 9.2 show
NLI models handle poorly.

**Assessment (revised after section 9.5): do not implement now, but not for the reason first given.**
On the four documents the shared-entity gate looked harmful. On 1,143 RefNLI pairs it roughly halves
false contradiction flags for a small recall cost (section 9.5). The two results disagree because the
datasets name their referents differently, so neither transfers to our target documents unchecked.
What decides it is an evaluation set drawn from the kind of document the pipeline is meant for, which
does not exist yet.

The comparison-wording filter stays, described as what it is: a narrow heuristic for one surface form
of reference mismatch. On the four samples it removed 5 false positives at no recall cost; on RefNLI it
removed 7 of 282 false flags and cost 1 true contradiction, which is close to no effect.

### 9.5 Our model on RefNLI: the reference failure measured at scale

`python scripts/eval_refnli.py` runs Stage 3's exact decision rule (P(contradiction) in both
directions, mean >= 0.50) on all 1,143 RefNLI pairs. It is an evaluation only: no pipeline code
changes. The data file is downloaded from the authors' repository (which states no license) into
`data/external/`, which is git-ignored and not redistributed. Full output: `outputs/refnli_eval.txt`.

**Headline: our model reproduces the paper's finding almost exactly.**

| | Contradiction precision | Recall | False flags on *neutral* | False flags on *ambiguous* |
|---|---|---|---|---|
| Our model, raw rule (1,143 pairs) | **16.8%** | **86.4%** | 228 / 905 = 25.2% | 51 / 135 = 37.8% |
| Chen et al. (2025), T5-Large | 15.8% | 92.4% | -- | -- |

About five of every six contradiction flags are wrong on this distribution: 282 false against 57
true. Pairs whose truth depends on which referent is meant ("ambiguous") are flagged most often. The
four false positives on our clean sample document are the expected behaviour of this model class, not a
quirk of that document.

**Our existing pre-filters barely help at scale:**

| Filter applied before NLI | Precision | Recall | False flags |
|---|---|---|---|
| none (raw rule) | 16.8% | 86.4% | 282 |
| similarity floor 0.20 alone | 17.8% | 84.8% | 258 |
| comparison-wording filter alone | 16.9% | 84.8% | 275 |
| all three pipeline pre-filters | 17.6% | 80.3% | 248 |

The comparison-wording filter removed 7 of 282 false flags and one true contradiction. Its benefit on
our samples was real, but it does not generalise, which is what a narrow surface heuristic predicts.

**Training overlap.** Our NLI model was trained on FEVER-NLI, and 733 of the pairs come from FEVER.

| Source | Pairs | Recall | False flags on neutral | False flags on ambiguous | Precision |
|---|---|---|---|---|---|
| FEVER (possible overlap) | 733 | 49 / 52 = 94.2% | 33.1% | 41.8% | 17.6% |
| VitaminC (not in training) | 410 | 8 / 14 = 57.1% | 9.6% | 33.8% | 13.3% |

On held-out VitaminC, recall is much lower and false flags on unrelated pairs much rarer: the model is
more trigger-happy in-domain. The false-flag rate on *ambiguous* pairs stays at roughly a third on both
sources, so the reference-ambiguity failure does not come from training overlap. (VitaminC has only 14
gold contradictions, so its recall figure is very uncertain.)

**The shared-entity gate, reconsidered.** On the four sample documents it blocked both true positives.
On RefNLI:

| Gate | Precision | Recall | False flags |
|---|---|---|---|
| none | 16.8% | 86.4% | 282 |
| shared entity token (`require_shared_entity`) | **26.2%** | **83.3%** | **155** |
| word overlap, Jaccard > 0.15 (the paper's heuristic) | 25.8% | 50.0% | 95 |

The entity gate cuts false flags by 45% and loses 2 of 57 true contradictions. The word-overlap gate
loses half of them, and is bad on both datasets.

The two entity-gate results disagree because of how referents are named. **63 of RefNLI's 66
contradictions (95%) share a capitalised entity token** -- they are Wikipedia claims about named
people, films and channels ("Aldous Huxley", "ChuChu TV"). **Neither** of our two caught document
contradictions does: their shared referent is a common noun phrase ("industrial sensor revenue",
"enterprise customers"). The gate works when referents are proper names and fails when they are
ordinary nouns. Research papers name their systems and datasets (RACN, MIMIC), so they may behave more
like RefNLI; business reports and transcripts behave like our samples. That can only be settled by an
evaluation set drawn from the target document type.

**Limits of this evaluation.** RefNLI pairs are a Wikipedia sentence against a claim, not two sentences
from one document, so these numbers measure the failure mode, not our pipeline's rate on real documents.
The paper's evaluation protocol was not reproduced (it scores one direction; Stage 3 averages both), so
the precision comparison with T5-Large shows the scale, not a like-for-like result.

### 9.6 Long-document summarization architectures (added 2026-09-19)

Four papers on how long or multi-part inputs are summarized. Unlike 9.1-9.3, these were checked only
against their published abstracts and landing pages (ACL Anthology, OpenReview, arXiv), not re-read in
full, so the connections below are at the level of design, not of measured results.

**Ou & Lapata (2025), "Context-Aware Hierarchical Merging for Long Document Summarization"** (Findings of
ACL 2025). The design reference for our Stage 2 (named in `anchored_summarization.py`, but not cited
here until now). Their finding is that enriching *hierarchical merging* with source context reduces the
hallucinations that recursive merging amplifies, measured with Llama 3.1. We applied the idea at a
different point -- context retrieved from other sections while summarising each section -- and with a
non-instruction model, DistilBART, which cannot be told that the context is background; we measured
it either ignoring the context or leaking it into the summary (Stage 2 docstring, `drop_context_leaks`).
Their method operates at the merge step, which our Stage 4 originally ran with no source context, and
Stage 4 is where we measured sections being dropped. **Tested at the merge step (section 14): no effect.**
With `merging.context_anchoring` on (each merged block's own top-2 source sentences added, up to 200
tokens), abstractive merging on sample 05 still cited 2, 4 and 3 of 20 sections at group sizes 3, 5 and 8 --
identical to without context -- and copied none of the 110 context sentences it was given. The most likely
reason is the same as in Stage 2: DistilBART is not instruction-tuned and cannot be told what the context
is for, whereas Ou & Lapata measured their gain with Llama 3.1. The idea is sound; our summarizer cannot
use it. The setting stays in the code, OFF.

**Kim & Kim (2025), "NexusSum: Hierarchical LLM Agents for Long-Form Narrative Summarization"** (ACL
2025). A multi-agent LLM pipeline for books and scripts, with a dialogue-to-description preprocessing
step that rewrites character dialogue into narrative prose before summarisation. The preprocessing
step speaks to our transcript findings: speaker labels (stripped before NLI), reporter introductions
posing as claims (11.2) and DistilBART inventing news-style attributions such as "President Obama says"
(11.3) are all symptoms of feeding dialogue to a model trained on news prose. A normalisation step
before Stage 2 is the principled version of our after-the-fact guards. The multi-agent LLM part needs
models far larger than our CPU budget allows.

**Liu & Lapata (2019), "Hierarchical Transformers for Multi-Document Summarization"** (ACL 2019). Encodes
paragraphs from MULTIPLE documents with inter-paragraph attention, after a learned ranker selects the
most salient paragraphs. Two contrasts with our design: our cross-part interaction is retrieval
(cosine top-k in Stage 2, candidate selection in Stage 3), not learned attention; and we have no
salience ranker -- every section is summarised and every claim is a Stage 3 candidate, which is why
Stage 3 pair counts and false flags grow with document size (12.2). It is also the natural reference
architecture for a multi-document extension (IDEAS.md, idea 2).

**Zhang et al. (2026), "Text Summarization via Global Structure Awareness" (GloSA-sum)** (ICLR 2026).
Builds a semantic-weighted graph from sentence embeddings, uses persistent homology (topological data
analysis) to find the document's core semantic and logical structure, locks it into a **protection
pool** that pruning may not delete, and prunes the rest iteratively with cheap proxy importance
scores; very long documents are segmented first and integrated globally. The protection pool is
conceptually similar to, but more general than, our `summarization.protect_cross_referenced`: ours
protects only sentences that some OTHER section's Stage 2 retrieval reached for (a local,
pairwise signal, added because both sides of sample 01's contradiction were being compressed away),
while theirs protects a document-wide structural backbone chosen from the geometry of the whole
sentence graph. Their segment-then-integrate strategy parallels our Stage 1 -> Stage 4 path. A cheap
approximation of the backbone idea is sketched in IDEAS.md, idea 1; the topological method itself is
not implemented.

## 10. Evaluation on real documents (labelled pair set)

Sections 8 and 9 rest on four synthetic documents and on RefNLI, whose pairs are a Wikipedia sentence
against a claim. This section uses the pair set in `data/eval/` (see its `MANIFEST.md`):
- **64 natural pairs:** real cross-section sentence pairs sampled by the pipeline's own candidate
  selection, all labelled not-contradiction.
- **36 constructed contradictions:** a verbatim real sentence against a hand-written counter-claim.

The sentences come from four real public documents, one per type in the brief: a research paper,
FOMC minutes, a CUAD contract and an FOMC press-conference transcript. Every label was written and
hashed before any NLI model saw the pairs; `scripts/eval_pairs.py` verifies the hashes before it runs.
Two labels were re-checked after evaluation (research-n06, contract-n10) and stand.

### 10.1 Pair-level results (`scripts/eval_pairs.py`, full output in `outputs/eval_pairs.txt`)

| Configuration | Contradictions caught | False flags on 64 real pairs |
|---|---|---|
| NLI rule alone | 24 / 36 (67%) | 2 (3.1%) |
| **default** (+ similarity floor, non-claim filter, comparison filter) | **23 / 36 (64%)** | **2 (3.1%)** |
| + shared-entity gate instead of the comparison filter | 12 / 36 (33%) | 2 (3.1%) |

**Recall by kind of contradiction (default):**

| Category | Caught |
|---|---|
| numeric_direct (two values for one quantity) | 5 / 5 |
| explicit_opposition | 12 / 14 |
| implicit | 4 / 5 |
| quantifier | 2 / 3 |
| **derived_numeric** (arithmetic or unit conversion) | **0 / 7** |
| **causal_denial** | **0 / 2** |

The derived-numeric result generalises the sample-04 finding from one case to seven, across all four
real documents and several kinds of reasoning:
- hours to days ("72 hours" vs "five days"), scored 0.14
- days to months ("sixty-day term" vs "three months"), 0.18
- subtraction ("from 7 to 2.2 percent" vs "fallen by roughly three points"), 0.00
- reading a ratio ("vacancies-to-unemployment ratio of 1.1" vs "fewer vacancies than unemployed"),
  0.00
- the sign of a difference ("core 3.2, headline 2.5" vs "core 0.7 below headline"), 0.08

This is the pattern Mahendra et al. (2025) report (section 9.1): the model catches every *direct*
numeric conflict (5/5) and none that needs a computation (0/7).

**Hard negatives were handled well.** None of the eleven natural look-alikes was flagged:
- 50 basis points vs 1/2 percentage point
- 2.2 percent now vs 2 percent by 2026
- reversed contract roles
- a reporter quoting the Chair and adding "even though"
- quantifier scope ("almost all" vs "most")

### 10.2 The entity-gate question is answered for these document types

| Contradictions whose subject is... | Share an entity token | Caught, default | Caught with entity gate |
|---|---|---|---|
| a proper name, acronym or figure (18) | 18 / 18 | 10 | 10 |
| an ordinary noun phrase (18) | 3 / 18 | 13 | **2** |

On real documents the shared-entity gate halves recall (23 to 12) and removes **no** false flags,
because both false flags do share entity tokens (acronyms in the research paper, the party names in
the contract). The damage falls on contradictions about ordinary things -- "the labor market",
"mortgage rates", "inflation" -- which is most of what the press conference and the minutes talk
about (transcript: 6/9 caught by default, 1/9 with the gate). RefNLI's 45% false-flag reduction
(section 9.5) came from Wikipedia claims about named entities and does not carry over. **Verdict: keep
`require_shared_entity` off.** This supersedes the "cannot decide" assessment in section 9.4.

### 10.3 The comparison-wording filter does not earn its place on real text

On the 100 pairs it removed no false flags and cost one true contradiction: research-c07, scored
**0.99**, whose real sentence says models "perform generally worse on recognizing contradictions
compared to recognizing entailments". Across the three evaluations:

| Evaluation | False flags removed | True contradictions lost |
|---|---|---|
| 4 synthetic samples | 5 | 0 |
| RefNLI (1,143 pairs) | 7 of 282 | 1 |
| real pair set (100 pairs) | 0 | 1 |

It only helped on text written with explicit baselines in mind. Recommendation: switch it off by
default (not yet done; a behaviour change awaiting approval).

### 10.4 Direction aggregation re-checked on real pairs

Computed from the saved scores, without re-running the model. Switching `mean` to `max` catches three
more contradictions (27/36) but adds three false flags, raising the false-flag rate from 3.1% to
7.8%. Real documents contain very few contradictions, so doubling the false-flag rate costs more than
three extra catches gain. `mean` stays.

### 10.5 What a user actually sees: the full pipeline on the same four documents

A 3% per-pair false-flag rate sounds tolerable until it multiplies by the number of pairs Stage 3
compares in a whole document. `scripts/run_real_documents.py` runs the default pipeline on the four
source documents. They are published, edited documents and presumably internally consistent, so nearly
every flag should be false.

| Document | Pages | Pairs compared (3a + 3b) | Flags | Flag rate | Runtime |
|---|---|---|---|---|---|
| research (RefNLI paper) | 10 | 1,955 | **123** | 6.3% | 9.9 min |
| financial (FOMC minutes) | 12 | 2,355 | **43** | 1.8% | 11.6 min |
| contract (CUAD) | 6 | 1,118 | **36** | 3.2% | 6.9 min |
| transcript (FOMC press conference) | 19 | 8,248 | **467** | 5.7% | 16.8 min |

The transcript is the worst case: 90 sections and 336 claims. 65 of its 467 flags involve a reporter introduction such as "Jeanna Smialek, New York Times." or "Elizabeth Schulze with ABC News." (these were counted by matching a list of news outlets). Such lines are not claims, but they pass the four-word `min_claim_words` filter and score P(contradiction) near 1.00 against almost anything. Another 67 flagged pairs were left unscored because they exceeded the `max_resolutions` budget of 200.

A sample of twelve of the research paper's 123 flags (the top five by score and seven at random) was
read by hand. **All twelve are false.** They are the reference-determinacy failure of section 9.3,
occurring *inside a single document*:
- A paper describes many different activities with the same subject, "we", so "We conduct an
  experiment with the ChaosNLI dataset" vs "We construct the RefNLI benchmark" scores 0.99.
- "The evaluation results are shown in Table 5" vs "Table 3 shows the classification results" --
  different tables, read as the same one.

**Resolution removed real content.** Most flags end `unresolved` and are only marked DISPUTED in the
evidence panel. But 18 claims won no support contest and were deleted (`action: remove`): contract 4,
financial 1, research 1, transcript 12. Each was checked by hand against its source document:

| Removed claim | Count | Removal justified? |
|---|---|---|
| Faithful to the source (the assignment clause, the FOMC policy statement, the paper's abstract, "the U.S. economy is in a good place", "we have a dual mandate", ...) | 11 | **no** |
| Real content with a wrong speaker tag added by DistilBART ("Chairwoman says...", a reporter credited with Powell's words, "she says") | 3 | no: the tag is wrong, the content is right |
| Generated text with no source claim behind it (a garbled contract sentence; a reporter's *question* "wouldn't the Committee already be too late?" turned into "Committee chairman says it's not too late to avoid a recession"; a reporter's question credited to Powell) | 3 | yes |
| A real misstatement by the summarizer ("seven of them wrote down three or more cuts"; the Chair corrected himself to "17 of the 19") | 1 | yes |

So 4 of the 18 removals were justified and **14 deleted correct information**. (The first version of
this table counted the "seven of them" claim as faithful; checked against the transcript, it is not.)
None of the 18
"winning" claims actually contradicts the claim it removed; every removal started from a false flag,
so even the four justified removals were right by coincidence.
The support score does not rescue this. Faithful paraphrases often score low support against their
own section (0.01 for the assignment clause and for "we have a dual mandate ... strong commitment",
both close to verbatim), so a false flag plus a lopsided support contest deletes true text. One
transcript sentence ("the rate at which those things happen will really depend on how the economy
performs") was *kept* in one pair and *removed* in another. On real documents `action: remove` does
more harm than good. The volume of flags is a review burden, but these removals corrupt the output.

**This is the pipeline's real limitation on real documents, and it is the opposite of what the
synthetic samples suggested.** Per pair, the model is reasonably precise (3% false flags). But a
10-page paper needs about 2,000 comparisons, so it still produces over a hundred flags. With genuine
contradictions this rare, even a good per-pair rate leaves the flag list almost entirely false. The
per-document false-flag rates (1.8-6.3%) match the pair set's per-document rates, so the two
measurements agree.

**Runtime correction.** Real prose has much longer sentences than the synthetic benchmark of section
5, and NLI cost grows with sentence length: 7-12 minutes per 6-12 page document, against 4.4 minutes
for the 15-page synthetic one. The 200-page projection in section 5 (about 45 minutes) is therefore
an underestimate. At the measured rate of roughly a minute per page, a 200-page real document would
take **around three hours** on this CPU.

### 10.6 What this changes

1. **Recall** is limited by reasoning the model cannot do (derived numbers 0/7, causal denials 0/2),
   not by the pre-filters.
2. **Usability** is limited by volume: tens to a hundred-plus flags per real document, nearly all
   false. Thresholds and keyword filters will not fix that (sections 8.4, 9.4, 10.3). What would:
   ranking flags so a reviewer sees the few most likely ones first, or restricting Stage 3 to claims
   that are structurally comparable (same metric and period, same contract obligation).
3. The shared-entity gate should stay off, the comparison filter should probably be switched off,
   and `mean` aggregation should stay.
4. **Resolution should not delete on real documents** (10.5): 14 of 18 removals deleted correct
   content. The safe default is `action: flag`, which keeps every claim and marks it DISPUTED.
5. **Speaker and reporter introductions are not claims** but pass `min_claim_words` (10.5): 65 of
   the transcript's 467 flags.


## 11. Fixes after the real-document evaluation (claim filter, attribution guard, topic gate)

### 11.1 Default changes (approved after section 10)

- `contradiction.action: remove -> flag`: on the four real documents 14 of 18 removals deleted correct
  content (10.5). Resolution is still computed and reported; nothing is deleted.
- `contradiction.skip_comparative_framing: true -> false` (10.3).
- `models.batch_size: 16 -> 8`: 29.6 vs 32.4 ms per NLI pair on this CPU, identical scores.

### 11.2 Non-claim filter: a fragment with neither a verb nor a number is not a claim

Reporter introductions ("Jeanna Smialek, New York Times.") passed the four-word filter and scored
~1.00 against almost anything: 65 of the press-conference transcript's 467 flags. New rule
(`require_verb_or_number`, POS tagger), applied only to sentences under 10 words
(`verbless_fragment_max_words`), because without that limit the tagger dropped two real claims
("This paper studies the impact..." read as a noun phrase; an ALL-CAPS clause). All-caps text is
lower-cased and curly apostrophes normalised before tagging. On the four real documents it drops
only reporter introductions, headings and signature blocks. Transcript flags: 467 -> 410.

### 11.3 Invented speaker attributions (Stage 2)

DistilBART rewrote first-person transcript speech in news style and invented tags: "..., he says.",
"Chairwoman says ...", "President Obama says" (Obama is not in the document) -- 18 of 336 transcript
summary sentences, 0 in the other three documents. `summarization.strip_invented_attribution` removes a
tag only when its text is absent from the section source and its subject looks like a person; each
removal is recorded in the section's notes. Not yet re-measured on a full run.

### 11.4 Shared-content-word ("topic") gate: measured, left OFF

Rule and decision criterion were written down before measuring: turn it on only if RefNLI recall drops
by at most 5 points AND real-document flags fall by at least 25%.

| Evaluation | Gate off | Gate on |
|---|---|---|
| RefNLI, raw rule: true contradictions caught | 57 / 66 | 57 / 66 |
| RefNLI, raw rule: false flags | 282 | 252 (-11%) |
| real documents, flags (research / minutes / contract / transcript) | 116 / 46 / 38 / 410 = 610 | 99 / 38 / 37 / 268 = 442 (-28%) |

It passes both criteria, narrowly and mostly on the transcript (-35%; the other documents -3% to
-17%), and the criterion did not say whether 25% applied to the total or per document. The planned
document-level check (3 contradictions planted into each real document) was stopped after two of four
documents when sample 05 arrived, so the gate stays OFF until that check is finished. On sample 05
(section 12) it would remove 9 of 92 false flags and none of the true ones.

## 12. Sample 05 (medium research paper): two new findings

Sample 05 (`data/samples/05_medium_paper_graphfault.txt`, ~1,900 words, 20 sections) was supplied by
the user without ground truth. Three contradictions were labelled from a full read before any run
(`GROUND_TRUTH.md`, awaiting confirmation). Default run: C2 caught by 3b (P=0.91, rank 25 of 95 flags;
a second flag at 0.52), C3 caught only by the opt-in source diagnostic (0.97; both sides compressed out
of their summaries), C1 missed everywhere (0.15). 95 flags in total, 92 false.

### 12.1 Finding 1: an explicit numeric conflict is missed when the numbers are phrased differently

C1: "The automotive dataset spans eighteen months of sensor readings from a stamping line, including two
hundred and forty individual sensors." (4.1) vs "We also note that the automotive dataset, spanning
twenty-four months of continuous stamping line operation, provided a particularly stable basis for
evaluating long-term drift ..., since longer observation windows make gradual equipment aging easier to
distinguish from short-term noise." (6). Both sentences survived into their summaries verbatim; the pair
WAS compared (ranked first by similarity). It scored 0.151.

Removing one piece at a time (`scripts/probe_dilution.py`, `outputs/probe_dilution.txt`;
P(contradiction), A = the 4.1 sentence, B = the Discussion sentence):

| Step | A->B | B->A | mean |
|---|---|---|---|
| 1 as written | 0.231 | 0.072 | 0.151 |
| 2 B without the trailing "since longer observation windows ..." clause | 0.203 | 0.078 | 0.140 |
| 3 ... and without "We also note that" | **0.949** | 0.084 | 0.516 |
| 4 B = "The automotive dataset spans twenty-four months of continuous stamping line operation." | 0.987 | 0.047 | 0.517 |
| 5 B bare | 0.996 | 0.004 | 0.500 |
| 6 A without ", including two hundred and forty individual sensors" | 0.995 | 0.010 | 0.502 |
| 7 both bare | 0.996 | **0.991** | 0.994 |

The trailing clause, which looked like the obvious culprit, changes nothing (0.151 -> 0.140). One
direction recovers when "We also note that" goes; the other only when A loses "of sensor readings".

Controls, each changing one thing relative to the bare pair:

| Control | tokens A/B | A->B | B->A | mean |
|---|---|---|---|---|
| LENGTH: B restates its own fact to 30 tokens, no new proposition | 9/30 | 0.971 | 0.997 | 0.984 |
| SAME extra clause on both sides | 26/28 | 0.997 | 0.995 | 0.996 |
| extra clause on B only / on A only | | 0.995 / 0.996 | 0.990 / 0.874 | 0.993 / 0.935 |
| number backgrounded in a participle ("..., spanning twenty-four months, provided ...") | 9/20 | 0.992 | 0.990 | 0.991 |
| A: "eighteen months **of sensor readings**" (qualifier ON the quantity) | 12/11 | 0.995 | **0.016** | 0.506 |
| A: same words as a separate clause ("... eighteen months and contains sensor readings ...") | 17/11 | 0.997 | 0.916 | 0.956 |
| SAME qualifier on both sides | 16/18 | 0.996 | 0.993 | 0.995 |
| different qualifiers ("of sensor readings" vs "of continuous operation") | 12/14 | 0.987 | 0.020 | 0.504 |
| B: "We also note that the automotive dataset spans twenty-four months." | 9/15 | **0.629** | 0.994 | 0.811 |

Generality, three synthetic explicit conflicts (pilot enrolled 120 vs 85 patients; warehouse holds 4,000
vs 3,200 pallets; contract runs three vs five years):

| Variant | pilot | warehouse | contract |
|---|---|---|---|
| bare (mean) | 0.999 | 0.989 | 0.999 |
| redundant padding to ~28 tokens (mean) | 0.997 | 0.993 | 0.996 |
| qualifier on the quantity, hypothesis direction | 0.986 | **0.039** | **0.012** |
| same words as a separate clause, same direction | 0.997 | 0.990 | 0.993 |
| "We also note that ...", hypothesis direction | **0.009** | **0.005** | **0.336** |
| same, mean of both directions | 0.504 | **0.498 (missed)** | 0.667 |
| causal / hedge / parenthetical / trailing framing on one side: hypothesis directions below 0.5 | 1 of 8 | 1 of 8 | 2 of 8 |
| same framing on BOTH sides (mean, 4 frames) | >= 0.999 | >= 0.987 | >= 0.997 |

What this establishes:
1. **Not length.** Padding to 26-30 tokens with the same content, or the same extra clause on both
   sides, never drops a pair below 0.98.
2. **Not generic "competing content" either.** Unrelated framings on one side lowered the hypothesis
   direction below 0.5 in only 4 of 24 cases, and the other direction always held, so the mean stayed
   above threshold in all 24.
3. **Two specific mechanisms, both acting on the HYPOTHESIS side only** (a long or qualified premise is
   harmless):
   - **Quantity qualifier:** a phrase attached to the number that narrows WHAT is measured ("18 months
     OF SENSOR READINGS", "4,000 pallets OF DRY GOODS", "three years OF ON-SITE SUPPORT") makes the model
     read the two numbers as measuring different things -> neutral (0.01-0.04). The same words elsewhere
     in the sentence do not (0.92-0.99). Reproduced in 2 of 3 synthetic facts; the exception ("120
     patients FROM THE NORTHERN CLINICS") qualifies where the patients came from, not the count. This is
     reference determinacy (section 9.3) at the level of a quantity rather than an entity.
   - **Reporting frame:** "We also note that X" as the hypothesis -> 0.005-0.63. The model treats it as a
     claim about what the authors note, which the premise does not address. Reproduced in 3 of 3.
4. **Why C1 is missed rather than borderline:** each mechanism kills ONE direction, so on its own the
   mean lands at ~0.50 -- a coin flip at the 0.5 threshold (0.498 missed, 0.504 caught). C1 has one
   mechanism on each side ("of sensor readings" in A, "We also note that" in B), so both directions
   collapse. Max-aggregation would not rescue it (0.231).
5. **Distinct from the date-arithmetic limitation (section 8.1).** Sample 04 needed a value to be
   DERIVED (24 months from a start date -> an end date). Here nothing needs deriving: both values are
   stated outright and the bare pair scores 0.994. The conflict is lost to how each number is phrased.

Tests pinning this behaviour: `tests/test_nli_dilution_slow.py` (6 tests, real NLI model).

Not fixed (investigation only). Candidate fixes, in order of cost: strip reporting frames ("We also note
that", "We find that", ...) before NLI, as speaker labels already are; decompose sentences into atomic
claims before comparison, so "spans eighteen months" is compared without its qualifier -- which risks the
opposite error, since sometimes the qualifier IS the difference.

### 12.2 Finding 2: 95 flags on a 4-page paper -- scale, not a harder document

Same current default config for all five documents (`scripts/analyze_false_flags.py`,
`outputs/false_flag_analysis.txt`):

| Document | sections | claims | pairs compared (3a+3b) | flags | true | false | false per 100 pairs |
|---|---|---|---|---|---|---|---|
| 01 planted | 5 | 13 | 76 | 7 | 1 | 6 | 7.9 |
| 02 clean | 6 | 18 | 175 | 9 | 0 | 9 | 5.1 |
| 03 subtle | 5 | 23 | 105 | 1 | 1 | 0 | 0.0 |
| 04 contract | 8 | 19 | 153 | 8 | 0 | 8 | 5.2 |
| 01-04 pooled | | | 509 | 25 | 2 | 23 | 4.5 |
| **05 paper** | 20 | 48 | **1,621** | 95 | 3 | 92 | **5.7** |

(The "14 false positives" quoted from earlier sections was under the old defaults, with the comparison
filter on; under the current defaults the four small samples give 23.)

Controlling for how similar the compared pairs are (false flags / pairs within each cosine band):

| cosine band | 0.2-0.3 | 0.3-0.4 | 0.4-0.5 | 0.5-0.6 | 0.6+ |
|---|---|---|---|---|---|
| 01-04 pooled | 1% (2/258) | 5% (7/133) | 14% (10/72) | 10% (3/30) | 6% (1/16) |
| 05 | 3% (18/591) | 5% (28/547) | 10% (30/313) | 11% (14/127) | 4% (2/56) |

Pairs with cosine >= 0.5: 9% of compared pairs in 01-04, 11% in 05. **The per-pair false-flag rate is
essentially the same.** The 95 flags come from comparing 3-20x more pairs (48 claims, each compared with
up to 30 claims and 30 sibling source sentences), not from a document type the model handles worse. It
is the same weakness at larger scale -- which also means it will keep growing with document length.

What the 92 false flags are (explicit rules, first match wins, printed by the script):

| Pattern | false flags | mean P | removed by: comparison filter | entity gate | topic gate |
|---|---|---|---|---|---|
| A referent lost in Stage 2 compression | 18 | 0.67 | 0 | 18 | 1 |
| B method vs its own ablation | 23 | 0.75 | 1 | 23 | 2 |
| C method vs a baseline or prior work | 26 | 0.69 | 6 | 25 | 5 |
| D different dataset / quantity, same numeric structure | 21 | 0.83 | 5 | 21 | 0 |
| E other (different parts of the method) | 4 | 0.82 | 1 | 4 | 1 |
| *true flags (C2, incl. one borderline Conclusion pair)* | *3* | *0.80* | *0* | ***3*** | *0* |

Examples (score, then the two sentences):
- A, 0.998: "In this work, we introduce GraphFault, a framework that jointly learns a dynamic sensor graph
  ..." vs "The network uses a fixed, manually specified sensor adjacency matrix based on the equipment's
  wiring diagram." The source says "The THIRD [baseline] is a static graph attention network that uses
  ..."; Stage 2 dropped the referent, and that one summary sentence produced 18 false flags.
- B, 0.998: the same introduction sentence vs "Removing the dynamic graph structure learning module and
  replacing it with a fixed graph ... reduces average lead time by roughly two point one hours".
- C, 0.938: "GraphFault is evaluated on three proprietary industrial datasets ..." vs "We compare
  GraphFault against four baselines."; 0.936: "GraphFault is a hybrid retrieval-augmented framework ..."
  vs "The fourth is a transformer-based multivariate time series forecasting model that does not
  explicitly encode graph structure."
- D, 0.987: "GraphFault achieves a fifteen percent improvement ... while maintaining a false alarm rate
  below two percent." vs "The LSTM autoencoder exhibits a false alarm rate above five percent on the
  semiconductor dataset."; 0.982: "The semiconductor dataset spans twelve months ..., including three
  hundred and ten sensors." vs "The automotive dataset contains two hundred and forty sensors".
- E, 0.823: "We formulate early fault detection as a sequence-to-sequence prediction problem over a
  graph." vs "We learn edge weights as a function of recent windowed correlation between sensor pairs."

Every one of patterns A-D is a **reference mismatch**: the two sentences are about different systems
(proposed vs ablated vs baseline), different datasets, or -- in A -- a referent the summarizer erased.
Note the symmetry with Finding 1: the model calls "automotive: 240 sensors" vs "semiconductor: 310
sensors" a contradiction (0.98), but "18 months of sensor readings" vs "24 months" compatible (0.01). It
decides whether two numbers describe the same thing from the surface phrasing of the quantity, not from
what the sentence is about.

### 12.3 Assessment: keyword filter or referent check?

- **Extending the comparison keyword list is not sufficient.** Only 13 of 92 false flags contain any
  comparison wording. Adding "Removing", "The first/second/third is" and similar phrases would cover
  patterns B and C on THIS paper -- a list tuned to one document's phrasing, which section 10.3 showed
  does not transfer, and which would also skip real contradictions phrased that way.
- **The existing shared-entity gate is not the referent check and must not be used here.** It would
  remove all three true flags: the paper spells its numbers out and names GraphFault at sentence starts,
  so the true pair shares no entity token. (It also removes nearly every false flag -- by removing
  nearly every pair.)
- **The topic gate** removes 9 of 92, no true flags: consistent with 11.4, small.
- **A referent (subject) check is now the most important line of work for precision**: patterns B+C+D
  are 70 of 92 false flags (76%). It has to decide what each claim is ABOUT (the proposed system, a
  named baseline or ablated variant, a specific dataset, a specific quantity), and it must be validated on
  pairs like C2, where the two sentences share a topic ("cross-plant transfer") but not a grammatical
  subject ("the learned sensor embeddings" vs "we"), so a naive subject match would drop it.
- **Pattern A (18 false flags, 20%) is a Stage 2 bug and has its own, cheaper fix**: a summary sentence
  that replaces a specific referent ("the third baseline") with a generic one ("The network") should be
  detected in Stage 2 (a referent-preservation check against the source sentence), not filtered in Stage 3.

Proposed order (nothing implemented): (1) strip reporting frames before NLI (cheap, recovers recall,
12.1); (2) Stage 2 referent preservation (removes the largest single cause, 12.2 A); (3) design and
validate a referent gate, measured on the labelled pair set, RefNLI and sample 05 before any default
changes.

## 13. Reporting-frame stripping (on) and a referent check (designed, NOT validated)

### 13.1 Reporting-frame stripping -- implemented, on by default

`contradiction.strip_reporting_frames: true`. Reporting frames ("We also note that", "We find that",
"Our results show that", "It is worth noting that", ...) are removed from both claims before NLI, as
speaker labels already are. Only the text the NLI model scores changes: candidate selection and the
reported claims are unchanged. Belief and expectation hedges ("we believe", "we expect", "results
suggest") are deliberately NOT stripped. Tests: `tests/test_reporting_frames.py`.

Effect, measured by re-running Stage 3 (plus the source diagnostic) on the saved Stage 2 output of all
five samples (`scripts/rerun_stage3.py`; the frames-off re-run reproduced the saved runs exactly, 120 flags):

| Document | flags off -> on | true | false | planted, default path |
|---|---|---|---|---|
| 01 planted | 7 -> 7 | 1 -> 1 | 6 -> 6 | caught (1.00) |
| 02 clean | 9 -> 9 | 0 | 9 -> 9 | -- |
| 03 subtle | 1 -> 1 | 1 -> 1 | 0 -> 0 | caught (0.61) |
| 04 contract | 8 -> 8 | 0 | 8 -> 8 | missed (date arithmetic, 8.1) |
| 05 paper | 95 -> 103 | 2 -> 3 | 92 -> 99 | C1 missed -> **caught (0.523)**; C2 0.98; C3 diagnostic only (0.97) |
| **Total** | **120 -> 128** | **4 -> 5** | **115 -> 122** | |

(Sample 05 also has one borderline flag in both runs -- Conclusion "generalize meaningfully across
manufacturing sites" vs the Limitations sentence -- counted in neither column.)

**Correction.** The first version of this table reported C1 "caught (0.84)" and 05 as "3 -> 5 true,
92 -> 98 false". That came from the evaluation matcher (cosine >= 0.6 to the planted sentences), which
accepted a FALSE pair -- semiconductor "spans twelve months" vs the automotive "twenty-four months"
sentence, scored 0.843 -- as C1. Checked pair by pair (`outputs/frame_diff_05.txt`), stripping added 8
flags on sample 05 and removed none: all 8 involve one of the three framed claims; **1 is true** (the C1
pair itself, at 0.523, just over the 0.5 threshold) and **7 are false**, 5 of them the un-framed
"automotive dataset, spanning twenty-four months" sentence against OTHER datasets' figures -- pattern D of
12.2. Removing the frame un-hides the number, and the reference problem then flags it against every
other dataset. Net: +1 true, +7 false. Kept on (the recall gain is real and the default was already
approved), but its precision cost is larger than first reported.

### 13.2 Grounded referent re-check -- designed, not validated, not wired in

Design and pass criteria were written before any measurement:
`data/eval/REFERENT_CHECK_PREREGISTRATION.md`. Code: `src/hcv_sum/referent.py` (not called by the
pipeline; no config key), unit tests `tests/test_referent.py`, validation script
`scripts/eval_referent_check.py`.

Idea: the author already said what each claim is about; compression lost it ("The third is a static graph
attention network that uses ..." -> "The network uses ..."). Each flagged claim is put back into its
referential context -- section title, the preceding source sentence, and the source sentence it came from
-- and the same detector is asked again with that context as the PREMISE only (premise-side additions
leave true contradictions intact, 12.1). No token, entity or grammatical-subject comparison. RefNLI cannot
validate it (no document context in its records); the pre-registered external set is the flags on the four
real documents.

**Validation against (a) the true C2/C3 flags, (b) the 18 referent-lost flags and (c) the real documents
has NOT been run.** Nothing about its effect is known yet, and no default depends on it.

### 13.3 Why it stopped

After a session restart, Windows Application Control began blocking a compiled scikit-learn DLL
(`sklearn/metrics/_dist_metrics`). sentence-transformers imports scikit-learn, so neither the embedder nor
the NLI model can load; every real-model measurement is blocked. Fast tests (fake models) are unaffected.
Resuming needs either the file allowed by the machine's policy, or loading both models directly through
`transformers` (a loader change that would first have to be shown to give identical embeddings and scores).

### 13.4 Idea 1 (centrality-based "backbone" protection): investigated, deprioritized

Offline check proposed in IDEAS.md, run before any prototype (`scripts/probe_centrality.py`,
`outputs/probe_centrality.txt`). Stage 1 source sentences of samples 03 (39 sentences) and 05 (73),
embedded with the pipeline's embedder; four graphs (kNN-5, kNN-10, kNN-20, dense cosine >= 0.1), each
with weighted PageRank (hubs) and betweenness (bridges). Percentile: share of the document's sentences
that are LESS central (100 = most central). The bar, stated in IDEAS.md beforehand: about the top 20%.

| Sentence | median over the 8 graph/measure pairs | best single one | pairs reaching 80 |
|---|---|---|---|
| 03 "every enterprise customer ... Helix Core" | 74 | 95 | 3 of 8 |
| 03 "roughly sixty enterprise accounts ..." | 53 | 85 | 1 of 8 |
| 05 abstract "different equipment vendors" (C2) | 49 | 64 | 0 of 8 |
| 05 5.3 "despite differing equipment vendors" (C3) | 63 | 82 | 2 of 8 |
| 05 Limitations "only evaluated ... similar vendors" (C2/C3) | **8** | 25 | 0 of 8 |
| 05 Limitations "not yet evaluated ... different equipment" (C2) | 26 | 99 | 1 of 8 |
| 05 4.1 "eighteen months" (C1) | 58 | 86 | 1 of 8 |
| 05 Discussion "twenty-four months" (C1) | 44 | 95 | 1 of 8 |

- The Limitations sentence that both C2 and C3 depend on is among the LEAST central sentences in every
  graph (3rd-25th percentile), as predicted in IDEAS.md.
- The others are middling (median 44-74). Each clears the bar in some graph, but a prototype has to
  commit to ONE graph and measure, and no single one lifts the contradiction sentences consistently: the
  best (betweenness on kNN-5) puts 4 of 8 above 80, none of them the four vendor sentences.
- Eight sentences in two documents is weak evidence in either direction; it is enough to show that
  "central" and "carries a contradiction" are not the same property here.

**Verdict: deprioritized, not prototyped.** Protection would add Stage 3 claims (and false flags,
which scale with pairs) without reliably protecting the sentences whose loss causes the recall misses.

### 13.5 Graph proximity as a referent check: measured, not separable -- line closed

Before building a graph-based "same referent" filter, the question was whether TRUE pairs are
measurably closer than false flags at all (`scripts/probe_referent_proximity.py`,
`outputs/probe_referent_proximity.txt`). Sample 05, current flags (frames stripped): 4 true pairs
labelled strictly -- C1, the two C2 flags, and C3 as a source pair, each grounded to the planted
sentences themselves; the borderline Conclusion flag excluded -- against 98 false flags. Each claim is
grounded to its source sentence (as in `referent.py`); measures on the kNN-10 sentence graph:

| Measure | true pairs | false flags (min / q25 / median / q75 / max) | threshold keeping all true pairs removes |
|---|---|---|---|
| cosine | 0.72, 0.55, 0.41, 0.48 | 0.21 / 0.32 / 0.39 / 0.47 / 0.63 | 54 / 98 (9 / 17 pattern A) |
| hops | 1, 1, 1, 1 | 1 / 1 / 2 / 2 / 3 | 60 / 98 (10 / 17) |
| weighted path distance | 0.28, 0.45, 0.59, 0.52 | 0.30 / 0.55 / 0.79 / 0.94 / 1.37 | 65 / 98 (10 / 17) |
| neighbourhood overlap (Jaccard) | 0.25, 0.25, 0.43, 0.54 | 0.00 / 0.11 / 0.18 / 0.25 / 0.54 | 59 / 98 (9 / 17) |

- **Not cleanly separable.** Every true pair lies inside the false-flag range on every measure. The
  weakest true pair (a C2 flag, cosine 0.41) sits at the false flags' median (55th percentile on
  cosine), and ~40% of false flags are as close (1 hop) as every true pair.
- The "removes" column is a BEST case: the threshold is fitted to these same 4 positives, which says
  nothing about the next document. It also barely helps the target: 7-8 of the 17 referent-lost flags
  (pattern A) survive every threshold.
- An outside check was not possible: the labelled pair set's contradictions are hand-written
  counter-claims that reuse the real sentence's wording, so all 36 have cosine >= 0.44, higher than the
  natural C2 pair (0.41); they cannot show what a proximity cut would lose.
- This is the same overlap already found for NLI scores (8.4) and entity tokens (10.2): the property
  that separates true from false flags -- whether the two claims are about the same thing -- is not
  visible in surface or embedding proximity.

**Verdict: exhausted.** No graph-proximity filter will be built. The NLI-based grounded re-check of 13.2
is a different mechanism (it asks the detector again with the author's context rather than thresholding
distance) and remains designed but unvalidated.

## 14. Context-anchored merging (Ou & Lapata 2025, at the merge step): measured, no effect

**What was built.** `merging.context_anchoring` (default off). In abstractive merging, each group of
blocks being fused is given source sentences alongside it: per block, the `context_top_k_per_block` (2)
source sentences most similar to that block, above `context_min_similarity` (0.35, Stage 2's floor),
skipping sentences the block already states, capped at `context_max_tokens` (200, Stage 2's cap). The
group budget shrinks by that cap only when the setting is on. Retrieval is per block, fixed before
measuring, so that every block in a group -- not only the first, which DistilBART favours -- is anchored.
Extractive merging makes no model call, so the setting does not apply to it. Code: `merging.py`
(`retrieve_merge_context`); tests: `tests/test_merge_context.py` (7).

**How it was measured.** `scripts/eval_merge_context.py --only 05`, on sample 05 (20 sections; the
31-section synthetic benchmark and samples 01-04 were not run). Abstractive merging with and without
context at group sizes 3, 5 (the large-document default) and 8, same Stage 1-3 output throughout. Coverage
= sections cited by at least one final sentence (Stage 5). DistilBART uses beam search, so the text is
deterministic; timings are not (below).

| Configuration | sections cited | final sentences | merge-introduced contradictions | context sentences given / copied into output | Stage 4 seconds | peak MB |
|---|---|---|---|---|---|---|
| extractive (reference) | 20 / 20 | 46 | -- | -- | 0.1 | 2657 |
| abstractive g3, no context | 2 / 20 | 3 | 0 | -- | 51.3 | 2793 |
| abstractive g3, WITH context | **2 / 20** | 3 | 0 | 57 / 0 | see below | 2831 |
| abstractive g5, no context | 4 / 20 | 5 | 0 | -- | 40.8 | 2818 |
| abstractive g5, WITH context | **4 / 20** | 5 | 1 | 30 / 0 | 47.0 | 2844 |
| abstractive g8, no context | 3 / 20 | 5 | 0 | -- | 42.7 | 2858 |
| abstractive g8, WITH context | **3 / 20** | 4 | 0 | 23 / 0 | 48.0 | 2897 |

- **Coverage: no change at any group size.** Context-anchored merging does not fix the measured loss.
  The merger copied none of the 110 context sentences it was given, and the output length barely moved
  (56 -> 61, 91 -> 90, 98 -> 93 words): DistilBART effectively ignored the added text, as it did with
  Stage 2's context (section 2, Stage 2 verdict).
- **Merge-introduced contradictions: 0 -> 0, 0 -> 1, 0 -> 0.** Too few to call a change in either
  direction. The earlier figure of 8 introduced contradictions came from the 31-section benchmark, which
  was not re-run.
- **Contradiction detection: unaffected by construction.** Stage 4 runs after Stage 3, so Stage 3's flags
  are identical with the setting on or off. On sample 05 (current defaults, strict labels): C1 caught by
  3a, C2 by 3b, C3 only by the source diagnostic; 100 false flags over 1,621 compared pairs (6.2 per 100).
  Stage 5 statuses were unchanged (1 unsupported sentence in every configuration).
- **Cost: small, noisy.** +6.2 s at g5 (+15%) and +5.3 s at g8 (+12%); peak memory +26 to +39 MB. The g3
  run with context first took 1,536 s; re-timed twice it took 68.6 s and 113.9 s against 63.6 s and 56.7 s
  without context. The 1,536 s did not reproduce and is treated as an interruption of the machine, not a
  property of the method; the g3 cost is between +5 s and +57 s on this machine.

**Verdict: abandon for this summarizer.** No coverage gain, no measurable effect on the output, some
added cost. The setting stays in the code, off, because the result is about DistilBART rather than the
technique: Ou & Lapata report their gain with an instruction-tuned LLM (Llama 3.1), which can be told that
the context is supporting evidence. If the Stage 4 summarizer is ever replaced by such a model, this is
the first thing to re-test. It is not recommended with the current model.

## 15. Context-anchored merging with an instruction-tuned merge model: measured, negative -- closed

**Why.** Section 14 found context-anchored merging had no effect with DistilBART, which is not
instruction-tuned and cannot be told what the context is for. Ou & Lapata measured their gain with an
instruction-tuned LLM (Llama 3.1). This retests the idea with the most capable instruction-tuned model
that fits this CPU.

**What was built** (all off by default; single-document defaults unchanged):
- `models.merge_summarizer`: the Stage 4 model, configurable independently of the Stage 2 summarizer.
  Empty (default) = the same model as Stage 2.
- `merging.merge_prompt: instruct`: instead of handing the merger plain text, it is given an explicit
  instruction -- merge these section summaries into one, keep every distinct fact, and use the CONTEXT
  sentences (from elsewhere in the same document) to check facts so the merged summary neither
  contradicts them nor re-invents details they already state, without summarising the context itself.
  The prompt text lives in the config (`merge_instruct_prompt`, `merge_instruct_context`). The context
  is placed LAST, after the summaries, so that if an input ever exceeds the model's limit the tokenizer
  truncates context rather than summary text. The instruction's own length is taken out of the group
  budget. Without context, the same instruction is used minus the context parts, so a "no context"
  vs "with context" comparison isolates the effect of the context from the effect of the model/prompt.
- Model chosen: **google/flan-t5-base** (~250M parameters). flan-t5-large (~780M) was also available.
  Chosen on memory and speed: this machine has 15.2 GB RAM with about 5.7 GB free during work and the
  pipeline already peaks near 2.9 GB; flan-t5-base brought the process to about 1.4 GB when loaded,
  while flan-t5-large's weights alone are about 3 GB, which would leave little headroom, and it is
  roughly three times slower on CPU. Both have a 512-token input limit (DistilBART: 1024). With the
  200-token context and the instruction, about 230 tokens remain per merge group, so with this model
  the group size is often limited by the token budget rather than by `max_group_sections`.

**How it is evaluated** (`scripts/eval_merge_context.py`, same tables as section 14 plus one column):
coverage = sections cited by >= 1 final sentence (Stage 5) / sections with a summary; merge-introduced
contradictions; Stage 4 seconds and peak memory; and two pieces of evidence that the model USES the
context: `copied` (final sentences that are near-verbatim copies of a context-only source sentence) and
`ctx-only words` (content words in the output that appear in the given context but in none of the
summaries being merged). Stage 3 cannot change (Stage 4 runs after it); it is reported for completeness.
Group sizes 3, 5 and 8, each without and with context, on sample 05 and on the 31-section synthetic
benchmark (data/bench/medium_15pages.md).

**Result** (`scripts/eval_merge_context.py`, run by the user on 2026-09-19/20; files
`outputs/merge_context_eval_{flan_05,distil_bench,flan_bench}.txt`). "cited" = sections cited by >= 1 final
sentence; "copied" = final sentences that are near-verbatim copies of a context-only source sentence;
"ctx-only" = output words found only in the context; "gens" = merge-model calls.

Sample 05 (20 sections; the DistilBART rows are from section 14):

| Model / prompt | Group size | cited: no ctx -> ctx | final sentences | copied / ctx-only (with ctx) | gens: no ctx -> ctx | introduced: no ctx -> ctx | Stage 4 s: no ctx -> ctx | peak MB |
|---|---|---|---|---|---|---|---|---|
| DistilBART / none | 3 | 2 -> 2 | 3 -> 3 | 0 / -- | 11 -> 11 | 0 -> 0 | 51 -> (outlier; 57-114 on re-time) | ~2800 |
| DistilBART / none | 5 | 4 -> 4 | 5 -> 5 | 0 / -- | 5 -> 5 | 0 -> 1 | 41 -> 47 | ~2820 |
| DistilBART / none | 8 | 3 -> 3 | 5 -> 4 | 0 / -- | 4 -> 4 | 0 -> 0 | 43 -> 48 | ~2870 |
| flan-t5-base / instruct | 3 | 1 -> 3 | 3 -> 3 | 1 / 0 | 11 -> 16 | 0 -> 0 | 119 -> 198 | ~2680 |
| flan-t5-base / instruct | 5 | 1 -> 4 | 3 -> 4 | 1 / 0 | 8 -> 14 | 0 -> 0 | 112 -> 356 | ~2690 |
| flan-t5-base / instruct | 8 | 1 -> 4 | 3 -> 4 | 1 / 0 | 7 -> 14 | 0 -> 0 | 108 -> 148 | ~2700 |

31-section synthetic benchmark (extractive reference: 30 / 31 cited, 88 sentences):

| Model / prompt | Group size | cited: no ctx -> ctx | final sentences | copied / ctx-only (with ctx) | gens: no ctx -> ctx | introduced: no ctx -> ctx | Stage 4 s: no ctx -> ctx | peak MB |
|---|---|---|---|---|---|---|---|---|
| DistilBART / none | 3 | 3 -> 3 | 4 -> 3 | 0 / 0 | 17 -> 17 | 3 -> 4 | 103 -> 95 | ~2780 |
| DistilBART / none | 5 | 3 -> 4 | 3 -> 5 | 0 / 0 | 9 -> 9 | 12 -> 8 | 81 -> 103 | ~2840 |
| DistilBART / none | 8 | 7 -> 8 | 11 -> 12 | 0 / 1 | 5 -> 5 | 9 -> 37 | 97 -> 75 | ~2860 |
| flan-t5-base / instruct | 3 | 1 -> 4 | 2 -> 4 | 0 / 1 | 17 -> 17 | 23 -> 38 | 68 -> 89 | ~3820 |
| flan-t5-base / instruct | 5 | 3 -> 2 | 5 -> 2 | 0 / 0 | 9 -> 15 | 14 -> 19 | 55 -> **excluded** | ~3880 |
| flan-t5-base / instruct | 8 | 7 -> 3 | 9 -> 3 | 0 / 0 | 5 -> 15 | 17 -> 12 | 45 -> 101 | ~3890 |

What the numbers say:
1. **flan-t5-base does not use the context as intended.** Across all six with-context runs, "ctx-only"
   is 0 or 1 -- no content taken from the context and fused in. The only trace of context use is
   copying: on sample 05 each with-context output contains one near-verbatim copy of a context sentence
   (1 of its 3-4 sentences), which matches a single-example check made before the runs, where the model
   output the context sentence in place of the summaries. That is the model echoing its input, not using
   it to check facts.
2. **Coverage does not improve reliably.** On the benchmark, adding context changed flan's coverage by
   +3, -1 and -4 sections of 31 at group sizes 3, 5 and 8 (DistilBART: 0, +1, +1): no consistent
   direction. On sample 05 flan went from 1 to 3-4 sections of 20 with context, but that gain is
   confounded: the room taken by the context and the instruction forces smaller groups, so the
   with-context runs make more model calls (11 -> 16, 8 -> 14, 7 -> 14) and more rounds (3 -> 4), and more
   calls per summary can raise coverage whether or not the context is read. Given point 1, the gain is
   not attributed to the context.
3. **flan-t5-base is a worse merger than DistilBART on the benchmark, with or without context.** Without
   context (the pure model effect) it introduced 23 vs 3 contradictions at group size 3, 14 vs 12 at 5 and
   17 vs 9 at 8; on sample 05 neither model introduced any. Its coverage without context is equal or lower
   (sample 05: 1 of 20 at every group size, vs DistilBART's 2-4).
4. **Cost is mixed.** flan-t5-base was slower on sample 05 (108-119 s vs 41-51 s without context) but faster
   on the benchmark (45-68 s vs 81-103 s), and used about 1 GB more peak memory on the benchmark (similar
   on sample 05).
5. **Excluded outlier.** "flan g5, WITH context" on the benchmark took 803.5 s against 89-101 s in
   neighbouring rows, and is excluded from any cost reading; it does not affect the coverage or
   introduced-contradiction numbers. Recorded as a pattern, not explained: it is the third 10-30x outlier
   in this series of tests (1,536 s and 820 s earlier), and all three were with-context rows. Earlier
   re-timings did not reproduce them, and the row was not re-run.

**Conclusion.** The hypothesis that DistilBART's failure in section 14 was purely a lack of instruction
following is only partly supported. Swapping in a small instruction-tuned model did not resolve it:
flan-t5-base still does not use the provided context in any measurable way beyond occasionally copying
it, and it is independently a worse merger (more invented contradictions on the benchmark). Either a
larger, more capable instruction-tuned model is needed to see Ou & Lapata's effect -- untested here, and
likely infeasible on this CPU-only setup -- or the technique's benefit does not transfer reliably to
small open-weight models, instruction-tuned or not.

**Verdict: negative; investigation closed.** Both attempts to apply context-anchored merging to our merge
step (DistilBART, section 14; flan-t5-base, this section) are closed. Defaults are unchanged:
`models.merge_summarizer` is empty (the merge step uses the Stage 2 model, DistilBART),
`merging.merge_prompt` is `none` and `merging.context_anchoring` is `false` -- the evaluated options were
never made defaults. The ability to configure a separate merge model and an instruction prompt stays in
the code, so a larger or differently tuned model can be tested later with the same script. No further
model will be tried for this technique unless explicitly requested.

## 16. Dialogue-to-description preprocessing: bug found, fixed and re-measured -- kept (off by default)

**What this is and is not.** A formalisation of what the pipeline already did informally (speaker labels
stripped before NLI, reporter introductions filtered in Stage 3), inspired by the dialogue-to-description
step of NexusSum (Kim & Kim, ACL 2025). NexusSum performs this rewriting with an LLM agent. This is
NOT a reimplementation of it: it is a small set of deterministic rules (`src/hcv_sum/dialogue.py`),
chosen because it runs on CPU in milliseconds and every rewrite can be traced to a rule.

**Rules** (applied before segmentation, only to paragraphs that start with a speaker label
"Name[, Title]: utterance"; everything else is untouched):
1. Sentences that assert nothing are dropped ("Thank you, operator.", "Please go ahead."): fewer than
   four content words, or a short fragment with no verb and no number -- the Stage 3 non-claim rule.
2. First person becomes third person: we / us -> the configured organisation reference
   (`preprocessing.organisation_reference`, default "the company"), our / ours -> "the company's",
   I / me -> the speaker's name, my -> "<name>'s". The present-tense verb right after the new subject
   is put in the third person singular ("we plan" -> "the company plans", "I think" -> "Daniel Okafor
   thinks"); contractions (we're, we've, I'm, ...) are expanded first. Verbs are found with nltk's
   part-of-speech tagger.
3. The first kept sentence of each turn is attributed: "Name, Title, said that <sentence>"; later
   sentences of the turn are not re-attributed. The operator becomes "The conference operator".
When the step is on, the pipeline also strips these "Name, Title, said that" frames before NLI (as it
already strips speaker labels), because a reporting frame on the hypothesis side suppresses
contradiction scores (section 12.1).

Known limits of the rules, visible before any evaluation: a greeting long enough to pass the non-claim
filter is attributed like a claim ("The conference operator said that good afternoon, and welcome
to..."); "you" is left as is; a "we" that does not mean the organisation (e.g. analysts on the call) is
still rewritten to it; only the verb directly after the pronoun is agreed.

**How it is evaluated** (`scripts/eval_dialogue.py`): samples 01 and 03, full pipeline with the step off
(current behaviour) and on. Sample 01 has NO speaker-labelled lines, so it is a control: its output
should be identical in both runs. Sample 03 (an earnings-call transcript, 9 speaker turns) is the real
test: segmentation (03 has no headings, so its sections come from topic segmentation and can move),
Stage 2 summaries (invented attributions removed, first-person sentences left, a by-eye comparison of
the first three section summaries), Stage 3 (planted contradiction caught or not, false flags, pairs
compared) and Stage 5 support.

**First measurement, BEFORE the fix below** (`scripts/eval_dialogue.py`, run by the user on 2026-09-19;
Python 3.14.6 environment). Kept for the record; superseded by the re-measurement after the fix.

| Sample | Metric | Off (current) | On (rewrite) |
|---|---|---|---|
| 01 | final summary | -- | **identical** (no speaker turns: 0/0/0) |
| 03 | speaker turns / sentences kept / dropped | -- | 9 / 30 / 9 |
| 03 | sections (sentences per section) | 5 (10, 11, 9, 4, 5) | 4 (7, 9, 9, 5) |
| 03 | summary sentences | 23 | 16 |
| 03 | invented attributions removed in Stage 2 | 0 | 0 |
| 03 | summary sentences with first person | 8 | **0** |
| 03 | pairs compared (3a + 3b) | 105 | 72 |
| 03 | planted contradiction | caught by 3b only (0.61) | caught by **3a** (0.61) and also by 3b (0.64) |
| 03 | false flags | 0 | **1** |
| 03 | final sentences supported / weak / unsupported; DISPUTED | 22 / 0 / 0; 1 | 16 / 0 / 0; 3 |

- **Control holds.** Sample 01 has no dialogue, and its output is identical with the step on.
- **What improved on 03.** First-person sentences in the summaries fell from 8 to 0, and speaker labels
  and pleasantries no longer leak into summaries (without the rewrite, the first section summary
  contains "Priya Raman, Investor Relations: Thank you, operator."). The planted contradiction is now
  caught by the summary-vs-summary check as well, because both sides survive into the summaries.
- **What got worse.** One new false flag, P=0.52: "The conference operator said that good afternoon, and
  welcome to the Helix Cloud Systems third quarter 2026 earnings conference call" against the migration
  sentence. This is the greeting flaw noted before the run -- a greeting long enough to pass the
  non-claim filter is attributed as a claim -- and the same sentence also appears in the first section
  summary. Summaries are shorter (23 -> 16 sentences), and segmentation changed (5 -> 4 sections), since
  03 has no headings and its topic boundaries move when the text changes.
- **Not tested by this sample.** The Stage 2 guard against invented speaker attributions did not fire in
  either run, so this sample says nothing about whether the rewrite prevents those (they were a problem
  on the FOMC press-conference transcript, 11.3).
- One transcript, one planted contradiction: indicative only.

### 16.1 Bug found in that measurement, and the fix

**The bug.** The first section summary of the rewritten sample 03 contained "The conference operator said
that good afternoon, and welcome to the Helix Cloud Systems third quarter 2026 earnings conference call."
-- a scripted greeting presented as a reported statement, and ungrammatical. A search of the whole
rewritten transcript found it was one of four malformed attributions among the nine "said that"
sentences (sample 01 has none; it contains no dialogue):
1. "The conference operator said that good afternoon, and welcome to ..." -- greeting.
2. "Priya Raman, Investor Relations, said that joining Priya Raman today are Daniel Okafor, ..." --
   speaker introduction.
3. "The conference operator said that the company's first question comes from Aaron Feld at Brookline
   Securities." -- call procedure; also "our" wrongly rewritten to "the company's".
4. "Aaron Feld, Brookline Securities, said that can you talk about hiring plans ...?" -- a question forced
   into "said that".
Four more non-substantive sentences were kept without attribution: "At this time all participants are in
a listen-only mode", "... would now like to turn the call over to Priya Raman ...", the
forward-looking-statements notice and "A reconciliation of non-GAAP measures is available ...".

**Root cause.** The rewrite did not call Stage 3's non-claim filter, but it duplicated the same RULE with
the same helpers: fewer than four content words (greeting words are stopwords), or a short fragment with
no verb and no number. That rule catches short pleasantries ("Thank you, operator.", "Please go ahead.")
and nothing longer. The greeting has nine content words (the company, "third quarter 2026", "earnings
conference call"), so the rewrite kept it -- and Stage 3's own filter also accepts it as a claim, which
is how it produced the one new false flag (P=0.52). The existing tests
(`test_pleasantries_and_speaker_labels_are_not_claims`, `test_different_speakers_are_not_compared_as_claims`)
only cover short pleasantries. A second, rewrite-only gap: "X said that" was attached to whichever
sentence of a turn came first, without checking that it was a statement.

**The fix** (`src/hcv_sum/dialogue.py`, `speech_act`):
- Non-substantive speech acts are recognised by FORM, whatever their length -- greetings and welcomes,
  thanks, closings, call procedure, speaker introductions and safe-harbour boilerplate -- with patterns
  anchored so that substantive sentences are not caught ("Thanks to strong demand, revenue grew ..." and
  "We welcomed 50 new customers" are not speech acts).
- **They are dropped**, not kept in their original form. Reasons: the rewrite already drops short
  pleasantries, so this applies the same decision to the long ones the word-count rule missed; keeping
  them as "Operator: Good afternoon ..." would put speaker labels and pleasantries back into the summaries
  (in the run without the rewrite, the first section summary contained "Priya Raman, Investor Relations:
  Thank you, operator.") and back into Stage 3 as claim candidates, which is where the false flag came
  from. None of these sentences can carry a contradiction worth detecting. To make that checkable rather
  than assumed, every dropped sentence and its category is recorded in the run stats and printed by the
  evaluation script.
- "Said that" is attached only to the first STATEMENT of a turn. A question is kept as
  "Name, Title, asked: <question>" with its wording unchanged (no pronoun rewriting: an analyst's
  "we"/"our" is not the company's). Questions stay Stage 3 claim candidates, as they are without the
  rewrite.
- Multi-document mode now reads a document's period markers from its ORIGINAL text, because a dropped
  welcome line ("... third quarter 2026 earnings conference call") can be a transcript's only period marker.
- Checked on sample 03's text alone (rules only, no models): 16 sentences dropped instead of 9 (1
  greeting, 6 procedure, 5 thanks, 1 introduction, 2 boilerplate, 1 short pleasantry), all 16
  non-substantive; both sides of the planted contradiction kept; the six remaining attributions are
  five well-formed "said that" statements and one "asked:" question. Sample 01 is unchanged.
- Tests: `tests/test_dialogue.py` -- including one with the exact operator line, which must not produce a
  "said that <greeting>" construction, and one that scans the whole rewritten sample 03 for malformed
  attributions.

### 16.2 Re-measurement after the fix

**Result** (`scripts/eval_dialogue.py`, run by the user after the fix; `outputs/dialogue_eval.txt`, plus
`--evidence` runs of sample 03 in both modes, `outputs/dialogue/03_evidence_{off,on}.txt`):

| Sample | Metric | Off (current) | On (rewrite, fixed) |
|---|---|---|---|
| 01 | final summary | -- | **identical** (no speaker turns) |
| 03 | speaker turns / sentences kept / dropped | -- | 9 / 23 / 16 |
| 03 | dropped by category | -- | greeting 1, procedure 6, thanks 5, introduction 1, boilerplate 2, pleasantry 1 |
| 03 | "said that good afternoon" present | -- | **no**; 6 attributed sentences, all well formed |
| 03 | sections (sentences per section) | 5 (10, 11, 9, 4, 5) | 3 (10, 9, 4) |
| 03 | summary sentences / final sentences | 23 / 22 | 14 / 14 |
| 03 | final sentences that are greetings, thanks, procedure or boilerplate | 10 of 22 | **0 of 14** |
| 03 | summary sentences with first person | 8 | 0 |
| 03 | pairs compared (3a + 3b) | 105 | 44 |
| 03 | planted contradiction | caught by 3b (0.609) | caught by 3a (0.609) **and** by 3b (0.642) |
| 03 | false flags | 0 | **0** |
| 03 | final sentences supported / weak / unsupported | 22 / 0 / 0 | 14 / 0 / 0 |

- **The bug is gone.** No malformed attribution; every dropped sentence is non-substantive (the list is
  printed by the evaluation script and was checked).
- **Real quality improvement on the transcript.** Without the rewrite, 10 of the 22 final sentences were
  greetings, thanks, procedure or boilerplate ("Priya Raman, Investor Relations: Thank you, operator.",
  "Our first question comes from Aaron Feld at Brookline Securities."); with it, none of the 14 is, all 14
  are supported, and no first-person sentence remains. The greeting-driven false flag of the first
  measurement is gone.
- **Double counting (a reporting nuance, not a detection error).** The rewrite-on run shows 2 flags, but
  both are the ONE planted contradiction: the summary-vs-summary check pairs "Every enterprise customer is
  now running on the new Helix Core platform" with "... roughly sixty enterprise accounts are still being
  served from the legacy environment ..." (0.609), and the summary-vs-source check pairs the latter with
  the full source sentence "Every enterprise customer is now running on the new Helix Core platform, and
  the company has shut down the legacy hosting environment for good." (0.642). It appears twice because the
  rewrite changed segmentation (5 -> 3 sections) so that the "sixty accounts" sentence now survives into a
  summary, which lets both checks fire. The pipeline does not merge two flags that point at the same
  underlying conflict, so counts (and `--brief`'s "found 2 contradictions") overstate the number of
  distinct conflicts by one here.
- Still one transcript: indicative, not a general result. The Stage 2 invented-attribution guard never
  fired on this sample, so whether the rewrite prevents invented speaker tags is untested here.

**Verdict: kept, off by default** (`preprocessing.dialogue_to_description`). On the one dialogue sample it
removes all non-substantive and first-person residue from the summary without losing the planted
contradiction and without adding a false flag; the double counting above is documented.

## 17. Multi-document mode: measured on the 3-document test case -- works as designed, with caveats

**What this is.** An unsupervised, retrieval-and-centrality-based approximation of multi-document
summarization, inspired by Liu & Lapata's cross-document attention concept, using no learned model and
no training data -- this is not a reimplementation of their method, which would require a trained
attention mechanism and salience ranker. It is evaluated only on our own small synthetic test case
below, not on their benchmark or dataset, and nothing here is a claim about how it would compare with
their approach.

**What was built** (`src/hcv_sum/multidoc.py`, `periods.py`, `centrality.py`; used only when several
documents are given, e.g. `hcv-sum q1.md q2.md q3.md`; single-document runs are unchanged):
1. **Document layer.** Each document is segmented on its own; all sections are then pooled into one
   index, each section title prefixed with its document name.
2. **Cross-document pooling.** Stage 2 context retrieval and Stage 3 contradiction checking run over the
   pooled sections, so a section of one document can be anchored against, and compared with, a section
   of another.
3. **"Possibly superseded" outcome.** A cross-document flag is downgraded to `possibly_superseded` when
   the two documents carry different explicit period/version markers (quarters, fiscal years, month-year
   dates, version or amendment numbers; `periods.py` finds explicit markers only) AND the two claims are
   not about the same explicit period. A claim's period is its own marker if it has one (a year-less
   "first quarter" takes its document's year), otherwise its document's period. If both claims name the
   same period, the flag stays a contradiction. Configurable: `multidoc.supersede_by_period`.
4. **Salience in place of a learned ranker.** The final summary keeps the `multidoc.max_summary_sentences`
   (12) de-duplicated section-summary sentences with the highest PageRank on their k-nearest-neighbour
   embedding graph, computed across all documents together, in document order. The graph and centrality
   code is the code from the centrality investigation (13.4), moved into `centrality.py` and shared.
5. Stage 5 provenance cites each final sentence to its source sentence in whichever document it came from.

**Test case** (`data/multidoc/`; ground truth `ground_truth.json`, written before any run): three short
synthetic quarterly updates (Q1, Q2, Q3 2026) from one fictional company, four headed sections each.
- C1, genuine cross-document contradiction about the SAME period: Q1 "First-quarter revenue was 120
  million dollars" vs Q2 "Revenue in the first quarter was 104 million dollars". Should be flagged and
  stay a contradiction.
- C2, genuine contradiction about a timeless fact: Q1 "Veltris has never paid a dividend" vs Q3 "has paid
  a quarterly dividend ... every quarter since 2019". **Prediction written before the run: the period
  rule will WRONGLY downgrade it**, because neither claim names a period and the documents' periods
  differ. It is included to measure that known weakness, not to show the rule working.
- S1, S2, legitimate updates: headcount 1,200 (Q1) vs 1,350 (Q2); gross margin 41% (Q2) vs 44% (Q3). If
  flagged, they should become `possibly_superseded`, never a contradiction.
- Other quarter-over-quarter differences (revenue per quarter, operating expenses, cash, delivery time)
  are also legitimate; any flag on them that stays a contradiction is a false flag.

**How it is evaluated** (`scripts/eval_multidoc.py`): runs the three documents together and matches
flags to the labelled pairs strictly (both claims must ground to the labelled source sentences).

**Result** (`scripts/eval_multidoc.py` and the CLI, run by the user on 2026-09-19; both gave the same output):

| Item | Expected | Observed |
|---|---|---|
| document periods | Q1 / Q2 / Q3 2026 | Q1 / Q2 / Q3 2026 |
| C1 (same-period restatement) | flagged, stays a contradiction | **caught (P=1.00), kept as a contradiction** |
| C2 (timeless fact) | flagged; predicted to be wrongly downgraded | caught (P=1.00), **wrongly downgraded**, as predicted |
| S1 headcount | not flagged, or possibly_superseded | flagged (P=1.00), downgraded -- correct |
| S2 gross margin | not flagged, or possibly_superseded | flagged (P=0.84), downgraded -- correct |
| other flags | as few as possible stay contradictions | 9, all legitimate updates; **0 stayed contradictions**, 9 downgraded |
| flags within one document / across documents / possibly superseded | -- | 0 / 13 / 12 |
| pairs compared | -- | 185 summary pairs (3a) + 8 summary-vs-source (3b) |
| final summary | 12 sentences, all three documents | 12 of 26 candidates, all three documents cited; 12/12 supported, 4 DISPUTED |

- **Cross-document pooling works on this case.** Every flag was cross-document (13), which is exactly
  what pooling makes possible: a single-document run cannot compare Q1 with Q2 at all.
- **The period rule does what it was written to do, including its predicted failure.** Of the 13 flags,
  the one that stays a contradiction is C1, and it is a genuine conflict; all 11 legitimate updates were
  downgraded. But the rule downgraded 12 of 13 flags: on documents with different periods it amounts to
  "a cross-document difference is an update unless both claims name the same explicit period". C1
  survived only because both sentences literally say "first quarter"; C2, a genuine contradiction about
  a timeless fact, was downgraded. So on this case the rule trades 1 of 2 real contradictions for the
  removal of 11 false ones -- recall 1/2, and no false contradictions left.
- **Stage 2 was barely exercised.** 10 of the 12 sections were short enough to be kept verbatim; only 2
  were summarised. The test measures pooling, the period rule and salience, not multi-document
  summarisation quality.
- **The final summary keeps both sides of C1** ("First-quarter revenue was 120 million dollars" and
  "Revenue in the first quarter was 104 million dollars"), marked through Stage 3's flags, not resolved.
  Salience picked the revenue and operating-expense lines of every quarter; it left out gross margin,
  most headcount, the dividend statements and the German customer.
- The test documents were written by us, with explicit period markers in every title, and are three
  pages in total. Nothing here says how the mode behaves on real multi-document input.

**Verdict: works as designed on its own small test case, with caveats.** The planted same-period
contradiction is surfaced and kept; every legitimate quarter-over-quarter update is kept out of the
contradiction list; the known weakness (timeless facts) occurred exactly as predicted. It remains an
unsupervised approximation evaluated on three synthetic pages, not a validated multi-document system.

### 17.1 Bugs found in the pre-scale code review (before any larger multi-document run)

Found by reading the multi-document path with 3-5 real documents in mind, confirmed with fake models, fixed,
and covered by tests in `tests/test_multidoc.py`:
1. **Crash:** `hcv-sum a.md b.md --verbose` raised `KeyError: 'embedded_texts_total'` after the whole run had
   finished -- the multi-document stats lacked fields the verbose report reads (also `per_stage` and the
   full pair counts). The multi-document path now records the same stats as the single-document one.
2. **Silent capping:** multi-document runs produced no RUN LIMITS at all, so a cap hit on a large pooled set
   (per-claim candidate limit, pair budget, input truncation, resolution budget, segmentation ceiling) would
   have gone unreported. They now report the same limits, with segmentation limits counted as hit if they
   bound in any document, plus per-stage time and peak memory.
3. **Looked like a hang:** no progress output during a long multi-document Stage 2. Progress lines now go to
   stderr once the pooled sections reach `scale.large_document_sections`, as in single-document runs.
Also: the CLI now says so when single-document options (`--diagnose-sources`, `--checkpoint`,
`--checkpoint-dir`) are given with several documents, instead of ignoring them silently; and salience
de-duplication reads the configured threshold instead of a hard-coded copy of the same value (0.85).

