# HCV-Sum — Hierarchical Consistency-Verified Summarization

## 1. Overview

HCV-Sum is a CPU-only pipeline that summarizes long documents section by section, then checks its own
output for **contradictions between different parts of the document** and traces every sentence of the
final summary back to the source sentence that supports it. The result is a summary plus an evidence
panel: a citation and support score for each sentence, and a list of flagged conflicts for a person to
review. It is a research and portfolio project built to investigate one under-explored question — can a
summarizer notice that two *sections* of a document disagree with each other, not merely that a summary
disagrees with its source? — and to measure honestly where the answer is yes and where it is no. It is
not a production tool.

The full investigation log, with every measurement, dead end and correction, is in
[FINDINGS.md](FINDINGS.md). Design notes for future work are in [IDEAS.md](IDEAS.md).

## 2. The problem and prior work

Most work on summary reliability asks whether a summary is **faithful to its source**. SummaC
([Laban et al., TACL 2022](https://aclanthology.org/2022.tacl-1.10/)) is a representative example: it
splits the source and the summary into sentences and uses a natural-language-inference (NLI) model to
check that each summary sentence is supported. That catches a summary that invents or distorts facts. It
cannot catch a document that is **inconsistent with itself** — a report whose operations section blames a
revenue drop on a plant shutdown while its revenue section says the drop was unrelated to the shutdown.
Each section summary can be perfectly faithful to its own section while the two contradict each other,
and a long-document summarizer that merges them either hides the conflict or silently picks one side.

Long-document summarization is usually hierarchical: split the input into chunks, summarize each, then
merge the summaries. [Ou & Lapata (Findings of ACL 2025)](https://aclanthology.org/2025.findings-acl.289/)
show that the merge step amplifies hallucination and that giving the merger relevant source context
reduces it; this project borrows the idea of retrieving context from other parts of the document.
[NexusSum (Kim & Kim, ACL 2025)](https://aclanthology.org/2025.acl-long.500/) uses a hierarchy of LLM
agents for books and scripts and rewrites dialogue into narrative before summarizing — relevant here
because much of this project's trouble with transcripts comes from feeding dialogue to a model trained
on news prose. [Liu & Lapata (ACL 2019)](https://aclanthology.org/P19-1500/) summarize *multiple*
documents with learned attention between paragraphs and a learned salience ranker, the natural reference
for extending this work beyond one document. [GloSA-sum (Zhang et al., ICLR 2026)](https://openreview.net/forum?id=uNaXiGL5uo)
uses topological analysis of a sentence-similarity graph to find a document's structural backbone and
protect it from deletion, a more general version of a protection mechanism this project uses.

None of these checks whether the parts of one document contradict **each other**. That is the gap this
project targets: **sibling-contradiction detection** — comparing the claims of different sections before
they are merged, and resolving or flagging conflicts rather than letting the merge hide them. The
detector itself is an off-the-shelf NLI model, so the project also runs into a known weakness of NLI,
documented by [Chen et al. (2025), "On Reference (In-)Determinacy in NLI"](https://arxiv.org/abs/2502.05793):
NLI models assume two sentences describe the same situation, and that assumption turns out to be the
central limit on precision (section 5).

## 3. Architecture

Five stages, all running on a CPU with three small pre-trained models (a sentence-embedding model, an
NLI model and a DistilBART summarizer; about 1.6 GB in total):

```
 document
    │
    ▼  1. Segmentation ............ split on real headings; otherwise find topic shifts from sentence
    │                                embeddings. Over-long sections are split at their weakest topic link.
    ▼  2. Context-anchored ........ summarize each section, with related sentences retrieved from OTHER
    │     summarization              sections as context. Sentences that other sections found relevant
    │                                are re-attached verbatim if compression dropped them.
    ▼  3. Contradiction ........... compare claims across sections with NLI, in both directions:
    │     detection                   (a) summary claim vs summary claim of another section
    │                                 (b) summary claim vs the SOURCE text of another section
    │                                Conflicts are resolved by which claim its own section supports more
    │                                strongly, or kept and flagged "unresolved". Nothing is deleted.
    ▼  4. Merge ................... combine the section summaries in document order, removing
    │                                near-duplicates (a model-rewritten merge is used only for very long
    │                                documents; see section 5)
    ▼  5. Provenance .............. for every final sentence: the source sentence that supports it,
                                     support score, and a DISPUTED mark if another section contradicts it
    │
    ▼
 summary + evidence panel
```

Why two kinds of comparison in stage 3: summarization often keeps one side of a conflict and drops the
other. Comparing summaries with summaries finds conflicts only when both sides survive; comparing a
summary with another section's source text finds the ones where one side was compressed away. Each
catches cases the other misses.

Everything tunable — models, thresholds, limits — lives in one configuration file; the code has no
hidden defaults. For long inputs the pipeline also estimates cost before running, checkpoints the
summarization stage so an interrupted run can resume, keeps memory bounded when comparing thousands of
sentences, and reports any cap or limit it hit so a constrained result is never mistaken for a complete one.

## 4. What it can do

Every number below comes from a fixed, labelled evaluation; the labels were written before the relevant
run.

- **It catches explicit, directly stated conflicts between two sentences.** On a frozen set of 100
  sentence pairs drawn from four real public documents (a research paper, central-bank minutes, a
  contract and a press-conference transcript), it catches **24 of 36** contradictions, including
  **5 of 5** pairs that state two different values for the same quantity, while flagging only **2 of 64**
  non-contradictory real pairs.
- **It finds most planted contradictions in whole documents.** Across five sample documents with six
  planted cross-section contradictions, the default pipeline catches **4 of 6**. One of the four only
  barely: a numeric conflict in a four-page paper scores 0.523 against a 0.5 threshold.
- **It never deletes content on its own judgement.** Early versions removed the claim that lost a
  contradiction. On real documents, 14 of 18 such removals turned out to delete correct content, all
  triggered by false flags, so conflicting claims are now kept in the summary and marked for review.
- **Every summary sentence is traceable.** The evidence panel cites the source sentence behind each
  final sentence with an entailment score, and marks sentences that another section disputes.

In short: it is a good detector of *explicit* opposition, and its output is a reviewable queue of
candidate conflicts with evidence attached — not a verdict.

## 5. What it cannot do, and why

These limits were each isolated with controlled tests; they are findings about the approach, not
unfinished features.

| Limitation | Evidence |
|---|---|
| **Contradictions that need arithmetic or derivation are missed.** | A contract states a 24-month term from 1 January 2026 and elsewhere an expiry of 31 December 2026. The NLI model scores the pair 0.02; spelling the end date out step by step raises it to 0.15, then 0.91, then 0.999. On real text: 0 of 7 conflicts that require arithmetic (hours to days, subtraction, reading a ratio) and 0 of 2 causal denials. |
| **A conflict is lost when a number is qualified or a claim is wrapped in a reporting frame.** | "The dataset spans eighteen months of sensor readings…" vs "We also note that the dataset, spanning twenty-four months…, provided…" scores 0.15, although the bare conflict scores 0.99. Controlled variants ruled out sentence length. Two specific triggers remained: a qualifier attached to the quantity ("months *of sensor readings*" is read as a different quantity; 0.01–0.04) and a reporting frame ("We also note that…"; 0.005–0.63). Stripping reporting frames before comparison recovered this case, at a cost of 7 new false flags. |
| **Summarization can remove the evidence before it is checked.** | If compression drops both sides of a conflict, neither comparison can see it. Re-attaching sentences that other sections found relevant fixed this for one planted case, but in the four-page paper one contradiction is still found only by an optional, slower check over the raw source text. |
| **Most flags are false, and there are more of them the longer the document.** | See below. |
| **One document at a time, in the measured pipeline.** | Everything measured above is single-document. An experimental multi-document mode (section 8) has only been checked on a three-page synthetic test case, and it uses no trained model. |
| **Not tested beyond about 20 pages.** | Real prose runs at about one minute per page on this CPU. The longest document run was 19 pages. For very long documents the pipeline switches to a model-rewritten merge, and that merge was measured citing only 2–4 of a paper's 20 sections, so its output on long inputs should not be trusted without review. |

**The precision ceiling.** The detector produces false flags at a roughly constant rate of **4.5–6 per 100
sentence pairs compared**, on short samples and on real papers alike. Because a longer document means
more pairs, the count grows with length: 122 false flags across the five samples (99 of them on the
four-page paper), and about 610 flags on four real 6–19-page documents that are presumably consistent.
Classifying the paper's false flags showed why. Nearly all are **reference errors**. The model reads two
sentences as rival claims about the same thing when they describe different things: the proposed method
against its own ablation or a baseline, one dataset's figures against another's, or a sentence whose
subject the summarizer erased ("The third baseline is a static graph network that uses…" became "The
network uses…", which alone caused 18 false flags). This matches the reference-determinacy weakness of
NLI models: on the RefNLI benchmark this model's contradiction precision is 16.8%.

Every fix tried was measured, and none separated true conflicts from false ones:
- **Filtering comparison wording** ("baseline", "unlike"): removed no false flags on real text and lost a true conflict.
- **Requiring a shared named entity:** halved recall on real documents, and would have removed every true flag on the paper.
- **Matching grammatical subjects:** fails the same way, since true conflicts often phrase their subject differently.
- **Requiring a shared content word:** a small reduction (11% on RefNLI), left off.
- **Protecting "central" sentences in a sentence-similarity graph:** the sentences that carry contradictions were not central.
- **Filtering by graph proximity between the two claims:** true pairs fell inside the false-flag range on every measure.
- **Context-anchored merging** was also tested twice, against a different problem: sections lost during a model-rewritten merge. With DistilBART it had no effect. With a small instruction-tuned model (flan-t5-base), the model still did not use the context beyond occasionally copying a sentence of it. Coverage did not improve consistently, and the model introduced more contradictions while merging than DistilBART did (23 vs 3 at one setting). Closed.

The property that distinguishes a real contradiction — whether the two claims are about the same thing —
is not visible in similarity scores, shared tokens or embedding geometry. With the current models this
appears to be a genuine limitation of NLI-based detection, not a bug.

## 6. Research process

Each design choice and each idea taken from the literature was treated as a hypothesis and tested
against ground truth written before the run: five sample documents, a frozen pair set built from real
public documents, and the external RefNLI benchmark. Where a decision criterion could be stated in
advance, it was written down before the measurement and applied as written — for example, for
content-word filtering and for the proposed re-check in section 8.

Negative results were kept and reported rather than tuned away. A graph-based "same referent" filter
was measured before being built, and dropped when true and false pairs proved inseparable. Retrieving
source context at the merge step, the core idea of Ou & Lapata (2025), was implemented and measured, and
changed nothing with DistilBART, which cannot be told what the context is for. A retest with the
instruction-tuned flan-t5-base did not change that conclusion. A centrality-based protection scheme inspired by GloSA-sum was checked
offline first and dropped when the relevant sentences ranked low.

Mistakes were corrected in place when found. The most instructive one: an evaluation step matched flags
to the planted contradictions by embedding similarity rather than by the actual source sentences, which
let a false flag count as a true one. That caused three reported numbers to be wrong before it was
caught. All three were corrected and the matching rule changed. Every correction is recorded in FINDINGS.md.

## 7. How to run it

**Setup (Windows, CPU only, Python 3.11).**

```powershell
# The virtual environment lives OUTSIDE OneDrive on purpose: torch is ~2 GB across tens of thousands of files.
uv venv C:\Users\Omkar\.venvs\hcv-sum --python 3.11
uv pip install --python C:\Users\Omkar\.venvs\hcv-sum\Scripts\python.exe `
    --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple `
    --index-strategy unsafe-best-match -e ".[dev]"
C:\Users\Omkar\.venvs\hcv-sum\Scripts\activate
python scripts\download_models.py      # downloads and load-checks every model, prints versions
```

Models: `all-MiniLM-L6-v2` (embeddings), `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (NLI) and
`sshleifer/distilbart-cnn-12-6` (summarizer). PDF and Word input need `pypdf` and `python-docx` (both in the
install above; in an existing environment: `pip install pypdf python-docx`).

**Which environment.** `C:\Users\Omkar\.venvs\hcv-sum` (Python 3.11.9) is the environment this project runs
in: it matches the pins in `requirements.txt` (torch 2.14.0+cpu, numpy 2.4.6, scipy 1.17.1, transformers
5.17.0, sentence-transformers 6.0.1), it has every runtime dependency including `psutil` and `python-docx`,
and every runtime and memory figure in FINDINGS.md was measured in it. `run.ps1` activates it first.

There is also a project-local `venv\` on Python 3.14.6, kept as a forward-compatibility check rather than a
second working environment. It has drifted from the pins (numpy 2.5.3, scipy 1.18.1, sentence-transformers
6.1.0, and plain `torch` 2.14.0 rather than the `+cpu` wheel) and has no `python-docx` or `lxml`, so `.docx`
input fails there. The suite last passed in it at 302/302, before the ingestion layer was added. To use it
for anything, install the missing packages first: `venv\Scripts\python.exe -m pip install python-docx`.

**Runtime.** About **one minute per page of real prose** on this CPU, measured on four public 6–19-page
documents. The short samples in `data/samples/` take 40–70 s each. Run with `--preflight` for an estimate
before a long document.

**Usage: the wrapper (recommended).** `run.ps1` at the project root is the short path for an ordinary run.
It activates the virtual environment if it is not active, adds `--evidence` (so the summary, the evidence
panel, RUN LIMITS and peak memory are all in the terminal) and saves the complete run to a timestamped file
under `outputs\`, so repeated runs never overwrite each other.

```powershell
C:\Users\Omkar\.venvs\hcv-sum\Scripts\activate               # optional: the script does this itself
.\run.ps1 "data\samples\gao-26-108073.pdf"                   # one document
.\run.ps1 "data\samples\doc1.pdf" "data\samples\doc2.pdf"    # several -> multi-document mode
.\run.ps1 "data\samples\report.pdf" --brief                  # extra flags pass straight through
.\run.ps1 "data\samples\report.pdf" --set contradiction.threshold=0.7
```

Runs are filed by category, so results stay grouped as the corpus grows:

| Folder | What lands there |
|---|---|
| `outputs\small\` | one document under 25 KB of text, or under 100 KB as PDF/DOCX (the `01`–`05` samples) |
| `outputs\medium\` | one document of 25–250 KB of text, or 100 KB–2 MB as PDF/DOCX (a 10–30 page report or paper, e.g. `gao_excerpt.txt`) |
| `outputs\large\` | one document over 250 KB of text, or over 2 MB as PDF/DOCX (the GAO reports) |
| `outputs\multidoc\` | any run given several document paths, whatever their sizes |

Text and PDF get different thresholds on purpose: 22 pages of extracted prose is 56 KB, while 22 pages of a
GAO PDF is several MB of images and fonts, so one cut-off cannot serve both. The folder is created on first
use, and `outputs/` in `.gitignore` already covers every subfolder.

It prints the path and the category at the end, for example
`full run saved to: outputs\medium\gao_excerpt_20260920_181530.json  (category: medium)`.
Document paths come first, flags after them. An output mode you pass yourself (`--brief`, `--verbose`,
`--evidence`) replaces the default `--evidence`. Your own `--json` path is used exactly as given, with no
category sorting, as is any path passed when running the CLI directly. If
PowerShell's execution policy blocks the venv's `Activate.ps1`, the script says so and prints the
`Set-ExecutionPolicy` command that fixes it.

**Advanced usage: the CLI directly.** The wrapper changes no behaviour, flag or default, so anything below
works exactly as before and is the way to reach the flags the wrapper does not fill in (`--preflight`,
`--checkpoint`, `--diagnose-sources`, a specific `--json` path, or piping the summary alone).

```powershell
hcv-sum doc.txt                    # the final summary only; safe to pipe
hcv-sum doc.txt --brief            # a few short lines on what the run did, then the summary
hcv-sum doc.txt --evidence         # the summary plus the evidence panel (citations, support, flags)
hcv-sum doc.txt --verbose          # every stage in full, with progress and library logs on stderr
hcv-sum doc.txt --preflight        # estimate sections, section size and runtime, then exit
hcv-sum doc.txt --json run.json    # also write the complete run to JSON (works with any mode)
```

Experimental options, all off by default (section 8; the merge-model option was tested and did not help):

```powershell
hcv-sum q1.md q2.md q3.md --brief                              # several documents -> multi-document mode
hcv-sum call.txt --set preprocessing.dialogue_to_description=true    # rewrite speaker turns as prose first
hcv-sum doc.txt --set merging.mode=abstractive --set merging.context_anchoring=true `
    --set models.merge_summarizer=google/flan-t5-base --set merging.merge_prompt=instruct
```

**Input formats.** Every file is converted to plain text before Stage 1 (`src/hcv_sum/ingestion.py`); the
stages never see the original format. Formats can be mixed in one multi-document run, and `--verbose` prints
how each file was ingested (e.g. `Ingested as PDF: extracted 14,203 characters from 42 pages, stripped 38
repeated header/footer lines ...`).

| Extension | How it is read | Example |
|---|---|---|
| `.txt`, `.md` | UTF-8, used unchanged (a leading BOM is dropped) | `hcv-sum report.md` |
| `.pdf` | text layer via pypdf; repeated page headers/footers and page-number lines removed, hyphenated line breaks re-joined | `hcv-sum report.pdf --brief` |
| `.json` | `{"text": "..."}`, `{"document": "..."}`, or `{"sections": [...]}` whose items are strings or `{"title": "...", "text": "..."}` (titles become headings); optional top-level `"title"` | `hcv-sum report.json --brief` |
| `.docx` | paragraphs via python-docx; Title / Heading 1–6 styles become markdown headings, so Stage 1 segments on the document's own structure | `hcv-sum report.docx --brief` |

Any other extension, a missing extension, or content that does not match its extension (a `.pdf` without a
PDF header, binary bytes in a `.txt`) stops the run with an error listing these formats; no model is loaded.
Format limitations:

- **Scanned PDFs are not supported.** A PDF with no text layer fails with a message saying OCR would be
  needed. There is no OCR step.
- **Multi-column PDFs** are read in the order the text is stored in the file. For most generated reports
  that is column by column; for some it interleaves the columns line by line. That produces jumbled
  sentences, not an error, so check `--verbose` output on a new source.
- **PDF tables, figures and footnotes** come out as flattened lines of text, in content order. A header or
  footer is recognised only if it repeats on at least half the pages (and at least 3). A running title that
  changes on every page is kept.
- **Word tables are skipped**. `--verbose` reports how many. Text boxes, headers, footers, footnotes and
  comments are not read either.
- **JSON** must use one of the shapes above. Anything else is rejected with the expected shapes, never guessed.

`python -m hcv_sum.cli doc.txt` is equivalent. Any setting can be overridden per run, for example
`--set contradiction.threshold=0.7`. A `--brief` run looks like this:

```
Split the document into 5 sections along the headings and summarized each one.
Compared 76 claim pairs across sections and found 7 unresolved contradictions; kept in the summary and flagged below for review.
Combined into a final summary of 13 sentences.
Every sentence traces back to the source; 6 are disputed by the contradictions above.
```

(Seven flags, one of them the planted contradiction: the flag list is a review queue.)

**Sample documents** (`data/samples/`; ground truth in `GROUND_TRUTH.md`, written before the runs):

| File | Contents | Planted cross-section contradictions |
|---|---|---|
| `01_planted_contradiction.md` | quarterly report | 1, explicit |
| `02_no_contradiction.md` | research write-up | 0, with look-alikes (different datasets, trade-offs) |
| `03_subtle_contradiction.txt` | earnings-call transcript, no headings | 1, implicit |
| `04_contract_term.txt` | contract | 1, needs date arithmetic |
| `05_medium_paper_graphfault.txt` | four-page research paper | 3 (labelled by us; not yet confirmed by the supplier) |

**Tests.**

```powershell
pytest                   # all 381 tests (~3 min)
pytest -m "not slow"     # 357 logic tests with deterministic fake models, ~10 s
pytest -m slow           # 24 tests with the real models, pinning known model behaviour and limitations
```

The fast tests check each stage's logic with fake models and say nothing about model quality. The slow
tests pin the real models' observed behaviour, including the limitations in section 5, so that a change
which alters it is noticed rather than silently accepted. Evaluation and investigation scripts are in
`scripts/`; FINDINGS.md names the script behind each result.

## 8. What's next

- **Validate the grounded re-check.** This is the one untested candidate for the precision problem: for
  each flag, put both claims back into their original source context — section title, preceding
  sentence, and the source sentence each was derived from — and ask the detector again. It is designed,
  implemented as a standalone module, unit-tested and pre-registered with pass criteria. It was never run
  against the validation data and is not connected to the pipeline.
- **Test beyond 20 pages.** The long-document machinery (cost estimate, checkpoint and resume,
  memory-bounded comparison, limit reporting) is built and tested with fake models but has not been run
  on a real long document. Two checks are outstanding: that it leaves short-document results unchanged,
  and which merge mode to default to for long inputs, given the section loss measured above.

**Extensions.** Three extensions, each based on one of the papers in section 2, were implemented and
measured. They are switched off by default, or used only when several documents are given, so the
single-document results above are unaffected. Details and result tables: FINDINGS.md sections 15-17.

- **An instruction-tuned merge model with source context: tested, did not work, closed.** This retested
  Ou & Lapata's idea, which had no effect with DistilBART. flan-t5-base was chosen as the largest
  instruction-tuned model that fits this CPU's memory comfortably, and it was given an explicit instruction
  to use the context only to check facts. It did not use the context beyond occasionally copying one of
  its sentences, and section coverage did not improve consistently. On the 31-section benchmark, without
  any context, it introduced more contradictions while merging than DistilBART (23 vs 3, 14 vs 12 and 17
  vs 9 at three group sizes). The default merge model stays DistilBART. The option to configure a different
  merge model remains, so a larger model could be tested later, which this CPU-only setup likely cannot run.
- **Dialogue-to-description preprocessing.** Speaker-labelled transcript turns are rewritten as
  descriptive prose before segmentation ("Name, Title, said that the company plans to…"), with
  pleasantries dropped. This is inspired by NexusSum's dialogue-to-description step but is not its
  method: NexusSum uses an LLM agent, while this is a small set of rules (pronoun and verb rewriting with a
  part-of-speech tagger), chosen for CPU cost and traceability. **Measured, kept (off by default).** The
  first measurement found a bug: long greetings, procedural lines and questions were rewritten into
  misleading "said that" sentences ("The conference operator said that good afternoon…"). The bug was fixed
  by recognising those speech acts by their form and dropping them. After the fix, on the one transcript
  sample, 10 of the 22 final summary sentences had been greetings, thanks or procedure without the rewrite,
  and none of the 14 were with it. No first-person sentences remain, the planted contradiction is still
  caught, and there are no false flags. One reporting nuance: the rewrite changed segmentation so that the
  same contradiction is now caught by both checks, and it is counted twice. The written-report sample has
  no dialogue and is unchanged. It is one sample, so the result is indicative only.
- **Multi-document mode.** An unsupervised, retrieval-and-centrality-based approximation of multi-document
  summarization, inspired by Liu & Lapata's cross-document attention concept, using no learned model and
  no training data — this is not a reimplementation of their method, which would require a trained
  attention mechanism and salience ranker. Sections from all documents are pooled for context retrieval
  and contradiction checking. A cross-document flag is marked "possibly superseded" when the documents
  carry different explicit periods (for example Q1 and Q2) and the two claims are not about the same
  period. The final summary keeps the most central sentences across all documents. It is evaluated only
  on a small synthetic test case of three quarterly updates, with ground truth written in advance.
  **Measured, works as designed with caveats.** The planted same-quarter contradiction (Q1 revenue
  reported as $120M in one update and $104M in the next) was caught and kept. All 11 flags on legitimate
  quarter-over-quarter changes were downgraded to "possibly superseded", so none was reported as a
  contradiction. The weakness predicted before the run happened as expected: a contradiction about a
  timeless fact (dividend history) was also downgraded. The period rule is blunt — it downgraded 12 of 13
  flags — and the test covers three synthetic pages in which most sections were too short to be
  summarized. So this shows the mechanism working, not a validated multi-document summarizer.
