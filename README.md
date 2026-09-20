# HCV-Sum
### Hierarchical Consistency-Verified Summarization

A CPU-only pipeline that summarizes long documents **and audits its own
output** — catching contradictions between sections and tracing every
sentence back to its source, instead of just compressing text and hoping
for the best.

---

## The Problem

Every long-document summarizer works the same way under the hood: split
the document into chunks, summarize each chunk, stitch the summaries
together. This works — but nothing checks whether the chunks *agree with
each other*. A report can say "every customer migrated to the new system"
in one section and "sixty customers are still on the old one" in another.
Each sentence is individually true to its own part of the document. Read
together, they contradict.

**No existing summarization method checks for this.** SummaC and similar
tools check a summary against its own source. Ou & Lapata's hierarchical
merging reduces hallucination during merging. Neither catches a summary
contradicting *itself* across sections. HCV-Sum does.

## Results

| Metric | Result |
|---|---|
| Direct numeric contradiction detection | **100%** (5/5 controlled test cases) |
| Overall constructed contradiction catch rate | 67% (24/36) |
| Planted contradictions caught, real documents | 4/6 |
| Sentence-to-source traceability | 100% of final sentences traced |
| Content deleted without human review | **0** — always flagged, never silently removed |
| Multi-document period-vs-contradiction accuracy | 3/4 labelled test cases correct |

## Architecture

```
  Document
     │
     ▼
  1. Segment          →  coherent sections (headings or topic-shift detection)
     │
     ▼
  2. Summarize         →  each section, using retrieved context from the rest
     │                     of the document to reduce hallucination
     ▼
  3. Cross-check       →  compare sections against each other AND against
     │                     sibling sections' full source — catches contradictions
     │                     invisible to a source-only check
     ▼
  4. Merge             →  combine into one final summary
     │
     ▼
  5. Trace             →  every final sentence cited back to its source
     │
     ▼
  Summary + Evidence Panel
```

**Inputs:** `.txt` `.md` `.pdf` `.docx` `.json` — single or multiple documents at once.
**Outputs:** plain summary, `--brief` status, `--evidence` full audit trail, `--verbose` debug detail.

## Quick Start

```bash
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
```

```powershell
.\run.ps1 "report.pdf"                    # one document
.\run.ps1 "q1.pdf" "q2.pdf" "q3.pdf"       # several documents at once
.\run.ps1 "report.pdf" --brief             # quick status
```

Results auto-sort into `outputs/small`, `outputs/medium`, `outputs/large`, `outputs/multidoc`.

<details>
<summary>Advanced usage</summary>

```bash
python -m hcv_sum.cli doc.pdf --evidence --json outputs\result.json
python -m hcv_sum.cli doc.pdf --set contradiction.threshold=0.6
```
</details>

## Grounded in Current Research

Eight recent papers on long-document and multi-document summarization were
reviewed and tested against this project's actual problems — not just
cited. The result: two techniques adopted directly, one genuine
contribution beyond the literature, and every other idea either extended,
validated, or rigorously closed with evidence.

| Paper | Outcome |
|---|---|
| Ou & Lapata (2025), *Context-Aware Hierarchical Merging* | **Adopted** — powers Stage 2 |
| SummaC (Laban et al.) | **Adopted, and their stated limitation solved** — Stage 5 traces inconsistencies to a sentence, which their own most-accurate model can't |
| *(this project)* | **Novel contribution** — cross-section contradiction detection; not addressed by any paper reviewed |
| GloSA-sum (2026) | Tested (centrality-based protection) — closed with evidence |
| Reference (In-)Determinacy in NLI (2025) | Diagnosis confirmed our false positives; proximity-based fix tested — closed with evidence |
| Ou & Lapata, applied to merging | Tested twice (two model classes) — closed with evidence |
| Mahendra et al. (2025); Havaldar et al. (2025) | Independently confirm our numeric-reasoning findings |
| NexusSum (2025) | Informally present already; formalized in final build |
| Liu & Lapata (2019) | Scoped as future work — requires a trained model, out of current project scope |

Full experiment log, every number, every negative result: **[`FINDINGS.md`](FINDINGS.md)**.

## Known Limitations

- Misses contradictions requiring arithmetic (e.g., a term length implied by dates, vs. a stated term length).
- A number wrapped in a qualifier or reporting frame can suppress an otherwise-easy contradiction.
- False-positive rate is constant per sentence pair — so absolute count grows on longer documents. Five different fixes were tested; none solved it. Treated here as a characterized, honest limit, not a bug.
- Multi-document mode is an unsupervised approximation, not a trained model — evaluated on a small hand-built case, not a public benchmark.
- Validated to ~20-30 pages in depth; large real-world documents (100+ pages) are the current validation focus.

## Project Structure

```
hcv_sum/          5-stage pipeline + orchestration
data/samples/      test documents, by size and type
tests/             350+ tests (fast + real-model)
outputs/           results, auto-sorted by run type
FINDINGS.md        complete research log
run.ps1            one-command runner
```

## Engineering Approach

Every design choice — architectural and literature-derived — was tested
against a labelled ground truth before being kept. Negative results are
documented with the same rigor as positive ones, and two evaluation bugs
found mid-project were corrected in the record rather than left standing.
This is what `FINDINGS.md` is: a real account of what was tried, not a
highlight reel.

---

**Status:** Core pipeline, multi-document support, and file ingestion are complete and tested. Active work: validation on 100+ page real-world documents.

