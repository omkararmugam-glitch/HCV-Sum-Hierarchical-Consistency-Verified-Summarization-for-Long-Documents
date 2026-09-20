# Sample documents: ground truth

Written BEFORE the first pipeline run. Do not edit a document to make a stage pass;
if a stage misses something, that is a finding about the method.

| File | Structure | True cross-section contradictions | False-positive traps |
|---|---|---|---|
| `01_planted_contradiction.md` | Markdown headings | **1 (explicit)** | Monterrey mentioned in 3 sections; "Europe demand soft" repeated consistently |
| `02_no_contradiction.md` | Markdown headings | **0** | "improves F1 on discharge summaries" vs "no benefit on radiology reports"; "training takes 40% longer" vs "inference runs faster"; "five retrieved notes" vs "ten reduced F1" |
| `03_subtle_contradiction.txt` | None (transcript) -> embedding fallback | **1 (implicit)** | Gross margin "down one point" vs Q4 guidance "around 72 percent"; migration "completed" vs "migration team moving onto product work" (consistent) |
| `04_contract_term.txt` | ALL-CAPS / numbered headings | **1 (requires date arithmetic)** | Liability cap 12 months (general) vs 6 months (termination claims only) is a narrower scoped cap, NOT a contradiction |

## Details

### 01 - explicit (planted)
- *Manufacturing Operations*: "Because the shutdown halted all industrial sensor output, backlog orders went
  unfilled and industrial sensor revenue fell 18% in the quarter."
- *Revenue and Sales Performance*: "The decline in industrial sensor revenue was unrelated to the Monterrey
  shutdown: safety stock ... covered every customer order during the outage, and no backlog orders went unfilled."

Expected Stage 3 behaviour: flag the pair. Resolution is NOT expected to be clean: each claim is fully
entailed by its own section, so both supports should be high and the honest outcome is `unresolved`.
(The document itself is inconsistent; source-support cannot decide which section is right.)

### 03 - implicit
- CEO: "Every enterprise customer is now running on the new Helix Core platform, and we have shut down the
  legacy hosting environment for good."
- CFO: "... roughly sixty enterprise accounts are still being served from the legacy environment while their
  data transfers are completed."

Detecting it requires inferring that "sixty accounts still on legacy" negates "every customer migrated /
legacy shut down". Expected to be harder for a small NLI model, and the summarizer may drop the CFO clause.

### 04 - arithmetic
- TERM AND RENEWAL: effective January 1, 2026, initial term of 24 months (=> ends December 31, 2027).
- TERMINATION: "expires at the end of the initial term on December 31, 2026."

Expected to be MISSED by sentence-pair NLI (needs the effective date from DEFINITIONS plus arithmetic).
Included to document a known limit, not to be made to pass.

## 05_medium_paper_graphfault.txt (supplied by the user, 2026-09-19)

No ground truth came with the file. Labelled by Claude from a full read, before any pipeline run on
it, and awaiting the user's confirmation:

1. Section 4.1 "The automotive dataset spans eighteen months" vs Section 6 "the automotive dataset,
   spanning twenty-four months" (direct numeric conflict; 24 months is the chemical dataset's span).
2. Abstract "transfer effectively across plants with different equipment vendors" vs Section 6
   "only evaluated between plants using similar underlying equipment vendors".
3. Section 5.3 "despite differing equipment vendors and process types" vs the same Section 6 sentence.

Not counted (ambiguous, needs arithmetic): Abstract "reducing the cold-start data requirement by
roughly sixty percent" vs 5.3 "only twenty percent of the target-domain training data".
