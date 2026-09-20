"""Data records produced by each pipeline stage.

Every intermediate result is kept in plain dataclasses so a run can be printed
stage by stage and exported to JSON for auditing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Section:
    index: int
    title: str | None
    sentences: list[str]
    sentence_offset: int          # global index of this section's first sentence
    origin: str                   # "heading" | "embedding" | "single", with "+split" if re-split for length

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

    @property
    def sentence_indices(self) -> range:
        return range(self.sentence_offset, self.sentence_offset + len(self.sentences))


@dataclass
class RetrievedSentence:
    sentence_index: int
    section_index: int
    text: str
    similarity: float


@dataclass
class SectionSummary:
    section_index: int
    model_input: str
    summary: str
    sentences: list[str]
    context: list[RetrievedSentence]
    generated: bool               # False => section was short and passed through verbatim
    input_truncated: bool
    context_leaks: list[int] = field(default_factory=list)  # kept summary sentence ids matched better by context than by section
    dropped: list[tuple[str, str]] = field(default_factory=list)  # (generated sentence, why it was removed)
    notes: list[str] = field(default_factory=list)                # e.g. a fallback that replaced the generation
    protected: list[str] = field(default_factory=list)            # source sentences re-attached verbatim (see Stage 2)


@dataclass
class ClaimRef:
    section_index: int
    sentence_index: int           # index within that section's summary (or source) sentences
    text: str


@dataclass
class Contradiction:
    claim_a: ClaimRef
    claim_b: ClaimRef
    score: float                  # P(contradiction) aggregated over both directions
    score_ab: float               # premise=a, hypothesis=b
    score_ba: float               # premise=b, hypothesis=a
    support_a: float = 0.0        # entailment of a by its own section's source text
    support_b: float = 0.0
    evidence_a: str = ""
    evidence_b: str = ""
    resolution: str = "pending"   # kept_a | kept_b | unresolved | superseded | diagnostic
    reason: str = ""
    kind: str = "summary"         # summary (summary vs summary) | one_sided (summary vs sibling source) | diagnostic

    @property
    def kept(self) -> ClaimRef | None:
        return {"kept_a": self.claim_a, "kept_b": self.claim_b}.get(self.resolution)

    @property
    def rejected(self) -> ClaimRef | None:
        return {"kept_a": self.claim_b, "kept_b": self.claim_a}.get(self.resolution)


@dataclass
class PairStats:
    """How much of the O(n^2) comparison space a Stage 3 check actually looked at."""

    total: int = 0              # all cross-section pairs that exist
    checked: int = 0            # pairs actually scored with NLI (2 calls each)
    skipped_top_k: int = 0      # dropped because a row already had candidate_top_k better matches
    skipped_budget: int = 0     # dropped because max_pairs was reached
    skipped_entity: int = 0     # dropped because the pair shared no entity-like token
    skipped_comparative: int = 0  # dropped because either sentence was framed as a comparison
    skipped_topic: int = 0      # dropped because the pair shared no content word

    @property
    def skipped_similarity(self) -> int:
        return max(self.total - self.checked - self.skipped_top_k - self.skipped_budget
                   - self.skipped_entity - self.skipped_comparative - self.skipped_topic, 0)

    @property
    def nli_calls(self) -> int:
        return self.checked * 2   # every pair is scored in both directions


@dataclass
class ContradictionReport:
    pairs_total: int
    pairs_checked: int
    contradictions: list[Contradiction]
    corrected_sentences: list[list[str]]      # per section, after resolution
    flagged: dict[str, str] = field(default_factory=dict)  # "section:sentence" -> note (action=flag or unresolved)
    one_sided: list[Contradiction] = field(default_factory=list)   # summary claim vs sibling SOURCE sentence
    one_sided_checked: int = 0
    pair_stats: PairStats = field(default_factory=PairStats)
    one_sided_stats: PairStats = field(default_factory=PairStats)
    unscored: int = 0                    # flagged pairs left unresolved because the resolution budget was reached
    resolution_budget: int = 0           # max(max_resolutions, max_resolutions_per_section x sections)

    @property
    def all_contradictions(self) -> list[Contradiction]:
        return self.contradictions + self.one_sided


@dataclass
class MergeRound:
    inputs: list[str]
    outputs: list[str]
    round_number: int = 1
    introduced: list["Contradiction"] = field(default_factory=list)   # contradictions this round created
    contexts: list[list[str]] = field(default_factory=list)   # per group: source sentences given as context


@dataclass
class MergeResult:
    mode: str
    input_sentences: list[str]
    removed_duplicates: list[tuple[str, str, float]]   # (removed, duplicate_of, similarity)
    rounds: list[MergeRound]
    summary: str
    sentences: list[str]
    resurrected: list[tuple[str, str, float]]          # (final sentence, rejected claim, entailment)

    @property
    def introduced_by_merge(self) -> list["Contradiction"]:
        return [c for r in self.rounds for c in r.introduced]


@dataclass
class Citation:
    sentence_indices: list[int]   # global source sentence ids forming the premise
    section_index: int
    section_title: str | None
    text: str
    similarity: float
    entailment: float
    contradiction: float


@dataclass
class ProvenanceRecord:
    index: int
    sentence: str
    status: str                   # supported | weakly_supported | unsupported
    similarity: float
    entailment: float
    citation: Citation | None
    candidates: list[Citation]
    flags: list[str] = field(default_factory=list)
    stage3_notes: list[str] = field(default_factory=list)


@dataclass
class PipelineResult:
    document_name: str
    sections: list[Section]
    section_summaries: list[SectionSummary]
    contradictions: ContradictionReport
    source_diagnostic: ContradictionReport | None
    merge: MergeResult
    provenance: list[ProvenanceRecord]
    timings: dict[str, float]
    config: dict[str, Any]
    stats: dict[str, Any] = field(default_factory=dict)   # counts of work done; see pipeline.py

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # asdict drops properties; add the ones an auditor needs.
        for key in ("contradictions", "one_sided"):
            for c_dict, c in zip(data["contradictions"][key], getattr(self.contradictions, key)):
                c_dict["kept"] = asdict(c.kept) if c.kept else None
                c_dict["rejected"] = asdict(c.rejected) if c.rejected else None
        return data
