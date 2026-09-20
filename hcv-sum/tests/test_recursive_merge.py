"""Stage 4 recursive batched merging.

The merge fuses at most ``max_group_sections`` blocks per summarizer call and repeats on the
outputs until one block remains. These tests pin the batching arithmetic, the multi-round
behaviour, and the between-round contradiction check.
"""

import pytest

from hcv_sum.merging import merge_summaries, pack_groups
from hcv_sum.types import MergeResult

from conftest import FakeEmbedder, FakeNLI, FakeSummarizer, make_cfg


# Each section must be lexically DISTINCT: sentences differing only by a number look identical to
# the fake embedder, and Stage 4 would deduplicate half of them before any merging happened.
SUBJECTS = ["freight", "resin", "tooling", "packaging", "warehousing", "calibration", "plating",
            "extrusion", "sintering", "lamination", "annealing", "moulding", "welding", "coating",
            "drilling", "polishing", "assembly", "inspection", "shipping", "recycling", "forging",
            "casting", "milling", "bonding", "curing", "etching", "trimming", "sealing",
            "stamping", "grinding", "bending", "cutting", "joining", "finishing", "washing",
            "drying", "sorting", "labelling", "palletising", "loading"]


def sections(n: int) -> list[list[str]]:
    assert n <= len(SUBJECTS)
    return [[f"The {SUBJECTS[i]} programme reached its milestone during the period under review."]
            for i in range(n)]


# These tests exercise BATCHING, so near-duplicate removal is switched off: templated test
# sentences are similar enough that dedup would drop half of them before any merging happened.
# Dedup itself is covered by test_merging.py.
NO_DEDUP = "merging.dedup_similarity=1.01"


def merge(n_sections: int, *overrides, summarizer=None) -> MergeResult:
    cfg = make_cfg("merging.mode=abstractive", NO_DEDUP, *overrides).merging
    return merge_summaries(sections(n_sections), [], summarizer or FakeSummarizer(),
                           FakeEmbedder(), FakeNLI(), cfg)


# ------------------------------------------------------------------ batching arithmetic

def test_pack_groups_batches_by_count():
    assert pack_groups([f"b{i}" for i in range(11)], FakeSummarizer(), budget=10_000, max_blocks=4) == [
        "b0 b1 b2 b3", "b4 b5 b6 b7", "b8 b9 b10"]


def test_pack_groups_respects_whichever_limit_binds_first():
    fs = FakeSummarizer()
    blocks = ["one two three four five", "six seven", "eight nine"]
    # token budget binds before the block count: at budget 3 no two blocks fit together
    assert len(pack_groups(blocks, fs, budget=3, max_blocks=99)) == 3
    # block count binds before the token budget
    assert len(pack_groups(blocks, fs, budget=10_000, max_blocks=2)) == 2


# ------------------------------------------------------------------ multi-round behaviour

def test_merge_runs_multiple_rounds_until_one_block_remains():
    result = merge(20, "merging.max_group_sections=4")
    assert len(result.rounds) >= 2, "20 sections cannot collapse to one block in a single round of 4"
    assert [r.round_number for r in result.rounds] == list(range(1, len(result.rounds) + 1))
    assert len(result.rounds[-1].outputs) == 1, "the last round must produce exactly one block"
    # each round must strictly reduce the number of blocks, or the recursion would not terminate
    counts = [len(r.inputs) for r in result.rounds] + [1]
    assert counts == sorted(counts, reverse=True), counts


@pytest.mark.parametrize("n_sections,batch,expected_first_round_groups", [
    (20, 4, 5), (20, 5, 4), (9, 5, 2), (3, 5, 1),
])
def test_first_round_group_count_follows_the_batch_size(n_sections, batch, expected_first_round_groups):
    result = merge(n_sections, f"merging.max_group_sections={batch}")
    assert len(result.rounds[0].inputs) == expected_first_round_groups


def test_single_batch_document_merges_in_one_round():
    result = merge(3, "merging.max_group_sections=5")
    assert len(result.rounds) == 1 and len(result.rounds[0].inputs) == 1


def test_generation_count_matches_the_number_of_groups_across_all_rounds():
    fs = FakeSummarizer()
    result = merge(20, "merging.max_group_sections=4", summarizer=fs)
    assert len(fs.calls) == sum(len(r.inputs) for r in result.rounds)


def test_max_rounds_stops_runaway_recursion():
    result = merge(40, "merging.max_group_sections=2", "merging.max_rounds=2")
    assert len(result.rounds) == 2
    assert result.summary, "a truncated recursion must still produce a summary"


def test_extractive_mode_does_no_rounds():
    cfg = make_cfg("merging.mode=extractive", NO_DEDUP).merging
    result = merge_summaries(sections(20), [], FakeSummarizer(), FakeEmbedder(), FakeNLI(), cfg)
    assert result.rounds == [] and len(result.sentences) == 20


def test_default_batch_size_is_bounded(cfg):
    assert cfg.merging.max_group_sections > 0, (
        "0 means 'fuse as many as fit', i.e. a flat single pass on long documents")


# ------------------------------------------------------------------ between-round checking

def test_contradiction_introduced_by_the_merger_is_detected_and_reported():
    """A merged block that contradicts another block is the merger's own doing, so it is flagged."""
    claim = "Every branch was converted to the new system."
    denial = "Every branch was converted to the new system not at all."
    outputs = iter([claim, denial, "Both branches reported progress."])

    class Inventing(FakeSummarizer):
        def summarize(self, text, *, min_new_tokens, max_new_tokens):
            self.calls.append({"text": text})
            return next(outputs, "Summary of the period.")

    cfg = make_cfg("merging.mode=abstractive", NO_DEDUP, "merging.max_group_sections=1",
                   "merging.max_rounds=2", "merging.check_between_rounds=true")
    result = merge_summaries(sections(2), [], Inventing(), FakeEmbedder(), FakeNLI(),
                             cfg.merging, contradiction_cfg=cfg.contradiction)
    introduced = result.introduced_by_merge
    assert introduced, "the merger produced two contradicting blocks; that must be reported"
    c = introduced[0]
    assert c.kind == "introduced_by_merge"
    assert c.resolution == "unresolved", "a merged block has no single source section to resolve against"
    assert "introduced by the merge round" in c.reason


def test_between_round_check_ignores_text_copied_from_its_input():
    """A conflict already present in the leaf claims belongs to Stage 3, not to the merger."""
    claim = "Every branch was converted to the new system."
    denial = "Every branch was converted to the new system not at all."
    leaves = [[claim], [denial]]
    passthrough = iter([claim, denial])

    class Copying(FakeSummarizer):
        def summarize(self, text, *, min_new_tokens, max_new_tokens):
            self.calls.append({"text": text})
            return next(passthrough, text)

    cfg = make_cfg("merging.mode=abstractive", NO_DEDUP, "merging.max_group_sections=1",
                   "merging.max_rounds=1")
    result = merge_summaries(leaves, [], Copying(), FakeEmbedder(), FakeNLI(),
                             cfg.merging, contradiction_cfg=cfg.contradiction)
    assert result.introduced_by_merge == []


def test_between_round_check_can_be_disabled():
    claim = "Every branch was converted to the new system."
    denial = "Every branch was converted to the new system not at all."
    outputs = iter([claim, denial, "Combined view of the period."])

    class Inventing(FakeSummarizer):
        def summarize(self, text, *, min_new_tokens, max_new_tokens):
            self.calls.append({"text": text})
            return next(outputs, "Summary.")

    cfg = make_cfg("merging.mode=abstractive", NO_DEDUP, "merging.max_group_sections=1",
                   "merging.max_rounds=2", "merging.check_between_rounds=false")
    result = merge_summaries(sections(2), [], Inventing(), FakeEmbedder(), FakeNLI(),
                             cfg.merging, contradiction_cfg=cfg.contradiction)
    assert result.introduced_by_merge == []


def test_no_contradiction_config_means_no_between_round_check():
    """merge_summaries must stay usable without a contradiction config (older call sites)."""
    result = merge(8, "merging.max_group_sections=2")
    assert result.summary and result.introduced_by_merge == []
