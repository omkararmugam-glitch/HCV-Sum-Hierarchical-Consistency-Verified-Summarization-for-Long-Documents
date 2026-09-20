from hcv_sum.merging import deduplicate, merge_summaries, pack_groups

from conftest import FakeNLI, FakeSummarizer, make_cfg


def test_deduplicate_drops_later_near_duplicate(embedder):
    sections = [["Revenue fell eighteen percent.", "Costs rose."], ["Revenue fell eighteen percent!", "Demand weak."]]
    out, removed = deduplicate(sections, embedder, threshold=0.95)
    assert out == [["Revenue fell eighteen percent.", "Costs rose."], ["Demand weak."]]
    assert removed[0][0] == "Revenue fell eighteen percent!"


def test_pack_groups_respects_budget():
    fs = FakeSummarizer()
    groups = pack_groups(["one two three", "four five", "six seven eight nine"], fs, budget=5)
    assert groups == ["one two three four five", "six seven eight nine"]


def test_pack_groups_respects_max_blocks():
    fs = FakeSummarizer()
    assert pack_groups(["a", "b", "c", "d", "e"], fs, budget=100, max_blocks=2) == ["a b", "c d", "e"]


def test_default_mode_is_extractive(cfg):
    # Measured on the samples: abstractive fusion with DistilBART dropped 2-5 of 5-8 sections.
    assert cfg.merging.mode == "extractive"


def test_extractive_merge_preserves_document_order(embedder, nli):
    cfg = make_cfg("merging.mode=extractive").merging
    result = merge_summaries([["Alpha rose."], [], ["Gamma fell."]], [], FakeSummarizer(), embedder, nli, cfg)
    assert result.summary == "Alpha rose. Gamma fell."
    assert result.rounds == [] and result.sentences == ["Alpha rose.", "Gamma fell."]


def test_abstractive_merge_runs_multiple_rounds_until_one_group(embedder, nli):
    cfg = make_cfg("merging.mode=abstractive", "merging.max_input_tokens=14", "merging.min_new_tokens=1",
                   "merging.max_new_tokens=4",
                   "merging.max_length_ratio=1.0", "merging.min_length_ratio=0.1").merging
    fs = FakeSummarizer()
    sections = [["Alpha one two."], ["Beta three four."], ["Gamma five six."], ["Delta seven eight."]]
    result = merge_summaries(sections, [], fs, embedder, nli, cfg)
    assert len(result.rounds) >= 2
    assert len(result.rounds[-1].outputs) == 1 or len(result.rounds) == cfg.max_rounds
    assert len(result.rounds[0].inputs) > len(result.rounds[-1].inputs)


def test_resurrected_rejected_claim_is_reported(embedder):
    rejected = "Sensor revenue fell not because of the shutdown."
    fs = FakeSummarizer(fixed_output=rejected)                 # merger re-hallucinates the rejected claim
    cfg = make_cfg("merging.mode=abstractive", "merging.resurrection_similarity=0.5").merging
    result = merge_summaries([["Sensor revenue fell because of the shutdown."]], [rejected], fs, embedder,
                             FakeNLI(), cfg)
    assert len(result.resurrected) == 1 and result.resurrected[0][1] == rejected


def test_rejected_claim_absent_from_input_is_not_reported(embedder, nli):
    cfg = make_cfg("merging.mode=abstractive").merging
    result = merge_summaries([["Resin lead times improved."]], ["Sensor revenue fell not because of the shutdown."],
                             FakeSummarizer(), embedder, nli, cfg)
    assert result.resurrected == []


def test_rejected_duplicate_of_surviving_claim_is_not_resurrected(embedder):
    # A leaked copy of a true claim was rejected, but the original survives in another section.
    claim = "Sensor revenue fell eighteen percent."
    fs = FakeSummarizer(fixed_output=claim)
    cfg = make_cfg("merging.mode=abstractive", "merging.resurrection_similarity=0.5").merging
    result = merge_summaries([[claim]], [claim], fs, embedder, FakeNLI(), cfg)
    assert result.resurrected == []


def test_abstractive_merge_drops_incomplete_tail(embedder, nli):
    fs = FakeSummarizer(fixed_output="Alpha rose sharply. Gamma fell because of")
    cfg = make_cfg("merging.mode=abstractive").merging
    result = merge_summaries([["Alpha rose sharply."], ["Gamma fell."]], [], fs, embedder, nli, cfg)
    assert result.sentences == ["Alpha rose sharply."] and result.summary == "Alpha rose sharply."


def test_extractive_merge_keeps_sentences_without_terminal_punctuation(embedder, nli):
    """Regression: joining + re-splitting fused sentences that lack a final full stop."""
    cfg = make_cfg("merging.mode=extractive").merging
    sections = [["revenue grew four percent this quarter"], ["costs fell three percent overall"]]
    result = merge_summaries(sections, [], FakeSummarizer(), embedder, nli, cfg)
    assert result.sentences == ["revenue grew four percent this quarter", "costs fell three percent overall"]


def test_extractive_merge_sentences_match_the_kept_claims(embedder, nli):
    cfg = make_cfg("merging.mode=extractive").merging
    sections = [["Alpha rose sharply.", "Beta held steady."], [], ["Gamma fell by half."]]
    result = merge_summaries(sections, [], FakeSummarizer(), embedder, nli, cfg)
    assert result.sentences == ["Alpha rose sharply.", "Beta held steady.", "Gamma fell by half."]
