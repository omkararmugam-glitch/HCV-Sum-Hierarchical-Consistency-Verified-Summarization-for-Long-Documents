"""Configuration loading.

``config/default.yaml`` is the single source of truth for model names and
thresholds. The dataclasses below deliberately carry NO default values, so a
threshold can never silently live in two places; loading fails loudly if a key
is missing or unknown.
"""

from __future__ import annotations

import copy
import os
import typing
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
CONFIG_ENV_VAR = "HCV_SUM_CONFIG"


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelsConfig:
    embedder: str
    nli: str
    summarizer: str
    merge_summarizer: str
    device: str
    batch_size: int
    torch_threads: int


@dataclass(frozen=True)
class SegmentationConfig:
    use_structure: bool
    min_headings: int
    max_heading_words: int
    detect_markdown: bool
    detect_numbered: bool
    detect_allcaps: bool
    window: int
    depth_std_factor: float
    min_depth: float
    min_segment_sentences: int
    max_section_tokens: int
    max_sections: int


@dataclass(frozen=True)
class SummarizationConfig:
    strip_invented_attribution: bool
    context_mode: str
    instruct_prompt: str
    retrieval_scoring: str
    context_top_k: int
    context_min_similarity: float
    max_input_tokens: int
    max_context_tokens: int
    passthrough_tokens: int
    max_length_ratio: float
    min_new_tokens: int
    max_new_tokens: int
    num_beams: int
    no_repeat_ngram_size: int
    length_penalty: float
    leak_margin: float
    drop_context_leaks: bool
    batch_size: int
    protect_cross_referenced: bool
    protection_coverage_similarity: float


@dataclass(frozen=True)
class ContradictionConfig:
    enabled: bool
    pair_min_similarity: float
    threshold: float
    direction_aggregation: str
    resolution_margin: float
    support_top_k: int
    support_window: int
    action: str
    min_claim_words: int
    candidate_top_k: int
    max_pairs: int
    max_resolutions: int
    max_resolutions_per_section: int
    diagnostic_max_pairs: int
    dense_pair_limit: int
    require_shared_entity: bool
    require_shared_content_word: bool
    require_verb_or_number: bool
    strip_reporting_frames: bool
    strip_attribution_frames: bool
    verbless_fragment_max_words: int
    skip_comparative_framing: bool
    check_sibling_sources: bool
    sibling_source_skip_similarity: float


@dataclass(frozen=True)
class MergingConfig:
    mode: str
    dedup_similarity: float
    max_input_tokens: int
    max_length_ratio: float
    min_length_ratio: float
    min_new_tokens: int
    max_new_tokens: int
    max_rounds: int
    max_group_sections: int
    coverage_guard: bool
    coverage_guard_similarity: float
    extractive_max_per_section: int
    context_anchoring: bool
    context_top_k_per_block: int
    context_min_similarity: float
    context_max_tokens: int
    merge_prompt: str
    merge_instruct_prompt: str
    merge_instruct_context: str
    check_between_rounds: bool
    resurrection_similarity: float
    resurrection_entailment: float


@dataclass(frozen=True)
class ProvenanceConfig:
    top_k: int
    min_candidate_similarity: float
    window: int
    supported_entailment: float
    weak_entailment: float
    source_contradiction_flag: float
    link_similarity: float
    flag_novel_tokens: bool
    novel_token_min_length: int


@dataclass(frozen=True)
class MultidocConfig:
    max_summary_sentences: int
    salience_measure: str
    salience_knn: int
    supersede_by_period: bool


@dataclass(frozen=True)
class PreprocessingConfig:
    dialogue_to_description: bool
    organisation_reference: str


@dataclass(frozen=True)
class ScaleConfig:
    large_document_sections: int
    large_document_merge_mode: str
    checkpoint: str
    checkpoint_dir: str
    progress_every_sections: int
    progress_every_pairs: int
    progress_every_seconds: float
    est_seconds_per_section: float
    est_ms_per_nli_pair: float
    est_claims_per_section: float


@dataclass(frozen=True)
class Config:
    models: ModelsConfig
    segmentation: SegmentationConfig
    summarization: SummarizationConfig
    contradiction: ContradictionConfig
    merging: MergingConfig
    provenance: ProvenanceConfig
    scale: ScaleConfig
    preprocessing: PreprocessingConfig
    multidoc: MultidocConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_CHOICES = {
    ("summarization", "context_mode"): {"none", "append", "instruct"},
    ("summarization", "retrieval_scoring"): {"max", "mean"},
    ("contradiction", "direction_aggregation"): {"max", "min", "mean"},
    ("contradiction", "action"): {"remove", "flag"},
    ("merging", "mode"): {"abstractive", "extractive"},
    ("merging", "merge_prompt"): {"none", "instruct"},
    ("multidoc", "salience_measure"): {"pagerank", "betweenness"},
    ("scale", "large_document_merge_mode"): {"abstractive", "extractive"},
    ("scale", "checkpoint"): {"auto", "always", "never"},
}


def _build(cls: type, data: Any, path: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"'{path or '<root>'}' must be a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in fields(cls)]
    missing = sorted(set(names) - set(data))
    unknown = sorted(set(data) - set(names))
    if missing:
        raise ConfigError(f"missing config keys under '{path or '<root>'}': {missing}")
    if unknown:
        raise ConfigError(f"unknown config keys under '{path or '<root>'}': {unknown}")
    kwargs = {}
    for name in names:
        hint, value = hints[name], data[name]
        key = f"{path}.{name}" if path else name
        if is_dataclass(hint):
            kwargs[name] = _build(hint, value, key)
            continue
        if hint is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if (hint is int and isinstance(value, bool)) or not isinstance(value, hint):
            raise ConfigError(f"'{key}' must be {hint.__name__}, got {value!r}")
        allowed = _CHOICES.get((path, name))
        if allowed and value not in allowed:
            raise ConfigError(f"'{key}' must be one of {sorted(allowed)}, got {value!r}")
        kwargs[name] = value
    return cls(**kwargs)


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``section.key=value`` overrides (value parsed as YAML) to a raw config dict."""
    raw = copy.deepcopy(raw)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override must look like section.key=value, got {item!r}")
        dotted, value = item.split("=", 1)
        parts = dotted.strip().split(".")
        node = raw
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                raise ConfigError(f"unknown config section in override {item!r}")
            node = node[part]
        if parts[-1] not in node:
            raise ConfigError(f"unknown config key in override {item!r}")
        node[parts[-1]] = yaml.safe_load(value)
    return raw


def resolve_config_path(path: str | os.PathLike | None = None) -> Path:
    candidate = Path(path or os.environ.get(CONFIG_ENV_VAR) or DEFAULT_CONFIG_PATH)
    if not candidate.is_file():
        raise ConfigError(
            f"config file not found: {candidate}. Pass --config, set {CONFIG_ENV_VAR}, "
            "or install the package in editable mode (pip install -e .) so config/default.yaml is found."
        )
    return candidate


def load_raw_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    with open(resolve_config_path(path), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(path: str | os.PathLike | None = None, overrides: list[str] | None = None) -> Config:
    raw = load_raw_config(path)
    if overrides:
        raw = apply_overrides(raw, overrides)
    return _build(Config, raw, "")
