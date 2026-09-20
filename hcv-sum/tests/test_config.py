import pytest
import yaml

from hcv_sum.config import ConfigError, apply_overrides, load_config, load_raw_config


def test_default_config_loads_and_is_complete():
    cfg = load_config()
    assert cfg.models.nli and cfg.models.embedder and cfg.models.summarizer
    assert 0 < cfg.contradiction.threshold < 1
    assert cfg.provenance.weak_entailment < cfg.provenance.supported_entailment


def test_override_is_parsed_as_yaml():
    cfg = load_config(overrides=["contradiction.threshold=0.4", "merging.mode=extractive"])
    assert cfg.contradiction.threshold == pytest.approx(0.4)
    assert cfg.merging.mode == "extractive"


def test_unknown_override_key_rejected():
    with pytest.raises(ConfigError):
        apply_overrides(load_raw_config(), ["contradiction.thresold=0.4"])


def test_missing_key_rejected(tmp_path):
    raw = load_raw_config()
    del raw["provenance"]["top_k"]
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="top_k"):
        load_config(path)


def test_invalid_choice_rejected():
    with pytest.raises(ConfigError, match="merging.mode"):
        load_config(overrides=["merging.mode=magic"])


def test_wrong_type_rejected():
    with pytest.raises(ConfigError):
        load_config(overrides=["provenance.top_k=three"])
