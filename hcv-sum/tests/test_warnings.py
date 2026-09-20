"""The TorchScript deprecation filter must be narrow.

transformers' DeBERTa-v2 modelling code applies @torch.jit.script at import time, and torch 2.14
turned that notice into a user-visible FutureWarning. We suppress exactly that message and nothing
else; these tests fail if the filter is ever widened into a blanket ignore.
"""

import warnings

import pytest

from hcv_sum.cli import _TORCHSCRIPT_WARNINGS, silence_torchscript_deprecation

TORCH_MESSAGE = "`torch.jit.script` is deprecated. Please switch to `torch.compile` or `torch.export`."
PY314_MESSAGE = ("`torch.jit.script` is not supported in Python 3.14+ and may break. "
                 "Please switch to `torch.compile` or `torch.export`.")


def emit(message: str, category=FutureWarning, module: str = "torch.jit._script"):
    """Raise a warning as if it came from ``module`` (mirrors how torch raises it)."""
    warnings.warn_explicit(message, category, f"{module.replace('.', '/')}.py", 1, module=module, registry={})


@pytest.mark.parametrize("message", [TORCH_MESSAGE, PY314_MESSAGE])
def test_torchscript_deprecation_is_suppressed(message):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        silence_torchscript_deprecation()
        emit(message)
    assert caught == []


def test_unrelated_future_warning_is_not_suppressed():
    """A blanket 'ignore FutureWarning' would hide this one; the narrow filter must not."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        silence_torchscript_deprecation()
        emit("pandas: silent upcasting is deprecated", module="pandas.core.frame")
        warnings.warn("some future API change", FutureWarning)
    assert len(caught) == 2


def test_same_message_from_another_module_is_not_suppressed():
    """The filter is module-scoped: only torch.jit._script may raise it quietly."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        silence_torchscript_deprecation()
        emit(TORCH_MESSAGE, module="some_other_library.jit")
    assert len(caught) == 1


def test_other_categories_from_torch_are_not_suppressed():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        silence_torchscript_deprecation()
        emit("something else entirely", category=UserWarning)
        emit("another torch deprecation", category=DeprecationWarning)
    assert len(caught) == 2


def test_pytest_config_and_cli_filters_stay_in_sync():
    """pyproject's filterwarnings cannot import Python, so the patterns are duplicated there."""
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    configured = pyproject["tool"]["pytest"]["ini_options"]["filterwarnings"]
    for pattern in _TORCHSCRIPT_WARNINGS:
        assert any(pattern in entry for entry in configured), f"pyproject is missing a filter for {pattern}"
    assert "default" in configured, "warnings must stay visible by default"
    assert not any(entry.startswith("ignore::FutureWarning") for entry in configured), "no blanket ignores"


def test_pytest_run_itself_sees_no_torchscript_warning():
    """Belt and braces: under the project's pytest config the warning is already filtered."""
    with warnings.catch_warnings(record=True) as caught:
        emit(TORCH_MESSAGE)          # inherits the filters pytest installed from pyproject.toml
    assert caught == []
