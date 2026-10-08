"""Tests for Executor id generation, parameter setting and id round-trip."""

from __future__ import annotations

import pytest

from hypex.dataset import ExperimentData
from hypex.executor.executor import Executor
from hypex.utils import ID_SPLIT_SYMBOL


class _Plain(Executor):
    def execute(self, data: ExperimentData) -> ExperimentData:
        return data


class _Hashed(Executor):
    """Executor whose params hash depends on two attributes."""

    def __init__(self, alpha: float = 0.05, mode: str = "a", key=""):
        self.alpha = alpha
        self.mode = mode
        super().__init__(key)

    def _generate_params_hash(self):
        self._params_hash = f"alpha{self.alpha}|mode{self.mode}"

    def execute(self, data: ExperimentData) -> ExperimentData:
        return data


class _Other(_Plain):
    pass


def test_id_has_class_hash_key_parts() -> None:
    ex = _Hashed(alpha=0.1, mode="b", key="k")
    assert ex.id == ID_SPLIT_SYMBOL.join(["_Hashed", "alpha0.1|modeb", "k"])
    assert ex.params_hash == "alpha0.1|modeb"


def test_default_id_has_empty_hash_and_key() -> None:
    assert _Plain().id == ID_SPLIT_SYMBOL.join(["_Plain", "", ""])


def test_id_split_symbol_in_key_is_escaped() -> None:
    ex = _Plain(key=f"a{ID_SPLIT_SYMBOL}b")
    parts = ex.id.split(ID_SPLIT_SYMBOL)
    assert len(parts) == 3
    assert parts[2] == "a|b"


def test_key_setter_regenerates_id() -> None:
    ex = _Plain()
    ex.key = "new"
    assert ex.id.endswith(f"{ID_SPLIT_SYMBOL}new")


def test_id_for_name_has_no_split_symbol() -> None:
    ex = _Hashed(key="k")
    assert ID_SPLIT_SYMBOL not in ex.id_for_name
    assert ex.id_for_name == ex.id.replace(ID_SPLIT_SYMBOL, "_")


def test_same_params_same_id_different_params_different_id() -> None:
    assert _Hashed(alpha=0.1).id == _Hashed(alpha=0.1).id
    assert _Hashed(alpha=0.1).id != _Hashed(alpha=0.2).id


def test_set_params_flat_dict_updates_attrs_and_id() -> None:
    ex = _Hashed()
    old_id = ex.id
    ex.set_params({"alpha": 0.5})
    assert ex.alpha == 0.5
    assert ex.id != old_id
    assert "alpha0.5" in ex.id


def test_set_params_ignores_unknown_attribute() -> None:
    ex = _Hashed()
    ex.set_params({"nonexistent": 1})
    assert not hasattr(ex, "nonexistent")


def test_set_params_by_class_applies_only_to_matching_instance() -> None:
    hashed, plain = _Hashed(), _Plain()
    params = {_Hashed: {"alpha": 0.9}}
    hashed.set_params(params)
    plain.set_params(params)
    assert hashed.alpha == 0.9
    assert not hasattr(plain, "alpha")


def test_set_params_class_dict_applies_to_subclass_instances() -> None:
    other = _Other()
    other.set_params({_Plain: {"key": "from_base"}})
    assert other.key == "from_base"


def test_set_params_invalid_key_type_raises() -> None:
    with pytest.raises(ValueError, match="params must be"):
        _Plain().set_params({1: {"a": 1}})


@pytest.mark.xfail(
    strict=True,
    reason="Issue: init_from_hash sets _params_hash but _generate_id immediately "
    "recomputes it via _generate_params_hash, so the hash is lost",
)
def test_build_from_id_round_trips_params_hash() -> None:
    original = _Plain()
    original.init_from_hash("custom")
    rebuilt = _Plain.build_from_id(original.id)
    assert rebuilt.params_hash == "custom"
    assert isinstance(rebuilt, _Plain)


def test_build_from_id_wrong_class_raises() -> None:
    with pytest.raises(ValueError, match="is not a valid"):
        _Other.build_from_id(_Plain().id)


@pytest.mark.xfail(
    strict=True,
    reason="Issue: init_from_hash hash is overwritten by _generate_params_hash",
)
def test_init_from_hash_regenerates_id() -> None:
    ex = _Plain(key="k")
    ex.init_from_hash("h")
    assert ex.id == ID_SPLIT_SYMBOL.join(["_Plain", "h", "k"])


def test_executor_is_abstract() -> None:
    with pytest.raises(TypeError):
        Executor()  # type: ignore[abstract]
