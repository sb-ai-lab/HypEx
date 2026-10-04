"""Executor repr/params, MLExecutor.calc/execute and MinSampleSize paths.

Some tests use the ``fixed_count_groups`` fixture: ``PandasDataset.count_groups``
currently emits a FutureWarning (``int(Series)``), which the pytest filters turn
into an error. The fixture swaps in a correct implementation so the code *downstream*
of ``len(GroupedDataset)`` can still be verified; the bug itself is recorded by the
strict xfails in other test files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hypex.dataset import (
    AdditionalMatchingRole,
    Dataset,
    ExperimentData,
    FeatureRole,
    TargetRole,
    TreatmentRole,
)
from hypex.dataset.backends import PandasDataset
from hypex.executor.calculators import MinSampleSize
from hypex.executor.executor import Calculator, Executor, MLExecutor
from hypex.utils import ID_SPLIT_SYMBOL, BackendsEnum, NotSuitableFieldError
from hypex.utils.constants import NAME_BORDER_SYMBOL


@pytest.fixture
def fixed_count_groups(monkeypatch):
    monkeypatch.setattr(
        PandasDataset,
        "count_groups",
        lambda self, cols: 1 if not cols else len(self.data[cols].drop_duplicates()),
    )


# ---------------------------------------------------------------------------
# get_params / repr / html
# ---------------------------------------------------------------------------
class _Child(Executor):
    def __init__(self, depth: int = 1, key=""):
        self.depth = depth
        super().__init__(key)

    def execute(self, data):
        return data


class _Parent(Executor):
    def __init__(
        self, child=None, items=None, table=None, span=None, *args, key="", **kw
    ):
        self.child = child
        self.items = items
        self.table = table
        self.span = span
        super().__init__(key, **kw)

    def execute(self, data):
        return data


class _WithGetParams:
    def get_params(self):
        return {"inner": 3}


class _Holder(Executor):
    def __init__(self, thing=None, key=""):
        self.thing = thing
        super().__init__(key)

    def execute(self, data):
        return data


def test_get_params_reads_init_arguments() -> None:
    ex = _Child(depth=5)
    assert ex.get_params() == {"depth": 5, "key": ""}


def test_get_params_falls_back_to_default_when_attribute_missing() -> None:
    class _NoAttr(Executor):
        def __init__(self, missing: int = 7, key=""):
            super().__init__(key)

        def execute(self, data):
            return data

    assert _NoAttr().get_params()["missing"] == 7


def test_get_params_deep_collects_nested_get_params() -> None:
    ex = _Holder(thing=_WithGetParams())
    params = ex.get_params(deep=True)
    assert params[f"thing{NAME_BORDER_SYMBOL}inner"] == 3
    assert "thing" in params


def test_get_params_deep_ignores_classes() -> None:
    ex = _Holder(thing=_WithGetParams)
    assert ex.get_params(deep=True) == {"thing": _WithGetParams, "key": ""}


def test_repr_lists_parameters() -> None:
    assert repr(_Child(depth=2, key="k")) == "_Child(depth=2, key='k')"


def test_repr_shows_calc_kwargs_passed_via_kwargs() -> None:
    ex = _Parent(alpha=0.1)
    text = repr(ex)
    assert "calc_kwargs={'alpha': 0.1}" in text


def test_repr_without_params() -> None:
    class _Empty(Executor):
        def __init__(self):
            super().__init__()

        def execute(self, data):
            return data

    assert repr(_Empty()) == "_Empty()"
    assert "calc_kwargs" not in _Empty()._repr_params()


def test_html_value_renders_nested_executor_collapsed() -> None:
    html = Executor._html_value(_Child(depth=9))
    assert html.startswith("<details>") and "_Child" in html and "depth" in html


def test_html_value_range_is_compact() -> None:
    html = Executor._html_value(range(100000))
    assert "range(0, 100000)" in html and html.count("<tr>") == 0


def test_html_value_list_is_truncated() -> None:
    n = Executor._HTML_MAX_ITEMS
    html = Executor._html_value(list(range(n + 5)))
    assert html.count("<tr>") == n + 1  # shown rows + the truncation row
    assert "5" in html.split("<tr>")[-1]


def test_html_value_short_list_not_truncated() -> None:
    html = Executor._html_value([1, 2])
    assert html.count("<tr>") == 2 and "[0]" in html and "[1]" in html


def test_html_value_mapping_uses_class_names_for_class_keys() -> None:
    html = Executor._html_value({_Child: 1, "plain": 2})
    assert "<code>_Child</code>" in html
    assert "&#x27;plain&#x27;" in html or "'plain'" in html


def test_html_value_escapes_text() -> None:
    assert "&lt;b&gt;" in Executor._html_value("<b>")


def test_repr_html_contains_all_parameters() -> None:
    html = _Parent(
        child=_Child(), items=[1], table={"a": 1}, span=range(3)
    )._repr_html_()
    assert html.count("Parameter") >= 1
    for name in ("child", "items", "table", "span"):
        assert f"<code>{name}</code>" in html
    assert "<b>_Parent</b>" in html


# ---------------------------------------------------------------------------
# set_params / build_from_id
# ---------------------------------------------------------------------------
def test_set_params_invalid_key_type_raises() -> None:
    with pytest.raises(ValueError, match="dict of str"):
        _Child().set_params({1: {"depth": 3}})


def test_set_params_by_class_only_touches_instances() -> None:
    a, b = _Child(), _Holder()
    params = {_Child: {"depth": 8}}
    a.set_params(params)
    b.set_params(params)
    assert a.depth == 8
    assert not hasattr(b, "depth")


def test_set_params_ignores_unknown_attributes() -> None:
    ex = _Child()
    ex.set_params({"nonexistent": 1})
    assert not hasattr(ex, "nonexistent")


def test_build_from_id_wrong_class_raises() -> None:
    other = ID_SPLIT_SYMBOL.join(["Other", "", ""])
    with pytest.raises(ValueError, match="not a valid"):
        _Child.build_from_id(other)


def test_calculator_search_types_is_abstract() -> None:
    from hypex.utils import AbstractMethodError

    class _Calc(Calculator):
        @staticmethod
        def _inner_function(data, **kwargs):
            return data

        def execute(self, data):
            return data

    with pytest.raises(AbstractMethodError):
        _Calc().search_types


def test_check_test_data_requires_dataset() -> None:
    with pytest.raises(ValueError, match="test_data is needed"):
        Calculator._check_test_data(None)
    sentinel = object()
    assert Calculator._check_test_data(sentinel) is sentinel


def test_default_set_value_returns_data_unchanged() -> None:
    marker = object()
    assert _Child()._set_value(marker, 5) is marker


def test_id_for_name_replaces_split_symbol() -> None:
    ex = _Child(key="k")
    assert ID_SPLIT_SYMBOL not in ex.id_for_name
    assert ex.id_for_name == ex.id.replace(ID_SPLIT_SYMBOL, "_")


def test_calculator_calc_delegates_to_inner_function() -> None:
    class _Sum(Calculator):
        @staticmethod
        def _inner_function(data, **kwargs):
            return (data, kwargs)

        def execute(self, data):
            return data

    assert _Sum.calc("d", a=1) == ("d", {"a": 1})


# ---------------------------------------------------------------------------
# MLExecutor
# ---------------------------------------------------------------------------
class _Mean(MLExecutor):
    """Predicts, for each test row, the mean of the first feature of control."""

    def fit(self, X, Y=None):
        return self

    def predict(self, X):
        return X

    @classmethod
    def _inner_function(cls, data, test_data=None, target_data=None, **kwargs):
        value = float(data.backend_data.data["f"].mean())
        if target_data is not None:
            value += float(target_data.backend_data.data["y"].mean())
        out = pd.DataFrame({"pred": [value] * len(test_data)}, index=test_data.index)
        return Dataset(
            roles={"pred": AdditionalMatchingRole()},
            data=out,
            backend=BackendsEnum.pandas,
        )


def _ml_data(with_target: bool = True) -> ExperimentData:
    df = pd.DataFrame(
        {
            "t": [0, 0, 0, 1, 1, 1],
            "f": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "y": [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        }
    )
    roles = {"t": TreatmentRole(), "f": FeatureRole()}
    if with_target:
        roles["y"] = TargetRole()
    return ExperimentData(Dataset(roles=roles, data=df, backend=BackendsEnum.pandas))


def test_mlexecutor_defaults_and_search_types() -> None:
    ex = _Mean()
    assert ex.search_types == [int, float]
    assert isinstance(ex.target_role, TargetRole)
    assert ex.score is not None
    with pytest.raises(NotImplementedError):
        ex.score(None, None)


def test_mlexecutor_get_fields() -> None:
    group, target = _Mean(grouping_role=TreatmentRole())._get_fields(_ml_data())
    assert group == ["t"] and target == ["y"]


def test_execute_inner_function_with_target_field_splits_target() -> None:
    ctrl = Dataset(
        roles={"f": FeatureRole(), "y": TargetRole()},
        data=pd.DataFrame({"f": [1.0, 3.0], "y": [10.0, 20.0]}),
        backend=BackendsEnum.pandas,
    )
    test = Dataset(
        roles={"f": FeatureRole(), "y": TargetRole()},
        data=pd.DataFrame({"f": [5.0], "y": [0.0]}),
        backend=BackendsEnum.pandas,
    )
    result = _Mean._execute_inner_function(
        [(0, ctrl), (1, test)], tmp_roles={}, target_field="y"
    )
    assert result.backend_data.data["pred"].tolist() == [2.0 + 15.0]


def test_execute_inner_function_without_target() -> None:
    ctrl = Dataset(
        roles={"f": FeatureRole()},
        data=pd.DataFrame({"f": [1.0, 3.0]}),
        backend=BackendsEnum.pandas,
    )
    test = Dataset(
        roles={"f": FeatureRole()},
        data=pd.DataFrame({"f": [5.0, 6.0]}),
        backend=BackendsEnum.pandas,
    )
    result = _Mean._execute_inner_function([(0, ctrl), (1, test)], tmp_roles={})
    assert result.backend_data.data["pred"].tolist() == [2.0, 2.0]


def test_mlexecutor_calc_single_group_raises(fixed_count_groups) -> None:
    ds = Dataset(
        roles={"t": TreatmentRole(), "f": FeatureRole()},
        data=pd.DataFrame({"t": [1, 1], "f": [1.0, 2.0]}),
        backend=BackendsEnum.pandas,
    )
    with pytest.raises(NotSuitableFieldError):
        _Mean.calc(ds, group_field="t", features_fields="f")


def test_mlexecutor_calc_groups_and_predicts(fixed_count_groups) -> None:
    data = _ml_data()
    result = _Mean.calc(data.ds, group_field="t", features_fields=["f"])
    # control mean(f) = 2 is predicted for each of the 3 test rows
    assert result.backend_data.data["pred"].tolist() == [2.0, 2.0, 2.0]


def test_mlexecutor_execute_stores_additional_fields(fixed_count_groups) -> None:
    data = _ml_data()
    ex = _Mean(grouping_role=TreatmentRole())
    out = ex.execute(data)
    assert ex.key == "y"
    stored = [c for c in out.additional_fields.columns if ex.id in c]
    assert len(stored) == 1
    col = out.additional_fields.backend_data.data[stored[0]]
    assert col.dropna().tolist() == [2.0, 2.0, 2.0]


def test_mlexecutor_execute_uses_prebuilt_groups() -> None:
    data = _ml_data()
    ctrl = Dataset(
        roles={"f": FeatureRole(), "y": TargetRole()},
        data=pd.DataFrame({"f": [100.0], "y": [1.0]}),
        backend=BackendsEnum.pandas,
    )
    test = Dataset(
        roles={"f": FeatureRole(), "y": TargetRole()},
        data=pd.DataFrame({"f": [0.0], "y": [0.0]}, index=[3]),
        backend=BackendsEnum.pandas,
    )
    data.groups["t"] = {0: ctrl, 1: test}
    ex = _Mean(grouping_role=TreatmentRole())
    # patched: len(list-of-pairs) == 2 so no count_groups is needed
    out = ex.execute(data)
    col = next(c for c in out.additional_fields.columns if ex.id in c)
    assert out.additional_fields.backend_data.data[col].dropna().tolist() == [100.0]


def test_mlexecutor_execute_without_targets_and_tmp_roles_returns_data() -> None:
    data = _ml_data(with_target=False)
    data.ds.tmp_roles = {"f": FeatureRole()}
    ex = _Mean(grouping_role=TreatmentRole())
    out = ex.execute(data)
    assert out is data
    assert ex.key == ""


# ---------------------------------------------------------------------------
# MinSampleSize
# ---------------------------------------------------------------------------
@pytest.fixture
def two_groups() -> Dataset:
    rng = np.random.RandomState(0)
    df = pd.DataFrame(
        {
            "g": ["a"] * 100 + ["b"] * 100,
            "y": np.r_[rng.normal(0, 2, 100), rng.normal(0, 4, 100)],
        }
    )
    return Dataset(
        roles={"g": TreatmentRole(), "y": TargetRole()},
        data=df,
        backend=BackendsEnum.pandas,
    )


def test_variance_by_group_matches_pandas_var(two_groups) -> None:
    df = two_groups.backend_data.data
    parts = [
        (
            g,
            Dataset(
                roles={"y": TargetRole()},
                data=df[df.g == g][["y"]],
                backend=BackendsEnum.pandas,
            ),
        )
        for g in ("a", "b")
    ]
    got = MinSampleSize._variance_by_group(parts, "y")
    expected = [df.y[df.g == g].var() for g in ("a", "b")]
    assert got == pytest.approx(expected)


def test_min_sample_size_calc_equal_variance(two_groups, fixed_count_groups) -> None:
    calc = MinSampleSize(mde=1.0, equal_variance=True, quantile_1=2.0, quantile_2=-1.0)
    result = calc.calc(two_groups)
    df = two_groups.backend_data.data
    var = np.mean([df.y[df.g == g].var() for g in ("a", "b")])
    expected = int(2 * var * ((2.0 + 1.0) / 1.0) ** 2) + 1
    assert result["y"]["min sample size"] == expected
    assert result["overall"]["min sample size"] == expected
    assert calc.key == "y"


def test_min_sample_size_execute_delegates_to_calc(
    two_groups, fixed_count_groups
) -> None:
    calc = MinSampleSize(
        mde=1.0, equal_variance=True, variances=4.0, quantile_1=2.0, quantile_2=-1.0
    )
    result = calc.execute(ExperimentData(two_groups))
    assert result["y"]["min sample size"] == int(2 * 4.0 * 9.0) + 1


def test_min_sample_size_without_targets_and_tmp_roles_raises(
    fixed_count_groups,
) -> None:
    ds = Dataset(
        roles={"g": TreatmentRole()},
        data=pd.DataFrame({"g": ["a", "b"]}),
        backend=BackendsEnum.pandas,
    )
    ds.tmp_roles = {"g": TreatmentRole()}
    with pytest.raises(Exception, match="No target fields"):
        MinSampleSize(mde=1.0).calc(ds)


def test_min_sample_size_unequal_variance_via_calc(
    two_groups, fixed_count_groups
) -> None:
    calc = MinSampleSize(
        mde=8.0,
        equal_variance=False,
        quantile_1=1.5,
        quantile_2=-0.8,
        power_iteration_size=40,
        random_state=0,
    )
    result = calc.calc(two_groups)
    assert result["y"]["min sample size"] % 100 == 0
    assert result["overall"] == result["y"]


def test_min_sample_size_unequal_float_quantiles_expand_per_sample() -> None:
    # float quantiles are expanded to one value per group before the search
    n = MinSampleSize._inner_function(
        num_samples=2,
        mde=50.0,
        variances=[1.0, 1.0],
        quantile_1=1.5,
        quantile_2=-0.8,
        equal_variance=False,
        power_iteration_size=30,
        random_state=1,
    )
    assert n == 100  # a huge MDE is detected at the first step of 100


def test_min_sample_size_equal_variance_scalar_variance_accepted() -> None:
    n = MinSampleSize._inner_function(
        num_samples=2,
        mde=1.0,
        variances=4.0,
        quantile_1=[2.0, 2.0],
        quantile_2=[-1.0, -1.0],
        equal_variance=True,
    )
    assert n == int(2 * 4.0 * 9.0) + 1
