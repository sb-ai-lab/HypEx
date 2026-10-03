"""MultiTest on Spark and kwargs forwarding (statsmodels extensions)."""

from __future__ import annotations

import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from hypex.dataset import Dataset, StatisticRole
from hypex.extensions import MultiTest
from hypex.utils import ABNTestMethodsEnum, BackendsEnum
from hypex.utils.constants import ID_SPLIT_SYMBOL as S

RAW_P = [0.001, 0.008, 0.02, 0.04, 0.3, 0.7]


def _pvalues(p_values, backend=BackendsEnum.pandas, session=None) -> Dataset:
    index = [f"GroupTTest{S}hash{S}y{i}{S}b" for i in range(len(p_values))]
    return Dataset(
        roles={"p-value": StatisticRole()},
        data=pd.DataFrame({"p-value": p_values}, index=index),
        backend=backend,
        session=session,
    )


def _frame(ds) -> pd.DataFrame:
    data = ds.backend_data.data
    return data.to_pandas() if hasattr(data, "to_pandas") else data


@pytest.mark.spark
@pytest.mark.xfail(
    strict=True,
    reason="Issue: MultiTest._calc_spark converts the Spark Dataset to pandas without "
    "its composite string index, so test/field/group labels are lost (field becomes "
    "0..n-1, every p-value is its own family and the correction is a no-op)",
)
def test_multitest_spark_matches_statsmodels(spark_session) -> None:
    ds = _pvalues(RAW_P, BackendsEnum.spark, spark_session)
    result = _frame(MultiTest(ABNTestMethodsEnum.holm, 0.05).calc(ds))
    expected = multipletests(RAW_P, alpha=0.05, method="holm")
    assert result["new p-value"].astype(float).tolist() == pytest.approx(
        expected[1].tolist()
    )
    assert [str(v) == "True" for v in result["rejected"]] == expected[0].tolist()
    assert result["field"].tolist() == [f"y{i}" for i in range(len(RAW_P))]


def test_multitest_forwards_kwargs_to_statsmodels() -> None:
    with pytest.raises(TypeError):
        MultiTest(ABNTestMethodsEnum.holm).calc(_pvalues(RAW_P), not_a_param=1)


def test_multitest_index_parts_without_group() -> None:
    tests, fields, groups = MultiTest._index_parts([f"T{S}h"])
    assert (tests, fields, groups) == (["T"], [""], [""])


def test_quantile_unequal_variance_returns_one_value_per_group() -> None:
    from hypex.extensions import MultitestQuantile

    mtq = MultitestQuantile(iteration_size=400, equal_variance=False, random_state=0)
    q = mtq.quantile_of_marginal_distribution(3, 0.9, variances=[1.0, 4.0, 9.0])
    assert len(q) == 3
    assert (
        len({round(v, 6) for v in q}) == 3
    )  # different variances -> different quantiles


def test_quantile_spark_wrapper_delegates_with_raw_data(monkeypatch) -> None:
    from hypex.extensions import MultitestQuantile

    seen = {}

    def fake(self, data, **kwargs):
        seen["rows"] = len(data)
        seen["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr(MultitestQuantile, "_calc_pandas", fake)
    ds = _pvalues(RAW_P)
    assert MultitestQuantile()._calc_spark(ds, group_field="g") == "ok"
    assert seen == {"rows": len(RAW_P), "kwargs": {"group_field": "g"}}
