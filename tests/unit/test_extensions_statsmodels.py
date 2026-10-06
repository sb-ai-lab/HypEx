"""MultiTest on Spark and kwargs forwarding (statsmodels extensions)."""

from __future__ import annotations

import pandas as pd
import pytest
from statsmodels.stats.multitest import multipletests

from hypex.dataset import Dataset, InfoRole, StatisticRole
from hypex.extensions import MultiTest
from hypex.utils import ABNTestMethodsEnum, BackendsEnum
from hypex.utils.constants import ID_SPLIT_SYMBOL as S

RAW_P = [0.001, 0.008, 0.02, 0.04, 0.3, 0.7]


def _pvalues(p_values, backend=BackendsEnum.pandas, session=None) -> Dataset:
    index = [f"GroupTTest{S}hash{S}y{i}{S}b" for i in range(len(p_values))]
    frame = pd.DataFrame({"p-value": p_values}, index=index)
    if backend == BackendsEnum.spark:
        # createDataFrame(pandas) drops the pandas index; from_pandas keeps it
        import pyspark.pandas as ps

        frame = ps.from_pandas(frame)
    return Dataset(
        roles={"p-value": StatisticRole()},
        data=frame,
        backend=backend,
        session=session,
    )


def _frame(ds) -> pd.DataFrame:
    data = ds.backend_data.data
    return data.to_pandas() if hasattr(data, "to_pandas") else data


@pytest.mark.spark
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


def test_quantile_calc_on_pandas() -> None:
    from hypex.extensions import MultitestQuantile

    frame = pd.DataFrame(
        {"g": [0, 0, 0, 1, 1, 1], "y": [1.0, 2.0, 3.0, 9.0, 10.0, 11.0]}
    )
    ds = Dataset(roles={"g": InfoRole(), "y": StatisticRole()}, data=frame)
    result = MultitestQuantile(random_state=0).calc(
        ds, group_field="g", target_field="y", quantiles=0.1
    )
    assert result.columns == ["field", "accepted hypothesis"]


@pytest.mark.spark
def test_quantile_calc_on_spark_not_supported(spark_session) -> None:
    from hypex.extensions import MultitestQuantile

    ds = _pvalues(RAW_P, BackendsEnum.spark, spark_session)
    with pytest.raises(NotImplementedError, match="not supported on the Spark backend"):
        MultitestQuantile().calc(ds, group_field="g", target_field="p-value")
