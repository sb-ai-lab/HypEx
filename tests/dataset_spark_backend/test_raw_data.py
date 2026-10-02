import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")

import pyspark.pandas as ps
import pyspark.sql as psql

from hypex import AATest, ABTest
from hypex.dataset import (
    Dataset,
    InfoRole,
    TargetRole,
    TreatmentRole,
)
from hypex.utils import BackendsEnum


def _roles():
    return {"a": InfoRole(), "b": TargetRole()}


def _pdf():
    return pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})


@pytest.fixture
def spark_ds(spark_session):
    return Dataset(
        roles=_roles(),
        data=_pdf(),
        backend=BackendsEnum.spark,
        session=spark_session,
    )


def test_spark_data_is_spark_dataframe(spark_ds):
    assert isinstance(spark_ds.data, psql.DataFrame)
    assert spark_ds.data.columns == ["a", "b"]


def test_spark_raw_data_is_pandas_on_spark(spark_ds):
    assert isinstance(spark_ds.raw_data, ps.DataFrame)
    assert spark_ds.raw_data is spark_ds.backend_data.data


@pytest.mark.parametrize("kind", ["sp", "ps", "pd"])
def test_spark_data_setter_accepts(spark_ds, spark_session, kind):
    new = pd.DataFrame({"a": [7, 8, 9], "b": [4.0, 5.0, 6.0]})
    if kind == "sp":
        value = spark_session.createDataFrame(new)
    elif kind == "ps":
        value = ps.from_pandas(new)
    else:
        value = new.set_axis([5, 6, 7])
    spark_ds.data = value
    assert isinstance(spark_ds.raw_data, ps.DataFrame)
    assert spark_ds.raw_data["a"].to_list() == [7, 8, 9]
    if kind == "pd":
        assert spark_ds.index.to_list() == [5, 6, 7]


def test_spark_data_setter_rejects_list(spark_ds):
    with pytest.raises(TypeError):
        spark_ds.data = [1]


def test_spark_setitem_writes_raw(spark_ds):
    spark_ds["a"] = [7, 8, 9]
    assert spark_ds.raw_data["a"].to_list() == [7, 8, 9]


def test_spark_append_and_mask(spark_ds):
    assert len(spark_ds.append([spark_ds])) == 6
    assert len(spark_ds[spark_ds["a"] > 1]) == 2


def test_pandas_data_unchanged():
    ds = Dataset(roles=_roles(), data=_pdf())
    assert type(ds.data) is pd.DataFrame
    assert ds.data is ds.raw_data
    assert ds.raw_data is ds.backend_data.data


def _ab_frame():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "uid": np.arange(n),
            "treat": rng.integers(0, 2, n),
            "y": rng.normal(size=n),
        }
    )


def _ab_roles():
    return {
        "uid": InfoRole(),
        "treat": TreatmentRole(),
        "y": TargetRole(),
    }


def test_abtest_spark_matches_pandas(spark_session):
    df = _ab_frame()
    ds_pd = Dataset(roles=_ab_roles(), data=df)
    ds_sp = Dataset(
        roles=_ab_roles(),
        data=df,
        backend=BackendsEnum.spark,
        session=spark_session,
    )
    r_pd = ABTest().execute(ds_pd).resume
    r_sp = ABTest().execute(ds_sp).resume
    a = r_pd.raw_data.reset_index(drop=True)
    b = r_sp.raw_data.reset_index(drop=True)
    assert list(a.columns) == list(b.columns)
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]):
            np.testing.assert_allclose(
                a[col].astype(float), b[col].astype(float), rtol=1e-9
            )
        else:
            assert a[col].tolist() == b[col].tolist()


def test_aatest_spark_runs(spark_session):
    df = _ab_frame()
    roles = {"uid": InfoRole(), "y": TargetRole(), "treat": InfoRole()}
    ds_pd = Dataset(roles=roles, data=df)
    ds_sp = Dataset(
        roles=roles,
        data=df,
        backend=BackendsEnum.spark,
        session=spark_session,
    )
    r_pd = AATest(n_iterations=3).execute(ds_pd)
    r_sp = AATest(n_iterations=3).execute(ds_sp)
    assert list(r_sp.resume.columns) == list(r_pd.resume.columns)
    assert isinstance(r_sp.best_split.data, psql.DataFrame)
