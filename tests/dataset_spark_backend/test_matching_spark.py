import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")

from hypex import Matching
from hypex.dataset import (
    Dataset,
    FeatureRole,
    InfoRole,
    TargetRole,
    TreatmentRole,
)
from hypex.utils import BackendsEnum


def _match_frame():
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    treat = (rng.random(n) < 0.5).astype(int)
    y = x1 + 2 * treat + rng.normal(scale=0.1, size=n)
    return pd.DataFrame(
        {"uid": np.arange(n), "treat": treat, "x1": x1, "x2": x2, "y": y}
    )


def _match_roles():
    return {
        "uid": InfoRole(),
        "treat": TreatmentRole(),
        "y": TargetRole(),
        "x1": FeatureRole(),
        "x2": FeatureRole(),
    }


@pytest.mark.parametrize("n_neighbors", [1, 2])
def test_matching_spark_matches_pandas(spark_session, n_neighbors):
    df = _match_frame()
    ds_pd = Dataset(roles=_match_roles(), data=df)
    ds_sp = Dataset(
        roles=_match_roles(),
        data=df,
        backend=BackendsEnum.spark,
        session=spark_session,
    )
    r_pd = Matching(n_neighbors=n_neighbors).execute(ds_pd)
    r_sp = Matching(n_neighbors=n_neighbors).execute(ds_sp)

    a = r_pd.resume.raw_data
    b = r_sp.resume.raw_data
    assert list(a.index) == list(b.index)
    assert list(a.columns) == list(b.columns)
    for col in a.columns:
        if pd.api.types.is_numeric_dtype(a[col]):
            np.testing.assert_allclose(
                a[col].astype(float), b[col].astype(float), rtol=1e-6
            )
        else:
            assert a[col].tolist() == b[col].tolist()

    idx_pd = r_pd.indexes.raw_data.sort_index()
    idx_sp = r_sp.indexes.raw_data.to_pandas().sort_index()
    assert list(idx_sp.index) == list(idx_pd.index)
    np.testing.assert_array_equal(idx_sp.values, idx_pd.values)

    for res in (a, b):
        att = float(res.loc["ATT", "Effect Size"])
        assert att == pytest.approx(2.0, abs=0.1)
