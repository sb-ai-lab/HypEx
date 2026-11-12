import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pyspark")

from hypex import Matching
from hypex.dataset import (
    AdditionalMatchingRole,
    Dataset,
    ExperimentData,
    FeatureRole,
    IndexRole,
    InfoRole,
    TargetRole,
    TreatmentRole,
)
from hypex.ui.matching import MatchingOutput
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
                a[col].astype(float), b[col].astype(float), rtol=1e-6, atol=1e-3
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


def test_matching_index_role_restores_business_ids_on_both_backends(spark_session):
    rng = np.random.default_rng(4)
    n = 20
    df = pd.DataFrame(
        {
            "uid": [f"user-{i}" for i in range(n)],
            "treat": [0] * (n // 2) + [1] * (n // 2),
            "x1": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )
    roles = {
        "uid": IndexRole(str),
        "treat": TreatmentRole(),
        "x1": FeatureRole(),
        "y": TargetRole(),
    }
    ds_pd = Dataset(roles=roles, data=df)
    ds_sp = Dataset(
        roles=roles,
        data=df,
        backend=BackendsEnum.spark,
        session=spark_session,
    )
    original_pd = ds_pd.raw_data.copy()
    original_sp_index = ds_sp.index.to_pandas().tolist()
    matching = Matching(
        distance="l2",
        faiss_mode="base",
        bias_estimation=False,
        quality_tests=[],
        n_neighbors=2,
        compute_indexes=True,
        extract_full_data=True,
    )

    out_pd = matching.execute(ds_pd)
    out_sp = Matching(
        distance="l2",
        faiss_mode="base",
        bias_estimation=False,
        quality_tests=[],
        n_neighbors=2,
        compute_indexes=True,
        extract_full_data=True,
    ).execute(ds_sp)

    idx_pd = out_pd.indexes.raw_data.sort_index()
    idx_sp = out_sp.indexes.raw_data.to_pandas().sort_index()
    full_pd = out_pd.full_data.raw_data.sort_index()
    full_sp = out_sp.full_data.raw_data.to_pandas().sort_index()

    pd.testing.assert_frame_equal(idx_pd, idx_sp, check_dtype=False)
    assert set(idx_pd.index) == set(df["uid"])
    assert idx_pd.index.name == "uid"
    assert idx_pd.columns.tolist() == ["indexes_0", "indexes_1"]
    for i, column in enumerate(idx_pd):
        assert idx_pd[column].isin(df["uid"]).all()
        pd.testing.assert_series_equal(
            full_pd[f"uid_matched_{i}"], idx_pd[column], check_names=False
        )
        pd.testing.assert_series_equal(
            full_sp[f"uid_matched_{i}"], idx_sp[column], check_names=False
        )
    assert full_pd.index.name == full_sp.index.name == "uid"
    pd.testing.assert_frame_equal(ds_pd.raw_data, original_pd)
    assert ds_sp.index.to_pandas().tolist() == original_sp_index


def test_matching_index_role_indexes_without_full_data(spark_session):
    rng = np.random.default_rng(5)
    n = 12
    df = pd.DataFrame(
        {
            "uid": [f"user-{i}" for i in range(n)],
            "treat": [0, 1] * (n // 2),
            "x1": rng.normal(size=n),
            "y": rng.normal(size=n),
        }
    )
    roles = {
        "uid": IndexRole(str),
        "treat": TreatmentRole(),
        "x1": FeatureRole(),
        "y": TargetRole(),
    }
    results = []
    for backend in (BackendsEnum.pandas, BackendsEnum.spark):
        ds = Dataset(
            roles=roles,
            data=df,
            backend=backend,
            session=spark_session if backend == BackendsEnum.spark else None,
        )
        output = Matching(
            distance="l2",
            faiss_mode="base",
            bias_estimation=False,
            quality_tests=[],
            compute_indexes=True,
            extract_full_data=False,
        ).execute(ds)
        assert output.full_data.is_empty()
        indexes = output.indexes.raw_data
        results.append(
            indexes.sort_index()
            if backend == BackendsEnum.pandas
            else indexes.to_pandas().sort_index()
        )

    pd.testing.assert_frame_equal(results[0], results[1], check_dtype=False)
    assert results[0].index.name == "uid"
    assert set(results[0].index) == set(df["uid"])
    assert results[0]["indexes_0"].isin(df["uid"]).all()


def test_spark_index_role_unmatched_becomes_null(spark_session):
    ds = Dataset(
        roles={"uid": IndexRole(str)},
        data=pd.DataFrame({"uid": ["a", "b", "c"]}),
        backend=BackendsEnum.spark,
        session=spark_session,
    ).reset_index(drop=True)
    output = MatchingOutput()
    output.indexes = Dataset(
        roles={"indexes_0": AdditionalMatchingRole()},
        data=pd.DataFrame({"indexes_0": [1, -1, 0]}),
        backend=BackendsEnum.spark,
        session=spark_session,
    )
    output.full_data = Dataset.create_empty(
        backend=BackendsEnum.spark, session=spark_session
    )

    output._restore_index_role(ExperimentData(ds))

    result = output.indexes.raw_data.to_pandas().sort_index()
    assert result.index.tolist() == ["a", "b", "c"]
    assert result["indexes_0"].iloc[[0, 2]].tolist() == ["b", "a"]
    assert pd.isna(result["indexes_0"].iloc[1])
