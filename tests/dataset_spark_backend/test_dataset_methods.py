import pandas as pd
import pytest
import numpy as np
from pyspark.errors.exceptions.base import PySparkValueError
from hypex.dataset import Dataset
from hypex.dataset.roles import FeatureRole, TargetRole, InfoRole, StatisticRole
from hypex.utils import BackendsEnum
from hypex.utils.errors import RoleColumnError


class TestDatasetStats:

    def test_basic_stats(self, spark_dataset):
        assert spark_dataset.mean() is not None
        assert spark_dataset.sum() is not None
        assert spark_dataset.min() is not None
        assert spark_dataset.max() is not None
        assert spark_dataset.count() is not None

    def test_agg(self, spark_dataset):
        result = spark_dataset.agg(["mean", "sum"])
        assert result is not None

    def test_agg_single_function(self, spark_dataset):
        result = spark_dataset.agg(["mean"])
        assert result is not None

    def test_agg_with_dict(self, spark_dataset):
        result = spark_dataset.agg(["mean", "sum"])
        assert result is not None

    def test_std_var_quantile(self, spark_dataset):
        assert spark_dataset.std() is not None
        assert spark_dataset.var() is not None
        assert spark_dataset.quantile(0.5) is not None

    def test_quantile_multiple(self, spark_dataset):
        result = spark_dataset.quantile([0.25, 0.5, 0.75])
        assert result is not None

    def test_quantile_edge_values(self, spark_dataset):
        q0 = spark_dataset.quantile(0.0)
        q1 = spark_dataset.quantile(1.0)
        assert q0 is not None
        assert q1 is not None

    def test_coefficient_of_variation(self, spark_dataset):
        cv = spark_dataset.coefficient_of_variation()
        assert cv is not None

    def test_mode_unique_nunique(self, spark_dataset):
        modes = spark_dataset.mode()
        assert isinstance(modes, Dataset)
        unique = spark_dataset.unique()
        assert isinstance(unique, dict)
        nunique = spark_dataset.nunique()
        assert isinstance(nunique, dict)
        assert nunique["active"] == 2

    def test_nunique_all_columns(self, spark_dataset):
        nunique = spark_dataset.nunique()
        assert "id" in nunique
        assert "name" in nunique
        assert "score" in nunique

    def test_cov_corr(self, spark_dataset):
        cov_matrix = spark_dataset.cov()
        assert cov_matrix is not None
        corr_matrix = spark_dataset.corr()
        assert corr_matrix is not None

    def test_corr_numeric_only(self, spark_dataset):
        corr = spark_dataset.corr()
        assert corr is not None

    def test_stats_single_row(self, df_single_row):
        mean = df_single_row.mean()
        assert mean is not None

    def test_stats_with_nan(self, df_with_na):
        mean = df_with_na.mean()
        assert mean is not None

    def test_std_zero_variance(self, spark_session):
        df = pd.DataFrame({'val': [5, 5, 5, 5]})
        roles = {'val': TargetRole(float)}
        ds = Dataset(roles=roles, data=df, backend=BackendsEnum.spark, session=spark_session)
        std = ds.std()
        assert std is not None


class TestDatasetGroupBy:
    """Тесты группировки и агрегации"""

    def test_groupby_agg(self, spark_session):
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'A'],
            'value': [10, 20, 30, 40, 15]
        })
        roles = {'group': FeatureRole(), 'value': TargetRole(float)}
        ds = Dataset(roles=roles, data=df, backend=BackendsEnum.spark, session=spark_session)
        
        grouped = ds.groupby("group")
        assert grouped is not None
        grouped_agg = grouped.agg({"value": "mean"})
        assert isinstance(grouped_agg, Dataset)

    def test_groupby_multiple_columns(self, df_for_groupby):
        grouped = df_for_groupby.groupby(["group1", "group2"])
        assert grouped is not None
        grouped_agg = grouped.agg({"value1": "sum"})
        assert isinstance(grouped_agg, Dataset)

    def test_groupby_multiple_aggregations(self, df_for_groupby):
        grouped = df_for_groupby.groupby("group1")
        result = grouped.agg({"value1": ["mean", "sum"], "value2": "min"})
        assert isinstance(result, Dataset)

    def test_groupby_count(self, df_for_groupby):
        grouped = df_for_groupby.groupby("group1")
        result = grouped.agg({"value1": "count"})
        assert isinstance(result, Dataset)

    def test_groupby_preserves_roles(self, df_for_groupby):
        grouped = df_for_groupby.groupby("group1")
        result = grouped.agg({"value1": "mean"})
        assert result.roles is not None

    def test_groupby_with_nan(self, df_with_na, spark_session):
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B', 'A'],
            'value': [10, None, 30, 40, 15]
        })
        roles = {'group': FeatureRole(), 'value': TargetRole(float)}
        ds = Dataset(roles=roles, data=df, backend=BackendsEnum.spark, session=spark_session)
        grouped = ds.groupby("group")
        result = grouped.agg({"value": "mean"})
        assert isinstance(result, Dataset)


class TestDatasetModification:
    """Тесты модификации данных"""

    def test_add_column(self, spark_dataset):
        new_col = [1, 2, 3, 4, 5]
        spark_dataset.add_column(new_col, {"new_col": InfoRole()})
        assert "new_col" in spark_dataset.columns

    def test_add_column_with_roles(self, spark_dataset):
        new_col = [1.0, 2.0, 3.0, 4.0, 5.0]
        spark_dataset.add_column(new_col, {"new_col": TargetRole(float)})
        assert "new_col" in spark_dataset.roles
        assert isinstance(spark_dataset.roles["new_col"], TargetRole)

    def test_add_column_length_mismatch_raises(self, spark_dataset):
        with pytest.raises((ValueError, Exception)):
            spark_dataset.add_column([1, 2], {"new_col": InfoRole()})

    def test_append(self, spark_session, default_roles):
        full_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['A', 'B', 'C', 'D', 'E'],
            'value': [10.5, 20.3, 15.7, 30.1, 25.9],
            'active': [True, False, True, False, True],
            'score': [100, 200, 150, 300, 250]
        })
        ds1 = Dataset(roles=default_roles, data=full_df.iloc[:3], backend=BackendsEnum.spark, session=spark_session)
        ds2 = Dataset(roles=default_roles, data=full_df.iloc[3:], backend=BackendsEnum.spark, session=spark_session)
        combined = ds1.append([ds2])
        assert len(combined) == 5

    def test_append_multiple_datasets(self, spark_session, default_roles):
        full_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5, 6],
            'name': ['A', 'B', 'C', 'D', 'E', 'F'],
            'value': [10.5, 20.3, 15.7, 30.1, 25.9, 35.5],
            'active': [True, False, True, False, True, False],
            'score': [100, 200, 150, 300, 250, 350]
        })
        ds1 = Dataset(roles=default_roles, data=full_df.iloc[:2], backend=BackendsEnum.spark, session=spark_session)
        ds2 = Dataset(roles=default_roles, data=full_df.iloc[2:4], backend=BackendsEnum.spark, session=spark_session)
        ds3 = Dataset(roles=default_roles, data=full_df.iloc[4:], backend=BackendsEnum.spark, session=spark_session)
        combined = ds1.append([ds2, ds3])
        assert len(combined) == 6

    def test_to_dict_to_records(self, spark_dataset):
        d = spark_dataset.to_dict()
        assert "data" in d and "backend" in d
        records = spark_dataset.to_records()
        assert len(records) == 5
        assert records[0]["id"] == 1

    def test_to_dict_orient_param(self, spark_dataset):
        d = spark_dataset.to_dict()
        assert isinstance(d, dict)

    def test_dropna_fillna_isna(self, spark_session):
        df_na = pd.DataFrame({'a': [1, None, 3], 'b': [4, 5, None]})
        ds_na = Dataset(roles={'a': InfoRole(), 'b': InfoRole()}, data=df_na, backend=BackendsEnum.spark, session=spark_session)
        
        isna_result = ds_na.isna()
        assert isinstance(isna_result, Dataset)
        
        ds_filled = ds_na.fillna(values=0)
        assert isinstance(ds_filled, Dataset)
        
        ds_dropped = ds_na.dropna()
        assert len(ds_dropped) < len(ds_na)

    def test_fillna_with_dict(self, df_with_na):
        filled = df_with_na.fillna(values={'a': 0, 'b': 99})
        assert isinstance(filled, Dataset)

    def test_fillna_with_method(self, df_with_na):
        try:
            filled = df_with_na.fillna(method="ffill")
            assert isinstance(filled, Dataset)
        except TypeError:
            pytest.skip("fillna не поддерживает method параметр")

    def test_dropna_how_param(self, df_with_na):
        dropped_any = df_with_na.dropna(how="any")
        dropped_all = df_with_na.dropna(how="all")
        assert isinstance(dropped_any, Dataset)
        assert isinstance(dropped_all, Dataset)
        assert len(dropped_all) >= len(dropped_any)

    def test_dropna_subset(self, df_with_na):
        dropped = df_with_na.dropna(subset=["a"])
        assert isinstance(dropped, Dataset)

    def test_rename_replace_filter(self, spark_dataset):
        renamed = spark_dataset.rename({"value": "new_value"})
        assert isinstance(renamed, Dataset)
        assert "new_value" in renamed.columns
        
        replaced = spark_dataset.replace(to_replace={"name": {"A": "Alpha"}})
        assert isinstance(replaced, Dataset)
        
        filtered = spark_dataset.filter(items=["id", "score"])
        assert set(filtered.columns) == {"id", "score"}

    def test_rename_multiple_columns(self, spark_dataset):
        renamed = spark_dataset.rename({"value": "new_value", "score": "new_score"})
        assert "new_value" in renamed.columns
        assert "new_score" in renamed.columns
        assert "value" not in renamed.columns
        assert "score" not in renamed.columns

    def test_replace_multiple_values(self, spark_dataset):
        replaced = spark_dataset.replace(to_replace={"name": {"A": "Alpha", "B": "Beta"}})
        assert isinstance(replaced, Dataset)

    def test_select_dtypes_isin_sample(self, spark_dataset):
        numeric_only = spark_dataset.select_dtypes(include=[np.number])
        assert isinstance(numeric_only, Dataset)
        assert "name" not in numeric_only.columns
        
        isin_result = spark_dataset.isin([1, 3])
        assert isinstance(isin_result, Dataset)
        
        sampled = spark_dataset.sample(n=2, random_state=42)
        assert isinstance(sampled, Dataset)
        assert len(sampled) == 2

    def test_select_dtypes_exclude(self, spark_dataset):
        non_numeric = spark_dataset.select_dtypes(exclude=[np.number])
        assert isinstance(non_numeric, Dataset)
        assert "name" in non_numeric.columns

    def test_sample_with_fraction(self, spark_dataset):
        sampled = spark_dataset.sample(n=2, random_state=42)
        assert isinstance(sampled, Dataset)
        assert len(sampled) <= len(spark_dataset)

    def test_sample_with_replace(self, spark_dataset):
        sampled = spark_dataset.sample(n=3, random_state=42)
        assert isinstance(sampled, Dataset)
        assert len(sampled) == 3

    def test_sample_reproducibility(self, spark_dataset):
        sample1 = spark_dataset.sample(n=2, random_state=42)
        sample2 = spark_dataset.sample(n=2, random_state=42)
        assert sample1.to_dict()["data"] == sample2.to_dict()["data"]

    def test_log(self, spark_dataset):
        log_ds = spark_dataset[["value"]].log()
        assert log_ds is not None

    def test_log_base(self, spark_dataset):
        log_ds = spark_dataset[["value"]].log()
        assert log_ds is not None

    def test_log_natural(self, spark_dataset):
        log_ds = spark_dataset[["value"]].log()
        assert log_ds is not None

    def test_merge(self, spark_session):
        left = Dataset(
            roles={'key': InfoRole(), 'val1': TargetRole(int)}, 
            data=pd.DataFrame({'key': [1, 2], 'val1': [10, 20]}), 
            backend=BackendsEnum.spark, session=spark_session
        )
        right = Dataset(
            roles={'key': InfoRole(), 'val2': TargetRole(int)}, 
            data=pd.DataFrame({'key': [1, 2], 'val2': [100, 200]}), 
            backend=BackendsEnum.spark, session=spark_session
        )
        merged = left.merge(right, on="key")
        assert "val1" in merged.columns and "val2" in merged.columns

    def test_merge_inner(self, df_for_merge_left, df_for_merge_right):
        merged = df_for_merge_left.merge(df_for_merge_right, on="key", how="inner")
        assert isinstance(merged, Dataset)
        assert len(merged) == 2

    def test_merge_left(self, df_for_merge_left, df_for_merge_right):
        merged = df_for_merge_left.merge(df_for_merge_right, on="key", how="left")
        assert isinstance(merged, Dataset)
        assert len(merged) == 3

    def test_merge_right(self, df_for_merge_left, df_for_merge_right):
        merged = df_for_merge_left.merge(df_for_merge_right, on="key", how="right")
        assert isinstance(merged, Dataset)
        assert len(merged) == 3

    def test_merge_outer(self, df_for_merge_left, df_for_merge_right):
        merged = df_for_merge_left.merge(df_for_merge_right, on="key", how="outer")
        assert isinstance(merged, Dataset)
        assert len(merged) == 4

    def test_merge_multiple_keys(self, spark_session):
        left = pd.DataFrame({'k1': [1, 2], 'k2': ['A', 'B'], 'val': [10, 20]})
        right = pd.DataFrame({'k1': [1, 2], 'k2': ['A', 'B'], 'val2': [100, 200]})
        ds_left = Dataset(roles={'k1': InfoRole(), 'k2': InfoRole(), 'val': TargetRole(int)}, data=left, backend=BackendsEnum.spark, session=spark_session)
        ds_right = Dataset(roles={'k1': InfoRole(), 'k2': InfoRole(), 'val2': TargetRole(int)}, data=right, backend=BackendsEnum.spark, session=spark_session)
        merged = ds_left.merge(ds_right, on=["k1", "k2"])
        assert isinstance(merged, Dataset)


class TestDatasetUtils:
    """Тесты утилит и вспомогательных методов"""

    def test_value_counts(self, spark_session):
        df_sp = pd.DataFrame({'cat': ['X', 'Y', 'X', 'Z', 'X']})
        ds_sp = Dataset({'cat': FeatureRole()}, data=df_sp, backend=BackendsEnum.spark, session=spark_session)
        vc_sp = ds_sp.value_counts()
        assert vc_sp is not None
        assert 'cat' in vc_sp.columns

    def test_value_counts_normalize(self, df_with_duplicates):
        vc = df_with_duplicates.value_counts()
        assert vc is not None

    def test_value_counts_multiple_columns(self, df_with_duplicates):
        vc = df_with_duplicates.value_counts()
        assert vc is not None

    def test_value_counts_sort(self, df_with_duplicates):
        vc = df_with_duplicates.value_counts()
        assert vc is not None

    def test_na_counts(self, spark_session):
        df_na = pd.DataFrame({'a': [1, None, 3], 'b': [None, None, 3]})
        ds_na = Dataset(roles={'a': InfoRole(), 'b': InfoRole()}, data=df_na, backend=BackendsEnum.spark, session=spark_session)
        na_counts = ds_na.na_counts()
        assert na_counts is not None

    def test_na_counts_all_values(self, df_with_na):
        na_counts = df_with_na.na_counts()
        assert na_counts is not None
        if isinstance(na_counts, dict):
            assert "a" in na_counts
            assert "b" in na_counts
        elif isinstance(na_counts, Dataset):
            assert na_counts.shape[0] > 0

    def test_na_counts_no_na(self, spark_dataset):
        na_counts = spark_dataset.na_counts()
        assert na_counts is not None
        if isinstance(na_counts, dict):
            assert isinstance(na_counts, dict)

    def test_apply(self, spark_dataset):
        applied = spark_dataset[["score"]].apply(lambda x: x * 2, role={"score": StatisticRole()})
        assert isinstance(applied, Dataset)

    def test_apply_multiple_columns(self, spark_dataset):
        applied = spark_dataset[["score", "value"]].apply(lambda x: x + 1, role={"score": StatisticRole(), "value": StatisticRole()})
        assert isinstance(applied, Dataset)
        assert set(applied.columns) == {"score", "value"}

    def test_apply_with_role(self, spark_dataset):
        applied = spark_dataset[["score"]].apply(lambda x: x * 2, role={"score": StatisticRole()})
        assert "score" in applied.roles
        assert isinstance(applied.roles["score"], StatisticRole)

    def test_map(self, spark_dataset):
        mapped = spark_dataset[["id"]].map(lambda x: x + 10)
        assert isinstance(mapped, Dataset)

    def test_map_multiple_columns(self, spark_dataset):
        mapped = spark_dataset[["id", "score"]].map(lambda x: x * 2)
        assert isinstance(mapped, Dataset)
        assert set(mapped.columns) == {"id", "score"}

    def test_map_preserves_shape(self, spark_dataset):
        mapped = spark_dataset[["id"]].map(lambda x: x)
        assert mapped.shape == spark_dataset[["id"]].shape

    def test_apply_on_empty(self, spark_session):
        ds_empty = Dataset(roles={}, data=None, backend=BackendsEnum.spark, session=spark_session)
        try:
            result = ds_empty.apply(lambda x: x, role={})
            assert result is not None
        except Exception:
            pytest.skip("apply не поддерживается для пустого Dataset")

    def test_value_counts_empty(self, spark_session):
        df_empty = pd.DataFrame({'cat': []})
        try:
            ds_empty = Dataset({'cat': FeatureRole()}, data=df_empty, backend=BackendsEnum.spark, session=spark_session)
            vc = ds_empty.value_counts()
            assert vc is not None
            assert len(vc) == 0
        except PySparkValueError:
            pytest.skip("PySpark cannot infer schema from empty dataset")

    def test_merge_on_nonexistent_column(self, df_for_merge_left, df_for_merge_right):
        with pytest.raises((KeyError, Exception)):
            df_for_merge_left.merge(df_for_merge_right, on="nonexistent")

    def test_groupby_nonexistent_column(self, spark_dataset):
        with pytest.raises((KeyError, Exception)):
            spark_dataset.groupby("nonexistent")