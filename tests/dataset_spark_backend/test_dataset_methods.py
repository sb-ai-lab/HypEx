import pandas as pd
import pytest
import numpy as np
from hypex.dataset import Dataset
from hypex.dataset.roles import FeatureRole, TargetRole, InfoRole, StatisticRole
from hypex.utils import BackendsEnum


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

    def test_std_var_quantile(self, spark_dataset):
        assert spark_dataset.std() is not None
        assert spark_dataset.var() is not None
        assert spark_dataset.quantile(0.5) is not None

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

    def test_cov_corr(self, spark_dataset):
        cov_matrix = spark_dataset.cov()
        assert cov_matrix is not None
        corr_matrix = spark_dataset.corr()
        assert corr_matrix is not None


class TestDatasetGroupBy:
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


class TestDatasetModification:
    def test_add_column(self, spark_dataset):
        new_col = [1, 2, 3, 4, 5]
        spark_dataset.add_column(new_col, {"new_col": InfoRole()})
        assert "new_col" in spark_dataset.columns

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

    def test_to_dict_to_records(self, spark_dataset):
        d = spark_dataset.to_dict()
        assert "data" in d and "backend" in d
        records = spark_dataset.to_records()
        assert len(records) == 5
        assert records[0]["id"] == 1

    def test_dropna_fillna_isna(self, spark_session):
        df_na = pd.DataFrame({'a': [1, None, 3], 'b': [4, 5, None]})
        ds_na = Dataset(roles={'a': InfoRole(), 'b': InfoRole()}, data=df_na, backend=BackendsEnum.spark, session=spark_session)
        
        isna_result = ds_na.isna()
        assert isinstance(isna_result, Dataset)
        
        ds_filled = ds_na.fillna(values=0)
        assert isinstance(ds_filled, Dataset)
        
        ds_dropped = ds_na.dropna()
        assert len(ds_dropped) < len(ds_na)

    def test_rename_replace_filter(self, spark_dataset):
        renamed = spark_dataset.rename({"value": "new_value"})
        assert isinstance(renamed, Dataset)
        assert "new_value" in renamed.columns
        
        replaced = spark_dataset.replace(to_replace={"name": {"A": "Alpha"}})
        assert isinstance(replaced, Dataset)
        
        filtered = spark_dataset.filter(items=["id", "score"])
        assert set(filtered.columns) == {"id", "score"}

    def test_select_dtypes_isin_sample(self, spark_dataset):
        numeric_only = spark_dataset.select_dtypes(include=[np.number])
        assert isinstance(numeric_only, Dataset)
        assert "name" not in numeric_only.columns
        
        isin_result = spark_dataset.isin([1, 3])
        assert isinstance(isin_result, Dataset)
        
        sampled = spark_dataset.sample(n=2, random_state=42)
        assert isinstance(sampled, Dataset)
        assert len(sampled) == 2

    def test_log(self, spark_dataset):
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


class TestDatasetUtils:
    def test_value_counts(self, spark_session):
        df_sp = pd.DataFrame({'cat': ['X', 'Y', 'X', 'Z', 'X']})
        ds_sp = Dataset({'cat': FeatureRole()}, data=df_sp, backend=BackendsEnum.spark, session=spark_session)
        vc_sp = ds_sp.value_counts()
        assert vc_sp is not None
        assert 'cat' in vc_sp.columns

    def test_na_counts(self, spark_session):
        df_na = pd.DataFrame({'a': [1, None, 3], 'b': [None, None, 3]})
        ds_na = Dataset(roles={'a': InfoRole(), 'b': InfoRole()}, data=df_na, backend=BackendsEnum.spark, session=spark_session)
        na_counts = ds_na.na_counts()
        assert na_counts is not None

    def test_apply(self, spark_dataset):
        applied = spark_dataset[["score"]].apply(lambda x: x * 2, role={"score": StatisticRole()})
        assert isinstance(applied, Dataset)

    def test_map(self, spark_dataset):
        print(type(spark_dataset[["id"]]))
        mapped = spark_dataset[["id"]].map(lambda x: x + 10)
        assert isinstance(mapped, Dataset)