import os

import pandas as pd
import pytest

from hypex.dataset import Dataset
from hypex.dataset.roles import InfoRole
from hypex.utils import BackendsEnum

class TestDatasetInit:
    def test_init_with_pandas(self, spark_dataset):
        assert len(spark_dataset) == 5
        assert spark_dataset.shape == (5, 5)

    def test_init_with_dict(self, spark_session):
        dict_data = {"data": {"x": [1, 2], "y": [3, 4]}}
        ds = Dataset(
            roles={"x": InfoRole(), "y": InfoRole()}, 
            data=dict_data, 
            backend=BackendsEnum.spark, 
            session=spark_session
        )
        assert ds.shape == (2, 2)

    def test_init_empty(self, spark_session):
        ds = Dataset(roles={}, data=None, backend=BackendsEnum.spark, session=spark_session)
        assert ds.is_empty()

    def test_read_csv(self, tmp_path, spark_session):
        csv_path = tmp_path / "test.csv"
        pd.DataFrame({'id': [1]}).to_csv(csv_path, index=False)
        ds = Dataset(roles={}, data=str(csv_path), backend=BackendsEnum.spark, session=spark_session)
        assert 'id' in ds.columns

    def test_read_parquet(self, tmp_path, spark_session):
        pq_path = tmp_path / "test.parquet"
        pd.DataFrame({'id': [1, 2, 3, 4, 5]}).to_parquet(pq_path)
        ds = Dataset(roles={}, data=str(pq_path), backend=BackendsEnum.spark, session=spark_session)
        assert ds.shape[0] == 5