import pytest
import pandas as pd
from pyspark.sql import SparkSession
from hypex.dataset import Dataset
from hypex.dataset.roles import TargetRole, FeatureRole, InfoRole, StatisticRole
from hypex.utils import BackendsEnum

@pytest.fixture(scope="session")
def spark_session():
    spark = (SparkSession.builder
        .appName("Hypex-Pytest")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()

@pytest.fixture
def sample_pd_df():
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['A', 'B', 'C', 'D', 'E'],
        'value': [10.5, 20.3, 15.7, 30.1, 25.9],
        'active': [True, False, True, False, True],
        'score': [100, 200, 150, 300, 250]
    })

@pytest.fixture
def default_roles():
    return {
        'id': InfoRole(),
        'name': FeatureRole(),
        'value': TargetRole(float),
        'active': FeatureRole(bool),
        'score': TargetRole(int)
    }

@pytest.fixture
def spark_dataset(sample_pd_df, default_roles, spark_session):
    return Dataset(
        roles=default_roles, 
        data=sample_pd_df, 
        backend=BackendsEnum.spark, 
        session=spark_session
    )