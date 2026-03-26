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
    
@pytest.fixture
def df_with_na(spark_session):
    """Dataset с пропусками (NaN)"""
    df = pd.DataFrame({
        'a': [1.0, None, 3.0, None, 5.0],
        'b': [None, 2.0, None, 4.0, 5.0],
        'c': [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    roles = {'a': TargetRole(float), 'b': TargetRole(float), 'c': TargetRole(float)}
    return Dataset(roles=roles, data=df, backend=BackendsEnum.spark, session=spark_session)

@pytest.fixture
def df_with_duplicates(spark_session):
    """Dataset с дубликатами"""
    df = pd.DataFrame({
        'cat': ['X', 'Y', 'X', 'Z', 'X', 'Y'],
        'val': [1, 2, 1, 3, 1, 2]
    })
    roles = {'cat': FeatureRole(), 'val': TargetRole(int)}
    return Dataset(roles=roles, data=df, backend=BackendsEnum.spark, session=spark_session)

@pytest.fixture
def df_for_groupby(spark_session):
    """Dataset для тестов группировки"""
    df = pd.DataFrame({
        'group1': ['A', 'A', 'B', 'B', 'A', 'C'],
        'group2': ['X', 'Y', 'X', 'Y', 'X', 'X'],
        'value1': [10, 20, 30, 40, 15, 50],
        'value2': [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]
    })
    roles = {
        'group1': FeatureRole(), 
        'group2': FeatureRole(),
        'value1': TargetRole(int),
        'value2': TargetRole(float)
    }
    return Dataset(roles=roles, data=df, backend=BackendsEnum.spark, session=spark_session)

@pytest.fixture
def df_for_merge_left(spark_session):
    """Левая таблица для merge"""
    df = pd.DataFrame({'key': [1, 2, 3], 'val1': [10, 20, 30]})
    roles = {'key': InfoRole(), 'val1': TargetRole(int)}
    return Dataset(roles=roles, data=df, backend=BackendsEnum.spark, session=spark_session)

@pytest.fixture
def df_for_merge_right(spark_session):
    """Правая таблица для merge"""
    df = pd.DataFrame({'key': [2, 3, 4], 'val2': [100, 200, 300]})
    roles = {'key': InfoRole(), 'val2': TargetRole(int)}
    return Dataset(roles=roles, data=df, backend=BackendsEnum.spark, session=spark_session)

@pytest.fixture
def df_single_row(spark_session):
    """Dataset с одной строкой (граничный случай)"""
    df = pd.DataFrame({'id': [1], 'value': [10.5]})
    roles = {'id': InfoRole(), 'value': TargetRole(float)}
    return Dataset(roles=roles, data=df, backend=BackendsEnum.spark, session=spark_session)

@pytest.fixture
def df_all_nan(spark_session):
    """Dataset где все значения NaN"""
    df = pd.DataFrame({'a': [None, None], 'b': [None, None]})
    roles = {'a': TargetRole(float), 'b': TargetRole(float)}
    return Dataset(roles=roles, data=df, backend=BackendsEnum.spark, session=spark_session)