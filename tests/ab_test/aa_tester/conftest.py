import pytest
from hypex.utils.tutorial_data_creation import create_test_data
from hypex import AATest
import pandas as pd


@pytest.fixture(scope='session')
def data():
    return create_test_data(rs=52)


@pytest.fixture(scope='session')
def iterations():
    return 20


@pytest.fixture(scope='session')
def info_col():
    return "user_id"

@pytest.fixture(scope='session')
def group_col():
    return "industry"

@pytest.fixture(scope='session')
def quant_field():
    return "gender"


@pytest.fixture(scope='session')
def simple_AATest(info_col):
    return AATest(target_fields=["pre_spends", "post_spends"], info_cols=info_col)

@pytest.fixture(scope='session')
def grouped_AATest(group_col):
    return AATest(target_fields=["pre_spends", "post_spends"], info_cols=group_col)

@pytest.fixture(scope='session')
def quanted_grouped_AATest(group_col,quant_field):
    return AATest(target_fields=["pre_spends", "post_spends"], info_cols=group_col, group_cols=group_col, quant_field=quant_field)


@pytest.fixture(scope='session')
def simple_AATest_results(simple_AATest, data, iterations)->tuple[pd.DataFrame, dict]:
    return simple_AATest.calc_uniform_tests(data=data, iterations=iterations, pbar=False)

@pytest.fixture(scope='session')
def grouped_AATest_results(grouped_AATest, data, iterations)->tuple[pd.DataFrame, dict]:
    return grouped_AATest.calc_uniform_tests(data=data, iterations=iterations, pbar=False)

@pytest.fixture(scope='session')
def quanted_grouped_AATest_results(quanted_grouped_AATest, data, iterations)->tuple[pd.DataFrame, dict]:
    return quanted_grouped_AATest.calc_uniform_tests(data=data, iterations=iterations, pbar=False)