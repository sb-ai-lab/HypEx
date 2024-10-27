import pytest
from hypex.utils.tutorial_data_creation import create_test_data


@pytest.fixture
def data():
    return create_test_data(rs=52)


@pytest.fixture
def iterations():
    return 20


@pytest.fixture
def info_col():
    return "user_id"