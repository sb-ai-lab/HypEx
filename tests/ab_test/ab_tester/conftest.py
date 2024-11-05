from hypex import ABTest
import pytest
import numpy as np
import pandas as pd



@pytest.fixture(scope="session",autouse=True)
def data_size():
    return 1_000

@pytest.fixture
def ab_test():
    return ABTest()


@pytest.fixture
def data(data_size):
    # Generate synthetic data for group A
    group_a_data = np.random.normal(loc=10, scale=2, size=data_size)
    # Generate synthetic data for group B
    group_b_data = np.random.normal(loc=12, scale=2, size=data_size)
    group_bp_data = np.random.normal(loc=10, scale=2, size=data_size * 2)
    return pd.DataFrame(
        {
            "group": ["control"] * len(group_a_data) + ["test"] * len(group_b_data),
            "value": list(group_a_data) + list(group_b_data),
            "previous_value": group_bp_data,
        }
    )
@pytest.fixture
def split_ab_result(ab_test, data, group_field):
    return ab_test.split_ab(data, group_field)

@pytest.fixture
def target_field():
    return "value"


@pytest.fixture
def group_field():
    return "group"


@pytest.fixture
def previous_value_field():
    return "previous_value"


@pytest.fixture
def alpha():
    return 0.05
