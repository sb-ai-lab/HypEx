from hypex import ABTest
import pytest
import numpy as np
import pandas as pd


DATA_SIZE = 1000


@pytest.fixture
def ab_test():
    return ABTest()


@pytest.fixture
def data():
    # Generate synthetic data for group A
    group_a_data = np.random.normal(loc=10, scale=2, size=DATA_SIZE)
    # Generate synthetic data for group B
    group_b_data = np.random.normal(loc=12, scale=2, size=DATA_SIZE)
    group_bp_data = np.random.normal(loc=10, scale=2, size=DATA_SIZE * 2)
    return pd.DataFrame(
        {
            "group": ["control"] * len(group_a_data) + ["test"] * len(group_b_data),
            "value": list(group_a_data) + list(group_b_data),
            "previous_value": group_bp_data,
        }
    )


@pytest.fixture
def target_field():
    return "value"


@pytest.fixture
def group_field():
    return "group"


@pytest.fixture
def previous_value():
    return "previous_value"


@pytest.fixture
def alpha():
    return 0.05
