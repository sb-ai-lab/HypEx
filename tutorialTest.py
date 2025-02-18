import random
import pandas as pd
import numpy as np
import pytest
import pandas.testing as pdt
from hypex.dataset import Dataset, InfoRole, TreatmentRole, TargetRole, StratificationRole, FeatureRole
from hypex import AATest, ABTest, Matching

file_path = 'data_output.xlsx'

@pytest.fixture
def aa_data():
    return Dataset(
        roles={
            "user_id": InfoRole(int),
            "treat": TreatmentRole(int),
            "pre_spends": TargetRole(),
            "post_spends": TargetRole(),
            "gender": StratificationRole(str),
        },
        data="data.csv"
    )

@pytest.fixture
def ab_data():
    random.seed(7)
    data = Dataset(
        roles={
            "user_id": InfoRole(int),
            "treat": TreatmentRole(),
            "pre_spends": TargetRole(),
            "post_spends": TargetRole(),
            "gender": TargetRole()
        },
        data="data.csv",
    )
    data["treat"] = [random.choice([0, 1, 2]) for _ in range(len(data))]
    return data

@pytest.fixture
def matching_data():
    data = Dataset(
        roles={
            "user_id": InfoRole(int),
            "treat": TreatmentRole(int), 
            "post_spends": TargetRole(float)
        },
        data="data.csv",
        default_role=FeatureRole()
    )
    data = data.bfill()
    return data

def test_aatest(aa_data):
    mapping = {
        'aa-casual': AATest(n_iterations=10),
        'aa-rs': AATest(random_states=[56, 72, 2, 43]),
        'aa-strat': AATest(random_states=[56, 72, 2, 43], stratification=True),
        'aa-sample': AATest(n_iterations=10, sample_size=0.3)
    }

    required_attrs = ['resume', 'aa_score', 'best_split', 'best_split_statistic', 'experiments']
    
    for test_name in mapping.keys():
        res = mapping[test_name].execute(aa_data)
        for attr in required_attrs:
            assert hasattr(res, attr), f"Результат должен содержать атрибут '{attr}'"

        for attr in required_attrs:
            expected_data = pd.read_excel(file_path, sheet_name=f'{test_name}.res.{attr}.data', index_col=0)
            actual_data = getattr(res, attr).data

            expected_data = expected_data.fillna(0).apply(pd.to_numeric, errors='ignore')
            actual_data = actual_data.fillna(0).apply(pd.to_numeric, errors='ignore')

            pdt.assert_frame_equal(expected_data, actual_data, check_dtype=False)

def test_abtest(ab_data):
    mapping = {
        'ab-casual': ABTest(),
        'ab-additional': ABTest(additional_tests=['t-test', 'u-test', 'chi2-test']),
        'ab-n': ABTest(multitest_method="bonferroni")
    }

    required_attrs = ['resume', 'sizes', 'multitest']
    
    for test_name in mapping.keys():
        res = mapping[test_name].execute(ab_data)
        for attr in required_attrs:
            assert hasattr(res, attr), f"Результат должен содержать атрибут '{attr}'"

        for attr in required_attrs:
            expected_data = pd.read_excel(file_path, sheet_name=f'{test_name}.result.{attr}.data', index_col=0)
            actual_data = getattr(res, attr).data

            expected_data = expected_data.fillna(0).apply(pd.to_numeric, errors='ignore')
            actual_data = actual_data.fillna(0).apply(pd.to_numeric, errors='ignore')

            pdt.assert_frame_equal(expected_data, actual_data, check_dtype=False)

def test_matchingtest(matching_data):
    mapping = {
        'matching': Matching(),
        'matching-atc': Matching(metric="atc"),
        'matching-att': Matching(metric='att'),
        'matching-l2': Matching(distance="l2", metric='att')
    }

    required_attrs = ['resume', 'indexes']
    
    for test_name in mapping.keys():
        res = mapping[test_name].execute(matching_data)
        for attr in required_attrs:
            assert hasattr(res, attr), f"Результат должен содержать атрибут '{attr}'"

        for attr in required_attrs:
            expected_data = pd.read_excel(file_path, sheet_name=f'{test_name}.result.{attr}.data', index_col=0)
            actual_data = getattr(res, attr).data

            expected_data = expected_data.fillna(0).apply(pd.to_numeric, errors='ignore')
            actual_data = actual_data.fillna(0).apply(pd.to_numeric, errors='ignore')

            pdt.assert_frame_equal(expected_data, actual_data, check_dtype=False)