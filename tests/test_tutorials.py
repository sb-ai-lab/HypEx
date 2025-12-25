# starts with HYPEX-dir: PYTHONPATH=$(pwd) pytest
import random

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from hypex import AATest, ABTest, Matching
from hypex.dataset import (
    Dataset,
    FeatureRole,
    InfoRole,
    StratificationRole,
    TargetRole,
    TreatmentRole,
)

# from hypex.utils import create_test_data
#
# df = create_test_data()


@pytest.fixture
def aa_data():
    return [
        Dataset(
            roles={
                "user_id": InfoRole(int),
                "treat": TreatmentRole(int),
                "pre_spends": TargetRole(),
                "post_spends": TargetRole(),
                "gender": StratificationRole(str),
            },
            data="tests/data.csv",
        ),
        Dataset(
            roles={
                "user_id": InfoRole(int),
                "treat": TreatmentRole(int),
                "pre_spends": TargetRole(),
                "post_spends": TargetRole(),
                "gender": TargetRole(str),
            },
            data="tests/data.csv",
        ),
    ]


@pytest.fixture
def ab_data():
    random.seed(7)
    data = Dataset(
        roles={
            "user_id": InfoRole(int),
            "treat": TreatmentRole(),
            "pre_spends": TargetRole(),
            "post_spends": TargetRole(),
            "gender": TargetRole(),
        },
        data="tests/data.csv",
    )
    data["treat"] = [random.choice([0, 1, 2]) for _ in range(len(data))]
    return data


@pytest.fixture
def matching_data():
    data = Dataset(
        roles={
            "user_id": InfoRole(int),
            "treat": TreatmentRole(int),
            "post_spends": TargetRole(float),
        },
        data="tests/data.csv",
        default_role=FeatureRole(),
    )
    data = data.fillna(method="bfill")
    return data


def test_aatest(aa_data):
    mapping = {
        "aa-casual": AATest(n_iterations=10),
        "aa-rs": AATest(random_states=[56, 72, 2, 43]),
        "aa-strat": AATest(stratification=True, random_states=[56, 72, 2, 43]),
        "aa-sample": AATest(n_iterations=10, sample_size=0.3),
        "aa-cat_target": AATest(n_iterations=10),
        "aa-equal_var": AATest(n_iterations=10, t_test_equal_var=False),
        "aa-n": AATest(n_iterations=10, groups_sizes=[0.5, 0.2, 0.3]),
    }

    mapping_resume = {
        "aa-casual": pd.DataFrame(
            {
                "TTest aa test": {0: "OK", 1: "OK"},
                "KSTest aa test": {0: "NOT OK", 1: "OK"},
                "TTest best split": {0: "OK", 1: "OK"},
                "KSTest best split": {0: "OK", 1: "OK"},
                "result": {0: "OK", 1: "OK"},
            }
        ),
        "aa-rs": pd.DataFrame(
            {
                "TTest aa test": {0: "OK", 1: "OK"},
                "KSTest aa test": {0: "NOT OK", 1: "OK"},
                "TTest best split": {0: "OK", 1: "OK"},
                "KSTest best split": {0: "OK", 1: "OK"},
                "result": {0: "OK", 1: "OK"},
            }
        ),
        "aa-strat": pd.DataFrame(
            {
                "TTest aa test": {0: "OK", 1: "NOT OK"},
                "KSTest aa test": {0: "OK", 1: "NOT OK"},
                "TTest best split": {0: "OK", 1: "OK"},
                "KSTest best split": {0: "OK", 1: "OK"},
                "result": {0: "OK", 1: "NOT OK"},
            }
        ),
        "aa-sample": pd.DataFrame(
            {
                "TTest aa test": {0: "OK", 1: "OK"},
                "KSTest aa test": {0: "OK", 1: "OK"},
                "TTest best split": {0: "NOT OK", 1: "NOT OK"},
                "KSTest best split": {0: "OK", 1: "OK"},
                "result": {0: "OK", 1: "OK"},
            }
        ),
        "aa-cat_target": pd.DataFrame(
            {
                "TTest aa test": ["OK", "OK", np.nan],
                "KSTest aa test": ["NOT OK", "OK", np.nan],
                "Chi2Test aa test": [np.nan, np.nan, "OK"],
                "TTest best split": ["OK", "OK", np.nan],
                "KSTest best split": ["OK", "OK", np.nan],
                "Chi2Test best split": [np.nan, np.nan, "OK"],
                "result": ["OK", "OK", "OK"],
            }
        ),
        "aa-equal_var": pd.DataFrame(
            {
                "TTest aa test": {0: "OK", 1: "OK"},
                "KSTest aa test": {0: "NOT OK", 1: "OK"},
                "TTest best split": {0: "OK", 1: "OK"},
                "KSTest best split": {0: "OK", 1: "OK"},
                "result": {0: "OK", 1: "OK"},
            }
        ),
        "aa-n": pd.DataFrame(
            {
                "TTest aa test": {0: "OK", 1: "OK", 2: "OK", 3: "OK"},
                "KSTest aa test": {0: "OK", 1: "OK", 2: "OK", 3: "OK"},
                "TTest best split": {0: "OK", 1: "OK", 2: "OK", 3: "OK"},
                "KSTest best split": {0: "OK", 1: "OK", 2: "OK", 3: "OK"},
                "result": {0: "OK", 1: "OK", 2: "OK", 3: "OK"},
            }
        ),
    }

    for test_name in mapping.keys():
        print(test_name)
        if test_name == "aa-cat_target":
            res = mapping[test_name].execute(aa_data[1])
        else:
            res = mapping[test_name].execute(aa_data[0])
        actual_data = res.resume.data.iloc[:, 2:-4]
        expected_data = mapping_resume[test_name]
        pdt.assert_frame_equal(expected_data, actual_data, check_dtype=False)


def test_abtest(ab_data):
    mapping = {
        "ab-casual": ABTest(),
        "ab-additional": ABTest(additional_tests=["t-test", "u-test", "chi2-test"]),
        "ab-n": ABTest(multitest_method="bonferroni"),
    }

    mapping_resume = {
        "ab-casual": pd.DataFrame(
            {"TTest pass": {0: "NOT OK", 1: "NOT OK", 2: "NOT OK", 3: "NOT OK"}}
        ),
        "ab-additional": pd.DataFrame(
            {
                "TTest pass": {
                    0: "NOT OK",
                    1: "NOT OK",
                    2: "NOT OK",
                    3: "NOT OK",
                    4: 0,
                    5: 0,
                },
                "UTest pass": {
                    0: "NOT OK",
                    1: "NOT OK",
                    2: "NOT OK",
                    3: "NOT OK",
                    4: 0,
                    5: 0,
                },
                "Chi2Test pass": {0: 0, 1: 0, 2: 0, 3: 0, 4: "NOT OK", 5: "NOT OK"},
            }
        ),
        "ab-n": pd.DataFrame(
            {"TTest pass": {0: "NOT OK", 1: "NOT OK", 2: "NOT OK", 3: "NOT OK"}}
        ),
    }

    for test_name in mapping.keys():
        res = mapping[test_name].execute(ab_data)
        actual_data = (
            res.resume.data.fillna(0)
            .apply(pd.to_numeric, errors="ignore")
            .iloc[:, 6::2]
        )
        expected_data = mapping_resume[test_name]
        pdt.assert_frame_equal(expected_data, actual_data, check_dtype=False)


def test_matchingtest(matching_data):
    mapping = {
        "matching": Matching(),
        "matching-l2": Matching(distance="l2"),
        "matching-faiss-auto": Matching(distance="l2", faiss_mode="auto"),
        "matching-faiss_base": Matching(distance="mahalanobis", faiss_mode="base"),
        "matching-n-neighbors": Matching(n_neighbors=2),
    }

    for test_name in mapping.keys():
        res = mapping[test_name].execute(matching_data)
        actual_data = res.resume.data
        assert actual_data.index.isin(["ATT", "ATC", "ATE"]).all()
        assert all(
            actual_data.iloc[:, :-1].dtypes.apply(
                lambda x: pd.api.types.is_numeric_dtype(x)
            )
        ), "Есть нечисловые колонки!"
