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


@pytest.fixture
def aa_data(data_csv):
    return [
        Dataset(
            roles={
                "user_id": InfoRole(int),
                "treat": TreatmentRole(int),
                "pre_spends": TargetRole(),
                "post_spends": TargetRole(),
                "gender": StratificationRole(str),
            },
            data=data_csv,
        ),
        Dataset(
            roles={
                "user_id": InfoRole(int),
                "treat": TreatmentRole(int),
                "pre_spends": TargetRole(),
                "post_spends": TargetRole(),
                "gender": TargetRole(str),
            },
            data=data_csv,
        ),
    ]


@pytest.fixture
def ab_data(data_csv):
    random.seed(7)
    data = Dataset(
        roles={
            "user_id": InfoRole(int),
            "treat": TreatmentRole(),
            "pre_spends": TargetRole(),
            "post_spends": TargetRole(),
            "gender": TargetRole(),
        },
        data=data_csv,
    )
    data["treat"] = [random.choice([0, 1, 2]) for _ in range(len(data))]
    return data


@pytest.fixture
def matching_data(data_csv):
    data = Dataset(
        roles={
            "user_id": InfoRole(int),
            "treat": TreatmentRole(int),
            "post_spends": TargetRole(float),
        },
        data=data_csv,
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
                "TTest aa score": ["OK", "OK"],
                "TTest best split": ["OK", "OK"],
                "KSTest aa score": ["OK", "OK"],
                "KSTest best split": ["OK", "OK"],
                "result": ["OK", "OK"],
            }
        ),
        "aa-rs": pd.DataFrame(
            {
                "TTest aa score": ["OK", "OK"],
                "TTest best split": ["OK", "OK"],
                "KSTest aa score": ["OK", "OK"],
                "KSTest best split": ["OK", "OK"],
                "result": ["OK", "OK"],
            }
        ),
        "aa-strat": pd.DataFrame(
            {
                "TTest aa score": ["OK", "OK"],
                "TTest best split": ["OK", "OK"],
                "KSTest aa score": ["OK", "OK"],
                "KSTest best split": ["OK", "OK"],
                "result": ["OK", "OK"],
            }
        ),
        "aa-sample": pd.DataFrame(
            {
                "TTest aa score": ["OK", "OK"],
                "TTest best split": ["OK", "OK"],
                "KSTest aa score": ["OK", "OK"],
                "KSTest best split": ["OK", "OK"],
                "result": ["OK", "OK"],
            }
        ),
        "aa-cat_target": pd.DataFrame(
            {
                "TTest aa score": [np.nan, "OK", "OK"],
                "TTest best split": [np.nan, "OK", "OK"],
                "KSTest aa score": [np.nan, "OK", "OK"],
                "KSTest best split": [np.nan, "OK", "OK"],
                "Chi2Test aa score": ["OK", np.nan, np.nan],
                "Chi2Test best split": ["OK", np.nan, np.nan],
                "result": ["OK", "OK", "OK"],
            }
        ),
        "aa-equal_var": pd.DataFrame(
            {
                "TTest aa score": ["OK", "OK"],
                "TTest best split": ["OK", "OK"],
                "KSTest aa score": ["OK", "OK"],
                "KSTest best split": ["OK", "OK"],
                "result": ["OK", "OK"],
            }
        ),
        "aa-n": pd.DataFrame(
            {
                "TTest aa score": ["OK", "OK", "OK", "OK"],
                "TTest best split": ["OK", "OK", "OK", "OK"],
                "KSTest aa score": ["OK", "OK", "OK", "OK"],
                "KSTest best split": ["OK", "OK", "OK", "OK"],
                "result": ["OK", "OK", "OK", "OK"],
            }
        ),
    }

    for test_name in mapping.keys():
        print(test_name)
        if test_name == "aa-cat_target":
            res = mapping[test_name].execute(aa_data[1])
        else:
            res = mapping[test_name].execute(aa_data[0])
        actual_data = res.summary.data.iloc[:, 2:-4]
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
                "Chi2Test pass": {
                    0: 0,
                    1: 0,
                    2: 0,
                    3: 0,
                    4: "NOT OK",
                    5: "NOT OK",
                },
            }
        ),
        "ab-n": pd.DataFrame(
            {"TTest pass": {0: "NOT OK", 1: "NOT OK", 2: "NOT OK", 3: "NOT OK"}}
        ),
    }

    for test_name in mapping.keys():
        res = mapping[test_name].execute(ab_data)
        summary = res.summary.data.fillna(0).apply(pd.to_numeric, errors="ignore")
        # Select the verdict columns by name: the number of metric columns in
        # front of them (means, differences, confidence interval) can change.
        actual_data = summary[[c for c in summary.columns if c.endswith(" pass")]]
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
        actual_data = res.summary.data
        assert actual_data.index.isin(["ATT", "ATC", "ATE"]).all()
        assert all(
            actual_data.iloc[:, :-1].dtypes.apply(
                lambda x: pd.api.types.is_numeric_dtype(x)
            )
        ), "Есть нечисловые колонки!"
