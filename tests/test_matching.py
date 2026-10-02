# pytest -s -v tests/test_matching.py
from itertools import product

import numpy as np
import pandas as pd
import pytest
from causalinference import CausalModel
from causalinference.utils import tools

from hypex import Matching
from hypex.dataset import (
    Dataset,
    FeatureRole,
    GroupingRole,
    InfoRole,
    TargetRole,
    TreatmentRole,
)


@pytest.fixture
def matching_data(data_csv):
    df = pd.read_csv(data_csv)
    df = df.bfill().fillna(0)
    df["gender"] = df["gender"].astype("category").cat.codes
    df["industry"] = df["industry"].astype("category").cat.codes
    df["treat"] = df["treat"]

    roles = {
        "user_id": InfoRole(int),
        "treat": TreatmentRole(int),
        "post_spends": TargetRole(float),
        "gender": FeatureRole(str),
        "pre_spends": FeatureRole(float),
        "industry": FeatureRole(str),
        "signup_month": FeatureRole(int),
        "age": FeatureRole(float),
    }

    return Dataset(roles=roles, data=df)


@pytest.fixture
def matching_data_with_group(data_csv):
    df = pd.read_csv(data_csv)
    df = df.bfill().fillna(0)
    df["gender"] = df["gender"].astype("category").cat.codes
    df["industry"] = df["industry"].astype("category").cat.codes
    df["treat"] = df["treat"].clip(0, 1)

    roles = {
        "user_id": InfoRole(int),
        "treat": TreatmentRole(int),
        "post_spends": TargetRole(float),
        "gender": GroupingRole(int),
        "pre_spends": FeatureRole(float),
        "industry": FeatureRole(int),
        "signup_month": FeatureRole(int),
        "age": FeatureRole(float),
    }

    return Dataset(roles=roles, data=df)


feature_subsets = [
    ["pre_spends"],
    ["age", "gender"],
    ["pre_spends", "gender", "industry"],
]

distances = ["mahalanobis"]
k_values = [1, 3, 5]

scenarios = [
    "default",
    "group_match_gender",
    "group_match_industry",
    "custom_weights_1",
    "custom_weights_2",
]

param_combinations = list(product(feature_subsets, distances, k_values, scenarios))

custom_weights_dict = {
    "custom_weights_1": {
        "pre_spends": 0.5,
        "gender": 0.3,
        "industry": 0.2,
    },
    "custom_weights_2": {
        "pre_spends": 0.2,
        "gender": 0.4,
        "industry": 0.4,
    },
}


def get_causalinference_pvalue(data_df, feature_subset, k, effect="ate"):
    Y = data_df["post_spends"].values
    D = data_df["treat"].values
    X = data_df[feature_subset].astype(float).values
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    model = CausalModel(Y, D, X)
    model.est_via_matching(matches=k, bias_adj=True)

    try:
        est = model.estimates.get("matching", model.estimates)
        coef = est[effect.lower()]
        se = est[f"{effect.lower()}_se"]
    except Exception:
        return np.nan

    if se == 0 or np.isnan(se):
        return 0.0 if abs(coef) > 1e-6 else 1.0

    entries = tools.gen_reg_entries(effect.upper(), float(coef), float(se))
    return float(entries[4])  # p-value


def calculate_relative_ratio(p1, p2, eps=1e-9):
    return (p1 + eps) / (p2 + eps)


@pytest.mark.parametrize("feature_subset,distance,k,scenario", param_combinations)
def test_matching_scenario(
    matching_data, matching_data_with_group, feature_subset, distance, k, scenario
):
    print(f"Features: {feature_subset}")

    if "group_match" in scenario:
        data_subset = matching_data_with_group
        group_match_flag = True
        scenario_weights = None

        if "gender" in scenario:
            data_subset.roles["gender"] = GroupingRole(int)
        elif "industry" in scenario:
            data_subset.roles["industry"] = GroupingRole(int)
            # The fixture groups by gender; keep a single grouping column.
            data_subset.roles["gender"] = FeatureRole(int)
    else:
        current_roles = {
            "user_id": InfoRole(),
            "treat": TreatmentRole(),
            "post_spends": TargetRole(),
        }
        for feature in feature_subset:
            current_roles[feature] = FeatureRole()
        data_subset = Dataset(roles=current_roles, data=matching_data.data)
        group_match_flag = False
        scenario_weights = custom_weights_dict.get(scenario, None)

    matcher_hypex = Matching(
        distance=distance,
        n_neighbors=k,
        weights=scenario_weights,
        group_match=group_match_flag,
        quality_tests=["t-test", "ks-test", "chi2-test"],
    )

    result_hypex = matcher_hypex.execute(data_subset)
    summary = result_hypex.summary.data

    if group_match_flag:
        # Group matching reports one block of columns per group ("<group> P-value").
        group_col = "gender" if "gender" in scenario else "industry"
        features = [
            col
            for col, role in data_subset.roles.items()
            if isinstance(role, FeatureRole) and col != group_col
        ]
        frame = matching_data_with_group.data
        for group, group_df in frame.groupby(group_col):
            pval_hypex = summary.loc["ATE", f"{group} P-value"]
            pval_causal = get_causalinference_pvalue(group_df, features, k)
            rel_ratio = calculate_relative_ratio(pval_hypex, pval_causal)
            assert 0.95 <= rel_ratio <= 1.05, (
                f"p-value relative ratio out of range: {rel_ratio:.2f} "
                f"for group {group_col}={group}, scenario={scenario}, k={k}"
            )
        return

    pval_hypex = summary.loc["ATE", "P-value"]
    pval_causal = get_causalinference_pvalue(matching_data.data, feature_subset, k)
    rel_ratio = calculate_relative_ratio(pval_hypex, pval_causal)

    assert 0.95 <= rel_ratio <= 1.05, (
        f"p-value relative ratio out of range: {rel_ratio:.2f} "
        f"for features {feature_subset}, scenario={scenario}, k={k}"
    )
