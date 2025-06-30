from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

ROOT = Path("").absolute().parents[0]
sys.path.append(str(ROOT))


def set_nans(
    data: pd.DataFrame,
    na_step: Sequence[int] | int | None = None,
    nan_cols: Sequence[str] | str | None = None,
) -> pd.DataFrame:
    """Fills specific columns in a DataFrame with NaN values at given intervals.

    Args:
        data: The input DataFrame.
        na_step: Step interval(s) for inserting NaNs.
        nan_cols: Column(s) to insert NaNs.

    Returns:
        pd.DataFrame: Modified DataFrame with NaNs.
    """
    # Convert na_step and nan_cols to lists
    if na_step is None:
        na_step_list: list[int] = [10]
    elif isinstance(na_step, int):
        na_step_list = [na_step]
    else:
        na_step_list = list(na_step)

    if nan_cols is None:
        nan_cols_list: list[str] = list(data.columns)
    elif isinstance(nan_cols, str):
        nan_cols_list = [nan_cols]
    else:
        nan_cols_list = list(nan_cols)

    if not na_step_list or not nan_cols_list:
        raise ValueError("na_step and nan_cols must not be empty.")

    if len(na_step_list) < len(nan_cols_list):
        na_step_list += [na_step_list[-1]] * (len(nan_cols_list) - len(na_step_list))
    elif len(na_step_list) > len(nan_cols_list):
        na_step_list = na_step_list[: len(nan_cols_list)]

    for col, step in zip(nan_cols_list, na_step_list):
        if col in data.columns:
            data.loc[step::step, col] = None

    return data


def create_test_data(
    num_users: int = 10000,
    na_step: Sequence[int] | int | None = None,
    nan_cols: Sequence[str] | str | None = None,
    file_name: str | None = None,
    exact_ATT: int = 100,
    rs=None,
):
    """Creates data for tutorial.

    Args:
        num_users: num of strings
        na_step:
            num or list of nums of period to make NaN (step of range)
            If list - iterates accordingly order of columns
        nan_cols:
            name of one or several columns to fill with NaN
            If list - iterates accordingly order of na_step
        file_name: name of file to save; doesn't save file if None
        exact_ATT: an accurate synthetic effect

    Returns:
        data: dataframe with
    """
    if rs is not None:
        np.random.seed(rs)

    if (nan_cols is not None) and isinstance(nan_cols, str):
        nan_cols = [nan_cols]
    # Simulating dataset with known effect size
    num_months = 12

    # signup_months == 0 means customer did not sign up
    signup_months = np.random.choice(
        np.arange(1, num_months), num_users
    ) * np.random.randint(0, 2, size=num_users)

    data = pd.DataFrame(
        {
            "user_id": np.repeat(np.arange(num_users), num_months),
            "signup_month": np.repeat(
                signup_months, num_months
            ),  # signup month == 0 means customer did not sign up
            "month": np.tile(
                np.arange(1, num_months + 1), num_users
            ),  # months are from 1 to 12
            "spend": np.random.poisson(500, num_users * num_months),
        }
    )

    # A customer is in the treatment group if and only if they signed up
    data["treat"] = data["signup_month"] > 0

    # Simulating an effect of month (monotonically decreasing--customers buy less later in the year)
    data["spend"] = data["spend"] - data["month"] * 10

    # Simulating a simple treatment effect of 100
    after_signup = (data["signup_month"] < data["month"]) & (data["treat"])
    data.loc[after_signup, "spend"] = data[after_signup]["spend"] + exact_ATT

    # Setting the signup month (for ease of analysis)
    i = 3
    data = (
        data.groupby(["user_id", "signup_month", "treat"])
        .apply(
            lambda x: pd.Series(
                {
                    "pre_spends": x.loc[x.month < i, "spend"].mean(),
                    "post_spends": x.loc[x.month > i, "spend"].mean(),
                }
            )
        )
        .reset_index()
    )

    # Additional category features
    gender_i = np.random.choice(a=[0, 1], size=data.user_id.nunique())
    gender = [["M", "F"][i] for i in gender_i]

    age = np.random.choice(a=range(18, 70), size=data.user_id.nunique())

    industry_i = np.random.choice(a=range(1, 3), size=data.user_id.nunique())
    industry_names = ["Finance", "E-commerce", "Logistics"]
    industry = [industry_names[i] for i in industry_i]

    data["age"] = age
    data["gender"] = gender
    data["industry"] = industry
    data["industry"] = data["industry"].astype("str")
    data["treat"] = data["treat"].astype(int)

    # input nans in data if needed
    data = set_nans(data, na_step, nan_cols)

    if file_name is not None:
        data.to_csv(ROOT / f"{file_name}.csv", index=False)

    return data


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic sigmoid ufunc for ndarrays.

    The sigmoid function, also known as the logistic sigmoid function,
    is defined as sigmoid(x) = 1/(1+exp(-x)).
    It is the inverse of the logit function.

    Args:
        x: Input array.

    Returns:
        Sigmoid function of x.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_division(x, dependent_division=True) -> np.ndarray:
    """
    Division method via sigmoid function.

    Args:
        x:
            Input array
        dependent_division:
            If True - returns the binary vector dependent on the X.
            If False - returns the binary vector independent on the X.

    Returns:
        Binary array coordinated with variable x.
    """
    if dependent_division:
        division = np.random.binomial(1, sigmoid((x - x.mean()) / x.std()))
    else:
        division = np.random.binomial(1, 0.5, size=len(x))
    return division


def gen_special_medicine_df(
    data_size=100, *, dependent_division=True, random_state=None
) -> pd.DataFrame:
    """Synthetic dataframe generator.

    Realises dependent/independent group splitting.

    Args:
        data_size:
            Length of output Dataframe
        dependent_division:
            If True - the returned Dataframe contains a division into
            groups dependent on the features.
            If False - the returned Dataframe contains a division into
            groups independent on the features.
        random_state:
            If specified - defines numpy random seed. Defaults to None.

    Returns:
        Synthetic dataframe also containing fictional information about patient treatment.
    """
    if random_state is not None:
        np.random.seed(random_state)

    disease_degree = np.random.choice(
        [1, 2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.1, 0.1], size=data_size
    ).astype(int)

    age = np.random.normal(40, scale=8, size=data_size).astype(int)

    age_effect = (age ^ 2 - 400) / 1000

    experimental_treatment = sigmoid_division(disease_degree, dependent_division)

    residual_lifetime = np.random.exponential(
        17 - 2.5 * disease_degree + 1 * experimental_treatment - age_effect
    )

    df = pd.DataFrame(
        dict(
            age=age,
            disease_degree=disease_degree,
            experimental_treatment=experimental_treatment,
            residual_lifetime=residual_lifetime,
        )
    )
    return df


def gen_oracle_df(
    data_size=8, *, dependent_division=True, factual_only=False, random_state=None
) -> pd.DataFrame:
    """Synthetic dataframe generator.

    Realises factual and counterfactual outcomes.

    Args:
        data_size:
            Length of output Dataframe
        dependent_division:
            If True - the returned Dataframe contains a division into
            groups dependent on the features.
            If False - the returned Dataframe contains a division into
            groups independent on the features.
        factual_only:
            Defines the display of counterfactual states.
        random_state:
            If specified - defines numpy random seed. Defaults to None.

    Returns:
        Synthetic dataframe also containing fictional information about
        factual and counterfactual states of the target.
    """
    if random_state is not None:
        np.random.seed(random_state)

    treatment = np.random.binomial(1, 0.5, size=data_size)

    if dependent_division:
        target_feature = np.random.binomial(1, 0.3 + 0.4 * treatment)
    else:
        target_feature = np.random.binomial(1, 0.5, size=data_size)

    target_untreated = (
        np.random.uniform(low=300, high=800, size=data_size).round(-2).astype(int)
    )

    target_treated = target_untreated + 50 + target_feature * 100

    if factual_only:
        target_untreated = np.where(1 - treatment, target_untreated, np.nan)
        target_treated = np.where(treatment, target_treated, np.nan)

    y_factual = np.where(treatment, target_treated, target_untreated).astype("int")

    treatment_effect = target_treated - target_untreated

    df = pd.DataFrame(
        dict(
            X=target_feature,
            Target_untreated=target_untreated,
            Target_treated=target_treated,
            Treatment=treatment,
            Target=y_factual,
            TE=treatment_effect,
        )
    )
    return df


def gen_control_variates_df(
    data_size=1000, *, dependent_division=True, random_state=None
) -> pd.DataFrame:
    """Synthetic dataframe generator.

    Realises 0-variation outcome mixed with linear x dependency.

    Args:
        data_size:
            Length of output Dataframe
        dependent_division:
            If True - the returned Dataframe contains a division into
            groups dependent on the features.
            If False - the returned Dataframe contains a division into
            groups independent on the features.
        random_state:
            If specified - defines numpy random seed. Defaults to None.

    Returns:
        Synthetic dataframe also containing fictional information about
        features and lagged features.
    """
    if random_state is not None:
        np.random.seed(random_state)

    mean_x_feature = np.random.uniform(0, 5, size=data_size)

    x_feature_lag_1 = np.random.normal(mean_x_feature, 2)
    x_feature = np.random.normal(mean_x_feature, 2)

    treatment = sigmoid_division(x_feature_lag_1, dependent_division)

    target_lag_1 = 200 + x_feature_lag_1 * 100
    target_factual = 200 + x_feature * 100 + treatment * 10

    df = pd.DataFrame(
        dict(
            X_lag_1=x_feature_lag_1,
            Target_lag_1=target_lag_1,
            X=x_feature,
            Treatment=treatment,
            Target=target_factual,
        )
    )
    return df
