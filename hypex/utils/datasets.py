import pandas as pd
import numpy as np


def sigmoid(x):
    """Logistic sigmoid ufunc for ndarrays.

    The sigmoid function, also known as the logistic sigmoid function,
    is defined as sigmoid(x) = 1/(1+exp(-x)).
    It is the inverse of the logit function."""
    return 1 / (1 + np.exp(-x))


def gen_special_medicine_df(
        data_size=100,
        *,
        dependent_division=True,
        random_state=None
) -> pd.DataFrame:
    """Synthetic dataframe generator.
    Realises dependent/independent group splitting.
    """
    if not random_state is None:
        np.random.seed(random_state)

    disease_degree = np.random.choice(
        [1, 2, 3, 4, 5],
        p=[0.3, 0.3, 0.2, 0.1, 0.1],
        size=data_size
    ).astype(int)

    if dependent_division:
        experimental_treatment = np.random.binomial(
            1,
            sigmoid((disease_degree - disease_degree.mean()) / disease_degree.std())
        )
    else:
        experimental_treatment = np.random.binomial(1, 0.5, size=data_size)

    residual_lifetime = np.random.exponential(13 - 2.5 * disease_degree + 1 * experimental_treatment)

    df = pd.DataFrame(dict(
        disease_degree=disease_degree,
        experimental_treatment=experimental_treatment,
        residual_lifetime=residual_lifetime,
    ))
    return df


def gen_oracle_df(
        data_size=8,
        *,
        dependent_division=True,
        treatment_effect_size=150,
        factual_Y_only=False,
        random_state=None
) -> pd.DataFrame:
    """Synthetic dataframe generator.
    Realises factual and contrfactual outcomes.
    """
    if not random_state is None:
        np.random.seed(random_state)

    T = np.random.binomial(1, 0.5, size=data_size)

    if dependent_division:
        X = np.random.binomial(
            1,
            0.3 + 0.4 * T
        )
    else:
        X = np.random.binomial(1, 0.5, size=data_size)

    Y0 = np.random.uniform(
        low=300,
        high=800,
        size=data_size
    ).round(-2).astype(int)

    Y1 = Y0 + 50 + X * 100

    if factual_Y_only:
        Y0 = np.where(1 - T, Y0, np.nan)
        Y1 = np.where(T, Y1, np.nan)

    Y = np.where(T, Y1, Y0).astype('int')

    TE = Y1 - Y0

    df = pd.DataFrame(dict(
        X=X,
        Y0=Y0,
        Y1=Y1,
        T=T,
        Y=Y,
        TE=TE,
    ))
    return df


def gen_control_variates_df(
        data_size=1000,
        *,
        treatment_effect_size=150,
        dependent_division=True,
        random_state=None
) -> pd.DataFrame:
    """Synthetic dataframe generator.
    Realises 0-variation outcome mixed with linear X dependency.
    """
    if not random_state is None:
        np.random.seed(random_state)

    X_means = np.random.uniform(0, 5, size=data_size)

    X_lag_1 = np.random.normal(X_means, 2)
    X = np.random.normal(X_means, 2)

    if dependent_division:
        T = np.random.binomial(
            1,
            sigmoid((X_lag_1 - X_lag_1.mean()) / X_lag_1.std())
        )
    else:
        T = np.random.binomial(1, 0.5, size=data_size)

    Y_lag_1 = 200 + X_lag_1 * 100
    Y = 200 + X * 100 + T * 10

    df = pd.DataFrame(dict(
        X_lag_1=X_lag_1,
        Y_lag_1=Y_lag_1,
        X=X,
        T=T,
        Y=Y,
    ))
    return df
