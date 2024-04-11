import pandas as pd
import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic sigmoid ufunc for ndarrays.

    The sigmoid function, also known as the logistic sigmoid function,
    is defined as sigmoid(x) = 1/(1+exp(-x)).
    It is the inverse of the logit function.
    Args:
        x:
            Input array
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_division(x, dependent_division=True) -> np.ndarray:
    """
    Returns binary array.
    Args:
        x:
            Input array
        dependent_division:
            If True - returns the binary vector dependent on the X.
            If False - returns the binary vector independent on the X.
    """
    if dependent_division:
        division = np.random.binomial(
            1,
            sigmoid((x - x.mean()) / x.std())
        )
    else:
        division = np.random.binomial(1, 0.5, size=len(x))
    return division


def gen_special_medicine_df(
        data_size=100,
        *,
        dependent_division=True,
        random_state=None
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
    """
    if random_state is not None:
        np.random.seed(random_state)

    disease_degree = np.random.choice(
        [1, 2, 3, 4, 5],
        p=[0.3, 0.3, 0.2, 0.1, 0.1],
        size=data_size
    ).astype(int)

    experimental_treatment = sigmoid_division(disease_degree, dependent_division)

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
        factual_y_only=False,
        random_state=None
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
        factual_y_only:
            Defines the display of counterfactual states.
        random_state:
            If specified - defines numpy random seed. Defaults to None.
    """
    if random_state is not None:
        np.random.seed(random_state)

    t = np.random.binomial(1, 0.5, size=data_size)

    if dependent_division:
        x = np.random.binomial(
            1,
            0.3 + 0.4 * t
        )
    else:
        x = np.random.binomial(1, 0.5, size=data_size)

    y0 = np.random.uniform(
        low=300,
        high=800,
        size=data_size
    ).round(-2).astype(int)

    y1 = y0 + 50 + x * 100

    if factual_y_only:
        y0 = np.where(1 - t, y0, np.nan)
        y1 = np.where(t, y1, np.nan)

    y = np.where(t, y1, y0).astype('int')

    te = y1 - y0

    df = pd.DataFrame(dict(
        X=x,
        Y0=y0,
        Y1=y1,
        T=t,
        Y=y,
        TE=te,
    ))
    return df


def gen_control_variates_df(
        data_size=1000,
        *,
        dependent_division=True,
        random_state=None
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
    """
    if random_state is not None:
        np.random.seed(random_state)

    x_means = np.random.uniform(0, 5, size=data_size)

    x_lag_1 = np.random.normal(x_means, 2)
    x = np.random.normal(x_means, 2)

    t = sigmoid_division(x_lag_1, dependent_division)

    y_lag_1 = 200 + x_lag_1 * 100
    y = 200 + x * 100 + t * 10

    df = pd.DataFrame(dict(
        X_lag_1=x_lag_1,
        Y_lag_1=y_lag_1,
        X=x,
        T=t,
        Y=y,
    ))
    return df
