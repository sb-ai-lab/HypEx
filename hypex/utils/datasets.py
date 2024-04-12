import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic sigmoid ufunc for ndarrays.

    The sigmoid function, also known as the logistic sigmoid function,
    is defined as sigmoid(x) = $\fraq{1}{(1+\exp{-x})}$.
    It is the inverse of the logit function.

    Args:
        x:
            Input array.

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

    Returns:
        Synthetic dataframe also containing fictional information about patient treatment.
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
        factual_only=False,
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
        target_feature = np.random.binomial(
            1,
            0.3 + 0.4 * treatment
        )
    else:
        target_feature = np.random.binomial(1, 0.5, size=data_size)

    target_untreated = np.random.uniform(
        low=300,
        high=800,
        size=data_size
    ).round(-2).astype(int)

    target_treated = target_untreated + 50 + target_feature * 100

    if factual_only:
        target_untreated = np.where(1 - treatment, target_untreated, np.nan)
        target_treated = np.where(treatment, target_treated, np.nan)

    y_factual = np.where(treatment, target_treated, target_untreated).astype('int')

    treatment_effect = target_treated - target_untreated

    df = pd.DataFrame(dict(
        X=target_feature,
        Target_untreated=target_untreated,
        Target_treated=target_treated,
        Treatment=treatment,
        Target=y_factual,
        TE=treatment_effect,
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

    df = pd.DataFrame(dict(
        X_lag_1=x_feature_lag_1,
        Target_lag_1=target_lag_1,
        X=x_feature,
        Treatment=treatment,
        Target=target_factual,
    ))
    return df
