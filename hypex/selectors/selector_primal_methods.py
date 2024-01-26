from typing import List, Optional, Union, Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def pd_fillna_inplace(df: pd.DataFrame, col_list: List[str], is_category: bool = False) -> None:
    """
    Fill NaN values in specified columns of a DataFrame in place.

    This function fills NaN values in the specified columns of the DataFrame
    with the most frequent value for categorical data or the median value
    for numerical data.

    Args:
        df (pd.DataFrame): The DataFrame in which NaN values will be filled.
        col_list (List[str]): A list of column names in which NaN values will be filled.
        is_category (bool, optional): A flag indicating whether the specified columns
                                      are categorical. If True, the most frequent value
                                      ('top') is used for filling NaN values. If False,
                                      the median value ('50%') is used. Defaults to False.

    Examples:
        >>> df = pd.DataFrame({'A': [1, 2, None], 'B': ['x', None, 'y']})
        >>> pd_fillna_inplace(df, ['A'], is_category=False)
        >>> pd_fillna_inplace(df, ['B'], is_category=True)
        >>> df
           A  B
        0  1.0  x
        1  2.0  x
        2  1.5  y
    """
    if col_list:
        dtype = 'str' if is_category else 'float'
        stats = 'top' if is_category else '50%'

        fill_values = df[col_list].astype(dtype).describe(include='all').loc[stats]
        df[col_list] = df[col_list].fillna(fill_values)


def pd_input_preproc(
        df: pd.DataFrame,
        info_col_list: Optional[List[str]] = None,
        target: Union[str, List[str]] = 'target',
        treatment: Optional[Union[str, List[str]]] = None,
        weights_col_list: Optional[List[str]] = None,
        category_col_list: Optional[List[str]] = None
) -> tuple[List[str], List[str], List[str], List[str], pd.DataFrame]:
    """
    Processes the input DataFrame by marking and converting column types and filling NaNs.

    Args:
        df (pd.DataFrame): The input DataFrame to process.
        info_col_list (Optional[List[str]]): List of informational column names.
        target (Union[str, List[str]]): Target column name(s).
        treatment (Optional[Union[str, List[str]]]): Treatment column name(s).
        weights_col_list (Optional[List[str]]): List of weights column names.
        category_col_list (Optional[List[str]]): List of categorical column names.

    Returns:
        tuple: A tuple containing:
            - feature_col_list (List[str]): List of feature column names.
            - target_col_list (List[str]): List of target column names.
            - numeric_col_list (List[str]): List of numeric column names.
            - category_col_list (List[str]): List of categorical column names.
            - df (pd.DataFrame): The processed DataFrame.
    """
    info_col_list = info_col_list or []
    weights_col_list = weights_col_list or []
    category_col_list = category_col_list or []
    treatment_col_list = [treatment] if isinstance(treatment, str) else treatment or []
    target_col_list = [target] if isinstance(target, str) else target

    df = df.copy(deep=True)

    feature_col_list = [
        col for col in df.columns
        if col not in info_col_list + target_col_list + treatment_col_list + weights_col_list
    ]

    numeric_col_list = [col for col in feature_col_list if col not in category_col_list]

    df[category_col_list] = df[category_col_list].astype('str')
    pd_fillna_inplace(df, category_col_list, is_category=True)
    pd_fillna_inplace(df, numeric_col_list, is_category=False)

    df = df[target_col_list + feature_col_list]

    return feature_col_list, target_col_list, numeric_col_list, category_col_list, df


def get_feature_importance_df(gb_model: Any, feature_names: List[str], target_name: str) -> pd.DataFrame:
    """
    Generates a DataFrame with feature importances from a gradient boosting model.

    This function is compatible with various gradient boosting models such as CatBoost
    and LightGBM.

    Args:
        gb_model (Any): The trained gradient boosting model (CatBoostRegressor, LGBMModel, etc.).
        feature_names (List[str]): List of feature names used in the model.
        target_name (str): Name of the target variable.

    Returns:
        pd.DataFrame: A DataFrame containing feature importances and their ranks.
    """
    feature_importance = pd.DataFrame(
        gb_model.feature_importances_,
        index=feature_names,
        columns=[f'"{target_name}" importance']
    ).sort_values(by=f'"{target_name}" importance', ascending=False)

    feature_importance[f'"{target_name}" rank'] = feature_importance[f'"{target_name}" importance'] \
        .rank(ascending=False, method='first').astype('int')

    return feature_importance


def concat_feature_importance_dfs(feature_importance_list: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenates a list of DataFrames containing feature importances and recalculates the ranks.

    This function assumes each DataFrame in the list contains columns of feature importances
    and their ranks. It concatenates these DataFrames side-by-side, recalculates the overall
    ranks, and sorts the features accordingly.

    Args:
        feature_importance_list (List[pd.DataFrame]): A list of DataFrames with feature importances
                                                      and ranks.

    Returns:
        pd.DataFrame: A concatenated DataFrame with updated overall ranks.
    """
    result = pd.concat(feature_importance_list, axis=1)
    rank_col_list = result.iloc[:, 1::2].columns.tolist()
    result.insert(0, 'rank', result[rank_col_list].min(axis=1))
    result = result.sort_values(['rank'] + rank_col_list, ascending=True)
    result['rank'] = result['rank'].rank(ascending=True, method='first').astype('int')

    return result


def pd_lgbm_feature_selector(
        df: pd.DataFrame,
        info_col_list=None,
        target='target',
        treatment_col=None,
        weights_col_list=None,
        category_col_list=None,
        model=None) -> pd.DataFrame:
    (
        feature_col_list,
        target_col_list,
        numeric_col_list,
        category_col_list,
        df
    ) = pd_input_preproc(
        df,
        info_col_list=info_col_list,
        target=target,
        treatment=treatment_col,
        weights_col_list=weights_col_list,
        category_col_list=category_col_list
    )
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        raise ImportError(
            "LightGBM is not installed. Install it by running "
            "'pip install hypex[lgbm]' or 'pip install hypex[all]' to use this feature."
        )

    feature_importance_list = []
    for _target_col in target:
        lgbm_selection_model = LGBMRegressor(
            silent=False,
            verbose=-1,
            num_leaves=31,
            n_estimators=50,
            importance_type='gain',
        )

        lgbm_selection_model.fit(
            df[feature_col_list],
            df[_target_col],
            categorical_feature=category_col_list,
        )

        feature_importance = get_feature_importance_df(
            lgbm_selection_model, feature_col_list, _target_col)
        feature_importance_list.append(feature_importance)

    report_df = concat_feature_importance_dfs(feature_importance_list)
    return report_df


def pd_catboost_feature_selector(
        df: pd.DataFrame,
        info_col_list=None,
        target='target',
        treatment_col=None,
        weights_col_list=None,
        category_col_list=None,
        model=None) -> pd.DataFrame:
    (
        feature_col_list,
        target_col_list,
        numeric_col_list,
        category_col_list,
        df
    ) = pd_input_preproc(
        df,
        info_col_list=info_col_list,
        target=target,
        treatment=treatment_col,
        weights_col_list=weights_col_list,
        category_col_list=category_col_list
    )
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        raise ImportError(
            "CatBoost is not installed. Install it by running "
            "'pip install hypex[cat]' or 'pip install hypex[all]' to use this feature."
        )

    feature_importance_list = []
    for _target_col in target:
        catboost_selection_model = CatBoostRegressor(
            #     iterations=100,
            learning_rate=1e-3,
            depth=7,
        )

        catboost_selection_model.fit(
            df[feature_col_list],
            df[_target_col],
            cat_features=category_col_list,
            silent=True,
        )

        feature_importance = get_feature_importance_df(
            catboost_selection_model, feature_col_list, _target_col)
        feature_importance_list.append(feature_importance)

    report_df = concat_feature_importance_dfs(feature_importance_list)
    return report_df


def pd_ridgecv_feature_selector(
        df: pd.DataFrame,
        info_col_list=None,
        target='target',
        treatment_col=None,
        weights_col_list=None,
        category_col_list=None,
        model=None) -> pd.DataFrame:
    (
        feature_col_list,
        target_col_list,
        numeric_col_list,
        category_col_list,
        df
    ) = pd_input_preproc(
        df,
        info_col_list=info_col_list,
        target=target,
        treatment=treatment_col,
        weights_col_list=weights_col_list,
        category_col_list=category_col_list
    )

    pipe = make_pipeline(
        SimpleImputer(strategy='median'),
        StandardScaler(with_mean=False),
        # RobustScaler(),
        RidgeCV(alphas=np.geomspace(1e5, 1e-5, num=100))
    )

    fitted_pipe = pipe.fit(
        df[feature_col_list].pipe(pd.get_dummies),
        df[target],
    )
    result = pd.DataFrame(
        fitted_pipe[-1].coef_,
        index='"' + pd.Series(target) + '" weight',
        columns=fitted_pipe[:-1].get_feature_names_out(),
    ).T
    result.insert(0, 'rank', result.abs().max(axis=1).rank(ascending=False).astype('int'))
    result = result.sort_values('rank', ascending=True)
    return result
