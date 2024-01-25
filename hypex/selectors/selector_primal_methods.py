import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV


def pd_fillna_inplace(df, col_list, is_category=False):
    if col_list:
        category_fill_values = (
            df.loc[:, col_list]
                .astype('str' if is_category else 'double')
                .describe(include='all')
                .loc['top' if is_category else '50%']
        )
        df.loc[:, col_list] = df.loc[:, col_list].fillna(category_fill_values)


def pd_input_preproc(
        df: pd.DataFrame,
        info_col_list=None,
        target='target',
        treatment=None,
        weights_col_list=None,
        category_col_list=None) -> tuple[
                                            list[str],
                                            list[str],
                                            list[str],
                                            list[str],
                                            pd.DataFrame
                                        ]:
    """
    Marks and convert column types.
    Fills NANs.
    Args:
        df:
        info_col_list:
        target:
        treatment:
        weights_col_list:
        category_col_list:

    Returns: marked column sets and copy of processed dataframe

    """
    if info_col_list is None:
        info_col_list = []
    if weights_col_list is None:
        weights_col_list = []
    if category_col_list is None:
        category_col_list = []

    if treatment is None:
        treatment_col_list = []
    elif isinstance(treatment, str):
        treatment_col_list = [treatment]
    else:
        treatment_col_list = treatment

    if isinstance(target, str):
        target_col_list = [target]
    else:
        target_col_list = target

    df = df.copy(deep=True)

    feature_col_list = [
        col
        for col in df.columns
        if col not in (
            *info_col_list,
            *target_col_list,
            *treatment_col_list,
            *weights_col_list,
        )
    ]

    numeric_col_list = [
        col
        for col in feature_col_list
        if col not in category_col_list
    ]

    df.loc[:, category_col_list] = df.loc[:, category_col_list].astype('str')
    pd_fillna_inplace(df, category_col_list, is_category=True)
    pd_fillna_inplace(df, numeric_col_list, is_category=False)

    df = df.loc[:, target_col_list + feature_col_list]
    return (
        feature_col_list,
        target_col_list,
        numeric_col_list,
        category_col_list,
        df
    )


def get_feature_importance_df(gb_model, feature_names, target_name):
    feature_importance = pd.DataFrame(
        gb_model.feature_importances_,
        index=feature_names,
        columns=[f'"{target_name}" importance']
    ).sort_index()

    feature_importance[f'"{target_name}" rank'] = feature_importance[f'"{target_name}" importance'] \
        .rank(ascending=False, method='first').astype('int')
    return feature_importance


def concat_feature_importance_dfs(feature_importance_list):
    result = pd.concat(feature_importance_list, axis=1)
    rank_col_list = result.iloc[:, 1::2].columns.tolist()
    result.insert(0, 'rank', result.loc[:, rank_col_list].min(axis=1))
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
