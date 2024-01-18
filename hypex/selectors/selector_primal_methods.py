import numpy as np
import pandas as pd


def pd_lgbm_feature_selector(
        df:pd.DataFrame,
        info_col_list=None,
        target='target',
        treatment_col=None,
        weights_col_list=None,
        category_col_list=None,
        model=None) -> pd.DataFrame:

    if category_col_list is None:
        category_col_list = []
    if weights_col_list is None:
        weights_col_list = []
    if treatment_col is None:
        treatment_col = []
    if info_col_list is None:
        info_col_list = []
    if model is None:
        model = 'lgbm_regressor'
    if isinstance(target, str):
        target = [target]

    feature_col_list = [
        col
        for col in df.columns
        if col not in (
            *info_col_list,
            *target,
            treatment_col,
            *weights_col_list,
        )
    ]

    numeric_col_list = [
        col
        for col in feature_col_list
        if col not in category_col_list
    ]

    if category_col_list:
        category_fill_values = df.loc[:, category_col_list].astype('str').describe(include='all').loc['top']
        df.loc[:, category_col_list] = df.loc[:, category_col_list].fillna(category_fill_values).astype('str')

    if numeric_col_list:
        numeric_fill_values = df.loc[:, numeric_col_list].describe(include='all').loc['50%']
        df.loc[:, numeric_col_list] = df.loc[:, numeric_col_list].fillna(numeric_fill_values)

    feature_importance_list = []
    for _target_col in target:
        from catboost import CatBoostRegressor
        feature_selection_model = CatBoostRegressor(
            #     iterations=100,
            learning_rate=1e-3,
            depth=7,
        )

        feature_selection_model.fit(
            df[feature_col_list],
            df[_target_col],
            cat_features=category_col_list,
            silent=True,
        )

        feature_importance = pd.DataFrame(
            feature_selection_model.feature_importances_,
            index=feature_col_list,
            columns=[f'"{_target_col}" importance']
        ).sort_index()

        feature_importance[f'"{_target_col}" rank'] = np.argsort(feature_importance[f'"{_target_col}" importance'])
        feature_importance_list.append(feature_importance)

    result = pd.concat(feature_importance_list, axis=1)
    rank_col_list = result.iloc[:, 1::2].columns.tolist()
    result.insert(0, 'total min rank', result.loc[:, rank_col_list].min(axis=1))
    result = result.sort_values(['total min rank'] + rank_col_list, ascending=True)
    return result


def pd_ridgecv_feature_selector(
        df:pd.DataFrame,
        info_col_list=None,
        target='target',
        treatment_col=None,
        weights_col_list=None,
        category_col_list=None,
        model=None) -> pd.DataFrame:

    if category_col_list is None:
        category_col_list = []
    if weights_col_list is None:
        weights_col_list = []
    if treatment_col is None:
        treatment_col = []
    if info_col_list is None:
        info_col_list = []
    if model is None:
        model = 'lgbm_regressor'
    if isinstance(target, str):
        target = [target]

    feature_col_list = [
        col
        for col in df.columns
        if col not in (
            *info_col_list,
            *target,
            treatment_col,
            *weights_col_list,
        )
    ]

    numeric_col_list = [
        col
        for col in feature_col_list
        if col not in category_col_list
    ]

    if category_col_list:
        category_fill_values = df.loc[:, category_col_list].astype('str').describe(include='all').loc['top']
        df.loc[:, category_col_list] = df.loc[:, category_col_list].fillna(category_fill_values).astype('str')

    if numeric_col_list:
        numeric_fill_values = df.loc[:, numeric_col_list].describe(include='all').loc['50%']
        df.loc[:, numeric_col_list] = df.loc[:, numeric_col_list].fillna(numeric_fill_values)

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import RidgeCV

    pipe = make_pipeline(
        # OneHotEncoder(sparse=False),
        # SimpleImputer(strategy='median'),
        StandardScaler(with_mean=False),
        # RobustScaler(),
        RidgeCV(alphas=100000 * 0.5**np.arange(1000))
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
    result.insert(0, 'total max weight', result.abs().max(axis=1))
    result = result.sort_values('total max weight', ascending=False)
    return result