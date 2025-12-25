from sklearn.linear_model import Lasso, LinearRegression, Ridge

try:
    from catboost import CatBoostRegressor

    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

CUPAC_MODELS = {
    "linear": {
        "pandasdataset": LinearRegression(),
        "polars": None,
    },
    "ridge": {
        "pandasdataset": Ridge(),
        "polars": None,
    },
    "lasso": {
        "pandasdataset": Lasso(),
        "polars": None,
    },
}

if CATBOOST_AVAILABLE:
    CUPAC_MODELS["catboost"] = {
        "pandasdataset": CatBoostRegressor(verbose=0),
        "polars": None,
    }
