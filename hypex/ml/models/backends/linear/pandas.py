from __future__ import annotations

from inspect import signature
from io import BytesIO
from typing import ClassVar, Literal

import joblib
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from .....dataset import ABCRole, Dataset
from .....dataset.backends import PandasDataset
from .....utils.registry import backend_factory
from ...linear import LinearModel, _l1_ratio


@backend_factory.register(LinearModel, PandasDataset)
class PandasLinearModel:
    PARAMS_CONVERSION: ClassVar[dict[str, str]] = {
        "fit_intercept": "fit_intercept",
        "reg_param": "alpha",
        "elastic_net_param": "l1_ratio",
        "solver": "solver",
        "max_iter": "max_iter",
        "tol": "tol",
    }

    def __init__(
        self,
        *,
        fit_intercept: bool = True,
        reg_param: float = 0.0,
        elastic_net_param: float = 0.0,
        reg_type: Literal["l1", "l2"] = "l1",
        solver: str | None = None,
        max_iter: int = 1000,
        tol: float = 0.0001,
    ) -> None:
        ratio = _l1_ratio(elastic_net_param, reg_type)
        estimator = (
            LinearRegression
            if reg_param == 0
            else {0.0: Ridge, 1.0: Lasso}.get(ratio, ElasticNet)
        )
        supported = signature(estimator).parameters
        if "solver" not in supported and solver not in (None, "auto"):
            raise ValueError(f"{estimator.__name__} does not accept solver={solver!r}.")
        params = {
            "fit_intercept": fit_intercept,
            "reg_param": reg_param,
            "elastic_net_param": ratio,
            "solver": "cholesky" if solver == "normal" else solver or "auto",
            "max_iter": max_iter,
            "tol": tol,
        }
        self._model = estimator(
            **{
                native_name: params[name]
                for name, native_name in self.PARAMS_CONVERSION.items()
                if native_name in supported
            }
        )

    def fit(self, data: Dataset, features: list[str], target: str) -> None:
        frame = data.data
        self._model.fit(frame[features], frame[target])

    def predict(
        self, X: Dataset, features: list[str], output: str, role: ABCRole
    ) -> Dataset:
        X_frame = X.data[features]
        result = X_frame.iloc[:, :0].copy()
        result[output] = self._model.predict(X_frame) if len(X_frame) else []
        return LinearModel.result_to_dataset(result, roles={output: role}, small=False)

    def dumps(self) -> bytes:
        with BytesIO() as buffer:
            joblib.dump(self._model, buffer)
            return buffer.getvalue()

    @classmethod
    def loads(cls, data: bytes) -> PandasLinearModel:
        instance = cls.__new__(cls)
        with BytesIO(data) as buffer:
            instance._model = joblib.load(buffer)
        return instance
