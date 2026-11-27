from __future__ import annotations

from inspect import signature
from io import BytesIO
from typing import Any, ClassVar

import joblib
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

from .....dataset import ABCRole, Dataset
from .....dataset.backends import PandasDataset
from .....utils.registry import backend_factory
from ...linear import LinearModel, _l1_ratio


@backend_factory.register(LinearModel, PandasDataset)
class PandasLinearModel(LinearModel):
    PARAMS_CONVERSION: ClassVar[dict[str, str]] = {
        "fit_intercept": "fit_intercept",
        "reg_param": "alpha",
        "elastic_net_param": "l1_ratio",
        "solver": "solver",
        "max_iter": "max_iter",
        "tol": "tol",
    }

    def _fit_native(self, data: Dataset, features: list[str], target: str) -> None:
        ratio = _l1_ratio(self.elastic_net_param, self.reg_type)
        estimator = (
            LinearRegression
            if self.reg_param == 0
            else {0.0: Ridge, 1.0: Lasso}.get(ratio, ElasticNet)
        )
        supported = signature(estimator).parameters
        if "solver" not in supported and self.solver not in (None, "auto"):
            raise ValueError(
                f"{estimator.__name__} does not accept solver={self.solver!r}."
            )
        params = self.model_params
        params["elastic_net_param"] = ratio
        params["solver"] = (
            "cholesky" if self.solver == "normal" else self.solver or "auto"
        )
        native_model = estimator(
            **{
                native_name: params[name]
                for name, native_name in self.PARAMS_CONVERSION.items()
                if native_name in supported
            }
        )

        frame = data.data
        self._model = native_model.fit(frame[features], frame[target])

    def _predict_native(
        self, X: Dataset, features: list[str], output: str, role: ABCRole
    ) -> Dataset:
        X_frame = X.data[features]
        result = X_frame.iloc[:, :0].copy()
        result[output] = self._model.predict(X_frame) if len(X_frame) else []
        return self.result_to_dataset(result, roles={output: role}, small=False)

    def dumps(self) -> bytes:
        with BytesIO() as buffer:
            joblib.dump(self._model, buffer)
            return buffer.getvalue()

    @classmethod
    def loads(cls, data: bytes, **params: Any) -> PandasLinearModel:
        instance = cls(**params)
        with BytesIO(data) as buffer:
            instance._model = joblib.load(buffer)
        return instance
