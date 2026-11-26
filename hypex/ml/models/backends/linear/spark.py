from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar, Literal
from zipfile import ZipFile

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.sql import functions as F

from .....dataset import ABCRole, Dataset
from .....dataset.backends import SparkDataset
from .....utils.registry import backend_factory
from ...linear import LinearModel, _l1_ratio

_FEATURES = "_linear_features"
_PREDICTION = "_linear_prediction"


@backend_factory.register(LinearModel, SparkDataset)
class SparkLinearModel:
    PARAMS_CONVERSION: ClassVar[dict[str, str]] = {
        "fit_intercept": "fitIntercept",
        "reg_param": "regParam",
        "elastic_net_param": "elasticNetParam",
        "solver": "solver",
        "max_iter": "maxIter",
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
        params = {
            "fit_intercept": fit_intercept,
            "reg_param": reg_param,
            "elastic_net_param": _l1_ratio(elastic_net_param, reg_type)
            if reg_param
            else 0.0,
            "solver": "normal" if solver == "cholesky" else solver or "auto",
            "max_iter": max_iter,
            "tol": tol,
        }
        self._estimator = LinearRegression(
            featuresCol=_FEATURES,
            predictionCol=_PREDICTION,
            **{
                native_name: params[name]
                for name, native_name in self.PARAMS_CONVERSION.items()
            },
        )
        self._model: LinearRegressionModel

    def fit(self, data: Dataset, features: list[str], target: str) -> None:
        training = data.data.select(*features, target)
        assembler = VectorAssembler(inputCols=features, outputCol=_FEATURES)
        self._model = self._estimator.setLabelCol(target).fit(
            assembler.transform(training)
        )

    def predict(
        self, X: Dataset, features: list[str], output: str, role: ABCRole
    ) -> Dataset:
        raw = X.raw_data
        index_columns = [f"_linear_index_{i}" for i in range(raw.index.nlevels)]
        frame = raw.to_spark(index_col=index_columns).select(*index_columns, *features)
        assembler = VectorAssembler(
            inputCols=features, outputCol=self._model.getFeaturesCol()
        )
        scored = self._model.transform(assembler.transform(frame))
        result = (
            scored.select(
                *index_columns, F.col(self._model.getPredictionCol()).alias(output)
            )
            .pandas_api(index_col=index_columns)
            .rename_axis(index=raw.index.names)
        )
        return LinearModel.result_to_dataset(result, roles={output: role}, small=False)

    def dumps(self) -> bytes:
        with TemporaryDirectory() as temporary, BytesIO() as buffer:
            root = Path(temporary) / "model"
            self._model.write().save(root.resolve().as_uri())
            with ZipFile(buffer, "w") as archive:
                for path in root.rglob("*"):
                    if path.is_file():
                        archive.write(path, path.relative_to(root))
            return buffer.getvalue()

    @classmethod
    def loads(cls, data: bytes) -> SparkLinearModel:
        instance = cls.__new__(cls)
        with TemporaryDirectory() as temporary, BytesIO(data) as buffer:
            with ZipFile(buffer) as archive:
                archive.extractall(temporary)
            instance._model = LinearRegressionModel.load(
                Path(temporary).resolve().as_uri()
            )
        return instance
