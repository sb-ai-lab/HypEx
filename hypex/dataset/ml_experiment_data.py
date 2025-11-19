from __future__ import annotations

from typing import Any, Literal

from .dataset import Dataset, ExperimentData
from ..transformers.abstract import Transformer


class MLExperimentData:
    def __init__(
        self,
        train_data: Dataset,
        test_data: Dataset | None = None,
        val_data: Dataset | None = None,
    ):
        self.train_data = ExperimentData(train_data)
        self.test_data = ExperimentData(test_data) if test_data is not None else None
        self.val_data = ExperimentData(val_data) if val_data is not None else None
        self._fitted_transformers: list[tuple[str, Transformer]] = []

    def get_data(
        self, mode: Literal["train", "test", "val", "all"]
    ) -> ExperimentData | dict[str, ExperimentData]:
        if mode == "train":
            return self.train_data
        elif mode == "test":
            if self.test_data is None:
                raise ValueError("Test data is not available")
            return self.test_data
        elif mode == "val":
            if self.val_data is None:
                raise ValueError("Validation data is not available")
            return self.val_data
        elif mode == "all":
            result = {"train": self.train_data}
            if self.test_data is not None:
                result["test"] = self.test_data
            if self.val_data is not None:
                result["val"] = self.val_data
            return result
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def apply_transformer(
        self, transformer: Transformer, fit_on: Literal["train"] = "train"
    ) -> MLExperimentData:
        if fit_on == "train":
            fit_data = self.train_data
        else:
            raise ValueError(f"fit_on must be 'train', got {fit_on}")

        transformed_train = transformer.execute(fit_data)
        self.train_data = transformed_train

        self._fitted_transformers.append((transformer.id, transformer))

        if self.test_data is not None:
            self.test_data = transformer.execute(self.test_data)
        if self.val_data is not None:
            self.val_data = transformer.execute(self.val_data)

        return self

    def reverse_transform(
        self, data: Dataset, transformer_id: str | None = None
    ) -> Dataset:
        if transformer_id is not None:
            for tid, transformer in reversed(self._fitted_transformers):
                if tid == transformer_id:
                    if hasattr(transformer, "reverse_transform"):
                        return transformer.reverse_transform(data)
                    else:
                        raise AttributeError(
                            f"Transformer {transformer_id} does not support reverse_transform"
                        )
            raise ValueError(f"Transformer {transformer_id} not found")
        else:
            result = data
            for tid, transformer in reversed(self._fitted_transformers):
                if hasattr(transformer, "reverse_transform"):
                    result = transformer.reverse_transform(result)
            return result

    def get_artifacts(self) -> dict[str, Any]:
        artifacts = {
            "train_shape": self.train_data.ds.shape,
            "fitted_transformers": [tid for tid, _ in self._fitted_transformers],
        }
        if self.test_data is not None:
            artifacts["test_shape"] = self.test_data.ds.shape
        if self.val_data is not None:
            artifacts["val_shape"] = self.val_data.ds.shape
        return artifacts

    def save_artifacts(self, path: str, format: Literal["pickle", "joblib"] = "pickle") -> None:
        import pickle

        try:
            import joblib
        except ImportError:
            joblib = None

        artifacts = {
            "train_data": self.train_data,
            "test_data": self.test_data,
            "val_data": self.val_data,
            "fitted_transformers": self._fitted_transformers,
        }

        if format == "pickle":
            with open(path, "wb") as f:
                pickle.dump(artifacts, f)
        elif format == "joblib":
            if joblib is None:
                raise ImportError("joblib is required for joblib format")
            joblib.dump(artifacts, path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    @staticmethod
    def load_artifacts(path: str, format: Literal["pickle", "joblib"] = "pickle") -> MLExperimentData:
        import pickle

        try:
            import joblib
        except ImportError:
            joblib = None

        if format == "pickle":
            with open(path, "rb") as f:
                artifacts = pickle.load(f)
        elif format == "joblib":
            if joblib is None:
                raise ImportError("joblib is required for joblib format")
            artifacts = joblib.load(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        ml_data = MLExperimentData(
            train_data=artifacts["train_data"].ds,
            test_data=artifacts["test_data"].ds if artifacts["test_data"] is not None else None,
            val_data=artifacts["val_data"].ds if artifacts["val_data"] is not None else None,
        )
        ml_data._fitted_transformers = artifacts["fitted_transformers"]
        return ml_data
