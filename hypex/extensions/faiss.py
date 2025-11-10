from __future__ import annotations

from typing import Literal

import faiss  # type: ignore
import numpy as np
import pandas as pd  # type: ignore

from ..dataset import AdditionalMatchingRole, Dataset
from .abstract import MLExtension


class FaissExtension(MLExtension):
    def __init__(
        self, n_neighbors: int = 1, faiss_mode: Literal["base", "fast", "auto"] = "auto"
    ):
        self.n_neighbors = n_neighbors
        self.faiss_mode = faiss_mode
        self.index = None
        super().__init__()

    @staticmethod
    def _prepare_indexes(index: np.ndarray, dist: np.ndarray, k: int):
        new = np.vstack(
            [
                np.concatenate(
                    [val[np.where(dist[i] == d)[0]] for d in sorted(set(dist[i]))[:k]]
                )
                for i, val in enumerate(index)
            ])
        return new

    def _predict(self, data: Dataset, test_data: Dataset, X: np.ndarray) -> pd.Series:
        dist, indexes = self.index.search(X, k=self.n_neighbors)
        if self.n_neighbors == 1:
            equal_dist = list(map(lambda x: np.where(x == x[0])[0], dist))
            indexes = [
                (
                    int(index[dist][0])
                    if abs(index[dist][0]) <= len(data) + len(test_data)
                    else -1
                )
                for index, dist in zip(indexes, equal_dist)
            ]
        else:
            indexes = self._prepare_indexes(indexes, dist, self.n_neighbors)
        return self.result_to_dataset(result=indexes, roles={})

    def _calc_pandas(
        self,
        data: Dataset,
        test_data: Dataset | None = None,
        mode: Literal["auto", "fit", "predict"] | None = None,
        **kwargs,
    ):
        mode = mode or "auto"
        X = data.data.values
        test = test_data.data.values
        if mode in ["auto", "fit"]:
            self.index = faiss.IndexFlatL2(X.shape[1])
            if ((
                len(X) > 1_000_000 and self.faiss_mode == "auto"
            ) or self.faiss_mode == "fast"
            ) and len(X) > 1_000 and len(test) > 1_000:
                self.index = faiss.IndexIVFFlat(self.index, X.shape[1], 1000)
                self.index.train(X)
            self.index.add(X)
        if mode in ["auto", "predict"]:
            if test_data is None:
                raise ValueError("test_data is needed for evaluation")
            X = test_data.data.values if mode == "auto" else data.data.values
            return self._predict(data, test_data, X)
        return self

    def fit(self, X: Dataset, Y: Dataset | None = None, **kwargs):
        return super().calc(X, target_data=Y, mode="fit", **kwargs)

    def predict(self, X: Dataset, **kwargs) -> Dataset:
        return self.result_to_dataset(
            super().calc(X, mode="predict", **kwargs), AdditionalMatchingRole()
        )
