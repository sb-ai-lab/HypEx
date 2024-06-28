from typing import Optional, Literal

import faiss  # type: ignore
import numpy as np
import pandas as pd  # type: ignore

from hypex.dataset import Dataset, MatchingRole
from hypex.extensions.abstract import MLExtension


class FaissExtension(MLExtension):
    def __init__(self, n_neighbors: int = 1):
        self.n_neighbors = n_neighbors
        super().__init__()

    def _calc_pandas(
        self,
        data: Dataset,
        test_data: Optional[Dataset] = None,
        mode: Optional[Literal["auto", "fit", "predict"]] = None,
        **kwargs,
    ):
        mode = mode or "auto"
        X = data.data.values
        if mode in ["auto", "fit"]:
            self.index = faiss.IndexFlatL2(X.shape[1])
            self.index.add(X)
        if mode in ["auto", "predict"]:
            X = test_data.data.values if mode == "auto" else X
            dist, indexes = self.index.search(X, k=self.n_neighbors)
            if self.n_neighbors == 1:
                equal_dist = list(map(lambda x: np.where(x == x[0])[0], dist))
                indexes = [
                    int(i[j][0]) if abs(i[j][0]) <= len(data) + len(test_data) else -1
                    for i, j in zip(indexes, equal_dist)
                ]
            else:
                indexes = self._prepare_indexes(indexes, dist, self.n_neighbors)
            return pd.Series(indexes)
        return self

    def fit(self, X: Dataset, y: Optional[Dataset] = None, **kwargs):
        return super().calc(X, target_data=y, mode="fit", **kwargs)

    def predict(self, X: Dataset, **kwargs) -> Dataset:
        return self.result_to_dataset(
            super().calc(X, mode="predict", **kwargs), MatchingRole()
        )

    @staticmethod
    def _prepare_indexes(index: np.ndarray, dist: np.ndarray, k: int):
        new = [
            np.concatenate(
                [val[np.where(dist[i] == d)[0]] for d in sorted(set(dist[i]))[:k]]
            )
            for i, val in enumerate(index)
        ]
        return new
