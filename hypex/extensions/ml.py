from typing import Optional, Literal

import faiss  # type: ignore
import numpy as np

from hypex.dataset import Dataset, MatchingRole
from hypex.extensions.abstract import MLExtension


class FaissExtension(MLExtension):
    def __init__(self, n_neighbors: int = 10):
        self.n_neighbors = n_neighbors
        super().__init__()

    def _calc_pandas(
        self,
        data: Dataset,
        other: Optional[Dataset] = None,
        test: Optional[Dataset] = None,
        mode: Optional[Literal["auto", "fit", "predict"]] = None,
        **kwargs,
    ):
        mode = "auto" if mode is None else mode
        X = data.data.values
        if mode in ["auto", "fit"]:
            y = other.data.values
            self.index = faiss.IndexFlatL2(X.shape[1])
            self.index.add(y)
        if mode in ["auto", "predict"]:
            X = test.data.values if mode == "auto" else X
            dist, indexes = self.index.search(X, k=self.n_neighbors)
            if self.n_neighbors == 1:
                equal_dist = list(map(np.where(X == X[0])[0], dist))
                indexes = [i[j] for i, j in zip(indexes, equal_dist)]
            else:
                indexes = self._prepare_indexes(indexes, dist, self.n_neighbors)
            return indexes
        return self

    def fit(self, data: Dataset, other: Optional[Dataset] = None, **kwargs):
        return super().calc(data, other=other, mode="fit", **kwargs)

    def predict(self, data: Dataset, **kwargs) -> Dataset:
        return self.result_to_dataset(
            super().calc(data, mode="predict", **kwargs), MatchingRole()
        )

    @staticmethod
    def _prepare_indexes(index: np.array, dist: np.array, k: int):
        new = [
            np.concatenate(
                [val[np.where(dist[i] == d)[0]] for d in sorted(set(dist[i]))[:k]]
            )
            for i, val in enumerate(index)
        ]
        return new
