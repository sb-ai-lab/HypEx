import numpy as np
from typing import Optional
from hypex.comparators.abstract import GroupComparator
from hypex.dataset import Dataset
from hypex.extensions.linalg import CholeskyExtension, InverseExtension


class MahalanobisDistance(GroupComparator):

    def _inner_function(self, data: Dataset, other: Optional[Dataset] = None):
        cov = (data.cov() + other.cov()) / 2 if other else data.cov()
        cholesky = CholeskyExtension().calc(cov)
        mahalanobis_transform = InverseExtension().calc(cholesky)
        yc = data.dot(mahalanobis_transform.transpose())
        if other: 
            yt = other.dot(mahalanobis_transform.transpose())
            return {self.id: {"control": yc, "test": yt}} 
        return {self.id: {"control": yc}} 
        