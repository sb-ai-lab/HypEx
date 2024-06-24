from typing import Optional

from hypex.comparators.abstract import GroupComparator
from hypex.dataset import Dataset
from hypex.extensions.linalg import CholeskyExtension, InverseExtension


class MahalanobisDistance(GroupComparator):

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ):
        test_data = cls._check_test_data(test_data)
        cov = (data.cov() + test_data.cov()) / 2 if test_data else data.cov()
        cholesky = CholeskyExtension().calc(cov)
        mahalanobis_transform = InverseExtension().calc(cholesky)
        y_control = data.dot(mahalanobis_transform.transpose())
        if test_data:
            y_test = test_data.dot(mahalanobis_transform.transpose())
            return {"control": y_control, "test": y_test}
        return {"control": y_control}
