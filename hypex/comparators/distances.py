from typing import Optional, List, Dict, Any

from hypex.dataset import Dataset, ExperimentData, FeatureRole
from hypex.executor import GroupCalculator
from hypex.extensions.scipy_linalg import CholeskyExtension, InverseExtension
from hypex.utils import ExperimentDataEnum


class MahalanobisDistance(GroupCalculator):

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_fields: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict:
        result = {}
        for i in range(1, len(grouping_data)):
            result.update(
                cls._inner_function(
                    data=(
                        grouping_data[0][1][target_fields]
                        if target_fields
                        else grouping_data[0][1]
                    ),
                    test_data=(
                        grouping_data[i][1][target_fields]
                        if target_fields
                        else grouping_data[i][1]
                    ),
                    **kwargs,
                )
            )
        return result

    def _set_value(
        self, data: ExperimentData, value: Optional[Dict] = None, key: Any = None
    ) -> ExperimentData:
        for key, value_ in value.items():
            data = data.set_value(
                ExperimentDataEnum.groups,
                self.id,
                str(self.__class__.__name__),
                value_,
                key=key,
            )
        return data

    def _get_fields(self, data: ExperimentData):
        group_field = self._field_searching(data, self.grouping_role)
        target_fields = self._field_searching(
            data, FeatureRole(), search_types=self.search_types
        )
        return group_field, target_fields

    @property
    def search_types(self) -> Optional[List[type]]:
        return [int, float]

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
