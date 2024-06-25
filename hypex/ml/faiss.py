from typing import Optional, Any, List

from hypex.comparators.distances import MahalanobisDistance
from hypex.dataset import Dataset, ABCRole, FeatureRole, ExperimentData
from hypex.executor import MLExecutor
from hypex.extensions.ml import FaissExtension
from hypex.utils import SpaceEnum, ExperimentDataEnum


class FaissNearestNeighbors(MLExecutor):
    def __init__(
        self,
        n_neighbors: int = 1,
        two_sides: bool = False,
        grouping_role: Optional[ABCRole] = None,
        space: SpaceEnum = SpaceEnum.auto,
        key: Any = "",
    ):
        self.n_neighbors = n_neighbors
        self.two_sides = two_sides
        super().__init__(
            grouping_role=grouping_role, target_role=FeatureRole(), space=space, key=key
        )

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        n_neighbors: Optional[int] = None,
        two_sides: Optional[bool] = None,
        **kwargs,
    ) -> List:
        data = cls._inner_function(
            data=grouping_data[0][1],
            test_data=grouping_data[1][1],
            n_neighbors=n_neighbors or 1,
            **kwargs,
        )
        if two_sides is None or two_sides == False:
            return data
        return data + cls._inner_function(
            data=grouping_data[1][1],
            test_data=grouping_data[0][1],
            n_neighbors=n_neighbors,
            **kwargs,
        )

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Optional[Dataset] = None,
        target_data: Optional[Dataset] = None,
        n_neighbors: Optional[int] = None,
        **kwargs,
    ) -> Any:
        return FaissExtension(n_neighbors=n_neighbors or 1).calc(
            data=data, test_data=test_data, target_data=target_data
        )

    def fit(self, X: Dataset, Y: Dataset) -> "MLExecutor":
        return FaissExtension(self.n_neighbors).fit(X=X, y=Y)

    def predict(self, data: Dataset) -> Dataset:
        return FaissExtension().predict(data)

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, features_fields = self._get_fields(data=data)
        if group_field[0] in data.groups:
            grouping_data = list(data.groups[group_field[0]].items())
        else:
            grouping_data = None
        distances_keys = data.get_ids(MahalanobisDistance, ExperimentDataEnum.groups)
        if len(distances_keys[MahalanobisDistance]) > 0:
            grouping_data = list(
                data.groups[distances_keys[MahalanobisDistance]["groups"][0]].items()
            )
        compare_result = self.calc(
            data=data.ds,
            group_field=group_field,
            grouping_data=grouping_data,
            features_fields=features_fields,
            n_neighbors=self.n_neighbors,
            two_sides=self.two_sides,
        )
        return self._set_value(data, compare_result)
