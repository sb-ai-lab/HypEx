from typing import Optional, Any, Dict

from hypex.comparators.distances import MahalanobisDistance
from hypex.dataset import Dataset, ABCRole, FeatureRole, ExperimentData, TargetRole
from hypex.executor import MLExecutor
from hypex.extensions.faiss import FaissExtension
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
    ) -> Dict:
        data = cls._inner_function(
            data=grouping_data[0][1],
            test_data=grouping_data[1][1],
            n_neighbors=n_neighbors or 1,
            **kwargs,
        )
        if two_sides is None or two_sides == False:
            return {"test": data}
        return {
            "test": data,
            "control": cls._inner_function(
                data=grouping_data[1][1],
                test_data=grouping_data[0][1],
                n_neighbors=n_neighbors or 1,
                **kwargs,
            ),
        }

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Optional[Dataset] = None,
        n_neighbors: Optional[int] = None,
        **kwargs,
    ) -> Any:
        return FaissExtension(n_neighbors=n_neighbors or 1).calc(
            data=data, test_data=test_data
        )

    def fit(self, X: Dataset, Y: Optional[Dataset] = None) -> "MLExecutor":
        return FaissExtension(self.n_neighbors).fit(X=X, Y=Y)

    def predict(self, X: Dataset) -> Dataset:
        return FaissExtension().predict(X)

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, features_fields = self._get_fields(data=data)
        if group_field[0] in data.groups:
            grouping_data = list(data.groups[group_field[0]].items())
        else:
            grouping_data = data.ds.groupby(group_field, fields_list=features_fields)
        distances_keys = data.get_ids(MahalanobisDistance, ExperimentDataEnum.groups)
        if len(distances_keys["MahalanobisDistance"]["groups"]) > 0:
            grouping_data = list(
                data.groups[distances_keys["MahalanobisDistance"]["groups"][0]].items()
            )
        compare_result = self.calc(
            data=data.ds,
            group_field=group_field,
            grouping_data=grouping_data,
            features_fields=features_fields,
            n_neighbors=self.n_neighbors,
            two_sides=self.two_sides,
        )
        matched_df = Dataset.create_empty()

        index_field = compare_result.fillna(-1)
        for i in range(len(compare_result.columns)):
            t_index_field = index_field[index_field.columns[i]]
            filtered_field = t_index_field.drop(
                t_index_field[t_index_field[t_index_field.columns[0]] == -1], axis=0
            )
            new_target = data.ds.iloc[
                list(map(lambda x: x[0], filtered_field.get_values()))
            ]
            new_target.index = filtered_field.index
            group = (
                grouping_data[0][1]
                if compare_result.columns[i] == "test"
                else grouping_data[1][1]
            )
            new_target = new_target.reindex(group.index, fill_value=0).rename(
                {field: field + "_matched" for field in new_target.columns}
            )
            matched_df = matched_df.append(new_target).sort()
        if len(matched_df) < len(data.ds):
            matched_df = matched_df.reindex(data.ds.index, fill_value=0)
        return self._set_value(data, matched_df, key="matched_df")
