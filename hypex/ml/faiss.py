from __future__ import annotations

from typing import Any, Literal

from ..comparators.distances import MahalanobisDistance
from ..dataset import (
    ABCRole,
    AdditionalMatchingRole,
    Dataset,
    ExperimentData,
    FeatureRole,
)
from ..executor import MLExecutor
from ..extensions.faiss import FaissExtension
from ..utils import ExperimentDataEnum
from ..utils.errors import PairsNotFoundError


class FaissNearestNeighbors(MLExecutor):
    def __init__(
        self,
        n_neighbors: int = 1,
        two_sides: bool = False,
        test_pairs: bool = False,
        grouping_role: ABCRole | None = None,
        key: Any = "",
        faiss_mode: Literal["base", "fast", "auto"] = "auto",
    ):
        self.n_neighbors = n_neighbors
        self.two_sides = two_sides
        self.test_pairs = test_pairs
        self.faiss_mode = faiss_mode
        super().__init__(
            grouping_role=grouping_role, target_role=FeatureRole(), key=key
        )

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_field: str | None = None,
        n_neighbors: int | None = None,
        two_sides: bool | None = None,
        test_pairs: bool | None = None,
        faiss_mode: Literal["base", "fast", "auto"] = "auto",
        **kwargs,
    ) -> dict:
        if test_pairs is not True:
            data = cls._inner_function(
                data=grouping_data[0][1],
                test_data=grouping_data[1][1],
                n_neighbors=n_neighbors or 1,
                faiss_mode=faiss_mode,
                **kwargs,
            )
            if two_sides is not True:
                return {"test": data}
            return {
                "test": data,
                "control": cls._inner_function(
                    data=grouping_data[1][1],
                    test_data=grouping_data[0][1],
                    n_neighbors=n_neighbors or 1,
                    faiss_mode=faiss_mode,
                    **kwargs,
                ),
            }
        data = cls._inner_function(
            data=grouping_data[1][1],
            test_data=grouping_data[0][1],
            n_neighbors=n_neighbors or 1,
            faiss_mode=faiss_mode,
            **kwargs,
        )
        if two_sides is not True:
            return {"control": data}
        return {
            "control": data,
            "test": cls._inner_function(
                data=grouping_data[1][1],
                test_data=grouping_data[0][1],
                n_neighbors=n_neighbors or 1,
                faiss_mode=faiss_mode,
                **kwargs,
            ),
        }

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Dataset | None = None,
        target_data: Dataset | None = None,
        n_neighbors: int | None = None,
        faiss_mode: Literal["base", "fast", "auto"] = "auto",
        **kwargs,
    ) -> Any:
        return FaissExtension(n_neighbors=n_neighbors or 1, faiss_mode=faiss_mode).calc(
            data=data, test_data=test_data
        )

    def fit(self, X: Dataset, Y: Dataset | None = None) -> MLExecutor:
        return FaissExtension(self.n_neighbors, self.faiss_mode).fit(X=X, Y=Y)

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
            faiss_mode=self.faiss_mode,
            two_sides=self.two_sides,
            test_pairs=self.test_pairs,
        )
        ds = data.ds.groupby(group_field)
        matched_indexes = Dataset.create_empty()
        for i in range(len(compare_result.columns)):
            group = (
                grouping_data[1][1]
                if compare_result.columns[i] == "test"
                else grouping_data[0][1]
            )
            t_ds = ds[0][1] if compare_result.columns[i] == "test" else ds[1][1]
            t_index_field = (
                compare_result[compare_result.columns[i]]
                .loc[: len(group) - 1]
                .rename({compare_result.columns[i]: "indexes"})
            )
            if t_index_field.isna().sum() > 0:
                raise PairsNotFoundError
            matched_indexes = matched_indexes.append(
                Dataset.from_dict(
                    data={
                        "indexes": t_ds.iloc[
                            list(map(lambda x: int(x[0]), t_index_field.get_values()))
                        ].index
                    },
                    roles={"indexes": AdditionalMatchingRole()},
                    index=group.index,
                )
            ).sort()
        if len(matched_indexes) < len(data.ds) and not self.two_sides:
            matched_indexes = matched_indexes.reindex(data.ds.index, fill_value=-1)
        elif len(matched_indexes) < len(data.ds) and self.two_sides:
            raise PairsNotFoundError
        return self._set_value(data, matched_indexes, key="matched")
