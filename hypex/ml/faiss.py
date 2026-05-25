from __future__ import annotations

from typing import Any, Literal
from warnings import warn

from ..comparators.distances import MahalanobisDistance
from ..dataset import (
    ABCRole,
    AdditionalMatchingRole,
    Dataset,
    ExperimentData,
    FeatureRole,
    InfoRole,
)
from ..executor import MLExecutor
from ..extensions.faiss import FaissExtension
from ..utils import ExperimentDataEnum
from ..utils.errors import PairsNotFoundError
from ..utils.registry import backend_factory


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
            grouping_role=grouping_role,
            target_role=FeatureRole(),
            key=key,
        )

    @classmethod
    def _set_global_match_indexes(
        cls, local_indexes: Dataset, data: tuple[str, Dataset]
    ) -> list[int, list[int]]:
        if len(local_indexes) == 0:
            return local_indexes
        global_indexes = local_indexes
        for col in local_indexes.columns:
            global_indexes[col] = data[1].index.take(
                local_indexes.get_values(column=col)
            )
        return global_indexes

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        tmp_roles, 
        target_field: str | None = None,
        n_neighbors: int | None = None,
        two_sides: bool | None = None,
        test_pairs: bool | None = None,
        faiss_mode: Literal["base", "fast", "auto"] = "auto",
        **kwargs,
    ) -> dict:
        (control_idx, _data), (test_idx, _test_data), *_ = grouping_data
        _data.tmp_roles = tmp_roles
        if test_pairs is not True:
            test_data = cls._inner_function(
                data=_data,
                test_data=_test_data,
                n_neighbors=n_neighbors or 1,
                faiss_mode=faiss_mode,
                **kwargs,
            )
            # This isn't nessesary due to `IndexIDMap`
            # test_data = cls._set_global_match_indexes(test_data, (control_idx, _data))
            if two_sides is not True:
                return {"test": test_data}
            control_data = cls._inner_function(
                data=_test_data,
                test_data=_data,
                n_neighbors=n_neighbors or 1,
                faiss_mode=faiss_mode,
                **kwargs,
            )
            # This isn't nessesary due to `IndexIDMap`
            # control_data = cls._set_global_match_indexes(control_data, (test_idx, _test_data))
            return {
                "test": test_data,
                "control": control_data,
            }
        data = cls._inner_function(
            data=_test_data,
            test_data=_data,
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
        
        faiss_cls = backend_factory.resolve_backend(FaissExtension, data)
        return faiss_cls(n_neighbors=n_neighbors or 1, faiss_mode=faiss_mode).calc(
            data=data, test_data=test_data
        )

    def fit(self, X: Dataset, Y: Dataset | None = None) -> MLExecutor:
        faiss_cls = backend_factory.resolve_backend(FaissExtension, X)
        return faiss_cls(self.n_neighbors, self.faiss_mode).fit(X=X, Y=Y)

    def predict(self, X: Dataset) -> Dataset:
        faiss_cls = backend_factory.resolve_backend(FaissExtension, X)
        return faiss_cls().predict(X)

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, features_fields = self._get_fields(data=data)
        if group_field[0] in data.groups:
            grouping_data = list(data.groups[group_field[0]].items())
        else:
            grouping_data = list(data.ds[group_field + features_fields].groupby(group_field))
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
        nans = 0

        for result in compare_result.values():
            #TODO: find solution without `data` field
            nans += (
                result.data.isna().sum().sum()
            )
            result = result.fillna(-1).astype({col: int for col in result.columns})
        if nans > 0:
            warn(
                f"Faiss returned {nans} nans, which were replaced with dummy matches. Check if the data is suitable for the test.",
                UserWarning,
            )
        matched_indexes = Dataset.create_empty(
            backend=data.ds.backend_type,
            session=data.ds.session
        )
        # matched_indexes.index.name = None
        for res_k, res_v in compare_result.items():
            group = grouping_data[1][1] if res_k == "test" else grouping_data[0][1]
            t_index_field: Dataset = res_v.limit(len(group))

            n_nans = t_index_field.data.isna().sum().sum()

            if n_nans:
                raise PairsNotFoundError
            t_index_field = t_index_field.rename(
                {col: f"indexes_{i}" for i, col in enumerate(t_index_field.columns)}
            )
            t_index_field.roles = {
                col: AdditionalMatchingRole() for col in  t_index_field.columns
            }

            # TODO: not supported in pyspark, maybe remove it or find a solution
            # t_index_field.index = group.index 
            t_index_field = t_index_field.add_column(group.index, {"index": InfoRole()}).set_index("index")
            t_index_field.index.name = None
            matched_indexes = matched_indexes.append(t_index_field)
        if matched_indexes is not None:
            matched_indexes = matched_indexes.sort()
        if len(matched_indexes) < len(data.ds) and not self.two_sides:
            matched_indexes = matched_indexes.reindex(data.ds.index, fill_value=-1)
        elif len(matched_indexes) < len(data.ds) and self.two_sides:
            raise PairsNotFoundError
        return self._set_value(data, matched_indexes, key="matched")
