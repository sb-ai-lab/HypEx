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
    """
    Executor for finding nearest neighbors using the FAISS library.

    This class leverages FAISS (Facebook AI Similarity Search) to perform
    efficient k-nearest neighbors (k-NN) matching between treatment and
    control groups. It supports one-sided or two-sided matching and can
    operate in different performance modes (base, fast, auto) depending
    on the dataset size and backend.

    Inherits from:
        MLExecutor: The base class for machine learning executors in the HypEx library.
    """
    def __init__(
        self,
        n_neighbors: int = 1,
        two_sides: bool = False,
        test_pairs: bool = False,
        grouping_role: ABCRole | None = None,
        key: Any = "",
        faiss_mode: Literal["base", "fast", "auto"] = "auto",
    ):
        """
        Initialize the FAISS nearest neighbors executor.

        Args:
            n_neighbors (int, optional): The number of nearest neighbors to find 
                for each observation. Defaults to 1.
            two_sides (bool, optional): If True, performs matching in both directions 
                (treatment to control and control to treatment). Defaults to False.
            test_pairs (bool, optional): If True, only matches test (treatment) 
                observations to control observations. Defaults to False.
            grouping_role (ABCRole | None, optional): The role defining the grouping 
                column (e.g., treatment assignment). Defaults to None.
            key (Any, optional): Optional identifier for the executor instance. 
                Defaults to "".
            faiss_mode (Literal["base", "fast", "auto"], optional): The FAISS execution 
                mode. "auto" automatically selects the best index type based on data 
                size, "fast" forces an optimized index, and "base" uses a standard 
                flat index. Defaults to "auto".
        """
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
        """
        Map local group indexes to global dataset indexes.

        This helper method translates the relative row indices returned by the FAISS
        search within a specific group back to the absolute indices of the original dataset.

        Args:
            local_indexes (Dataset): The dataset containing local match indices.
            data (tuple[str, Dataset]): A tuple containing the group name and the original
                Dataset for that group, used to resolve the global index.

        Returns:
            list[int] | list[list[int]] | Dataset: The updated dataset or list containing 
            global match indices. Returns the input unchanged if it is empty.
        """
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
        """
        Execute the core matching logic on grouped data.

        This class method processes the grouped data, applying the FAISS nearest neighbor
        search based on the specified matching configuration (one-sided, two-sided, or test-only).

        Args:
            grouping_data: Grouped dataset containing the control and treatment slices.
            tmp_roles: Temporary roles to be applied during the matching process.
            target_field (str | None, optional): The name of the target field. Defaults to None.
            n_neighbors (int | None, optional): Number of neighbors to match. Defaults to None.
            two_sides (bool | None, optional): Whether to perform bidirectional matching. Defaults to None.
            test_pairs (bool | None, optional): Whether to only match test pairs. Defaults to None.
            faiss_mode (Literal["base", "fast", "auto"], optional): FAISS execution mode. Defaults to "auto".
            **kwargs: Additional keyword arguments passed to the underlying FAISS extension.

        Returns:
            dict: A dictionary containing the matched datasets. Keys can be "test" and/or "control"
            depending on the `two_sides` and `test_pairs` flags.
        """
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
        """
        Resolve the backend-specific FAISS extension and execute the calculation.

        This method acts as a bridge to the backend-specific implementation (e.g., Pandas or Spark)
        of the FAISS algorithm.

        Args:
            data (Dataset): The baseline (control) dataset.
            test_data (Dataset | None, optional): The compared (test) dataset. Defaults to None.
            target_data (Dataset | None, optional): Optional target dataset. Defaults to None.
            n_neighbors (int | None, optional): Number of neighbors to find. Defaults to None.
            faiss_mode (Literal["base", "fast", "auto"], optional): FAISS execution mode. Defaults to "auto".
            **kwargs: Additional keyword arguments for the backend extension.

        Returns:
            Any: The result of the nearest neighbor search from the backend-specific FAISS extension.
        """
        faiss_cls = backend_factory.resolve_backend(FaissExtension, data)
        return faiss_cls(n_neighbors=n_neighbors or 1, faiss_mode=faiss_mode).calc(
            data=data, test_data=test_data
        )

    def fit(self, X: Dataset, Y: Dataset | None = None) -> MLExecutor:
        """
        Fit the FAISS index on the provided dataset.

        Args:
            X (Dataset): The dataset to build the FAISS index from.
            Y (Dataset | None, optional): Optional target dataset (not typically used for FAISS indexing).
                Defaults to None.

        Returns:
            MLExecutor: The fitted executor instance (or the backend-specific extension instance).
        """
        faiss_cls = backend_factory.resolve_backend(FaissExtension, X)
        return faiss_cls(self.n_neighbors, self.faiss_mode).fit(X=X, Y=Y)

    def predict(self, X: Dataset) -> Dataset:
        """
        Predict the nearest neighbors for the given dataset.

        Args:
            X (Dataset): The dataset for which to find nearest neighbors.

        Returns:
            Dataset: A dataset containing the indices of the nearest neighbors.
        """
        faiss_cls = backend_factory.resolve_backend(FaissExtension, X)
        return faiss_cls().predict(X)

    def execute(self, data: ExperimentData) -> ExperimentData:
        """
        Execute the FAISS nearest neighbors matching on the given experiment data.

        This method orchestrates the entire matching process:
        1. Retrieves grouping and feature fields.
        2. Groups the data or retrieves pre-grouped data.
        3. Calls the `calc` method to perform the FAISS search.
        4. Handles missing values (NaNs) by replacing them with a dummy match (-1) and warns the user.
        5. Formats the matched indices with appropriate roles and appends them to the result.
        6. Stores the final matched indices in the `ExperimentData` object.

        Args:
            data (ExperimentData): The experiment data containing the dataset to be matched.

        Returns:
            ExperimentData: The updated ExperimentData object with the matched indices stored
            in the `additional_fields` space.

        Raises:
            PairsNotFoundError: If the number of NaNs exceeds expectations or valid pairs cannot be found
            when `two_sides` is True.
        """
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

        # for result in compare_result.values():
        for group, result in compare_result.items():
            #TODO: find solution without `data` field
            result.data.to_pandas().to_csv(f"faiss_{group}.csv")
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
            # res_v has index similar to group data
            #`limit` may be removed
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
            matched_indexes = matched_indexes.append(t_index_field)
        if matched_indexes is not None:
            matched_indexes = matched_indexes.sort()
        if len(matched_indexes) < len(data.ds) and not self.two_sides:
            matched_indexes = matched_indexes.reindex(data.ds.index, fill_value=-1)
        elif len(matched_indexes) < len(data.ds) and self.two_sides:
            raise PairsNotFoundError
        return self._set_value(data, matched_indexes, key="matched")
