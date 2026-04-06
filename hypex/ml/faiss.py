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
    TreatmentRole,
)
from ..executor.ml_executor import MLExecutor
from ..executor.state import MLExecutorParams
from ..dataset.ml_data import MLExperimentData
from ..extensions.faiss import FaissExtension
from ..utils import ExperimentDataEnum
from ..utils.errors import PairsNotFoundError


class FaissMLExecutor(MLExecutor):
    """ML executor for Faiss-based nearest neighbor matching.

    This executor performs nearest neighbor matching between test and control groups
    using the Faiss library. It supports multiple matching modes:
    - One-sided matching (test->control or control->test)
    - Two-sided matching (both directions)
    - Test pairs mode (for ATC - average treatment effect on controls)

    Args:
        n_neighbors: Number of neighbors to find for each point
        two_sides: If True, match in both directions (test->control and control->test)
        test_pairs: If True, match control->test instead of test->control (for ATC)
        grouping_role: Role used to split data into groups (default: TreatmentRole)
        faiss_mode: Faiss index mode - "base", "fast", or "auto"
        key: Unique identifier for the executor
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
        self.n_neighbors = n_neighbors
        self.two_sides = two_sides
        self.test_pairs = test_pairs
        self.faiss_mode = faiss_mode
        self.grouping_role = grouping_role or TreatmentRole()
        self.target_role = FeatureRole()
        super().__init__(key=key)

    def execute_fit(self, data: MLExperimentData) -> MLExperimentData:
        """Fit mode: train faiss indices and save to MLExperimentData."""
        grouping_data = self._get_grouping_data(data)
        fitted_indices: dict[str, FaissExtension] = {}

        # Train indices for each direction
        if not self.test_pairs:
            # Train index on control group (for test->control matching)
            control_data = grouping_data[0][1]
            ext = FaissExtension(n_neighbors=self.n_neighbors, faiss_mode=self.faiss_mode)
            ext.calc(data=control_data, test_data=grouping_data[1][1], mode="fit")
            fitted_indices["test_to_control"] = ext

            if self.two_sides:
                # Train index on test group (for control->test matching)
                test_data = grouping_data[1][1]
                ext = FaissExtension(n_neighbors=self.n_neighbors, faiss_mode=self.faiss_mode)
                ext.calc(data=test_data, test_data=grouping_data[0][1], mode="fit")
                fitted_indices["control_to_test"] = ext
        else:
            # Train index on test group (for control->test matching in ATC mode)
            test_data = grouping_data[1][1]
            ext = FaissExtension(n_neighbors=self.n_neighbors, faiss_mode=self.faiss_mode)
            ext.calc(data=test_data, test_data=grouping_data[0][1], mode="fit")
            fitted_indices["control_to_test"] = ext

            if self.two_sides:
                # Also train reverse index
                control_data = grouping_data[0][1]
                ext = FaissExtension(n_neighbors=self.n_neighbors, faiss_mode=self.faiss_mode)
                ext.calc(data=control_data, test_data=grouping_data[1][1], mode="fit")
                fitted_indices["test_to_control"] = ext

        # Save fitted state to MLExperimentData
        state = MLExecutorParams(
            executor_id=self.id,
            executor_class=self.__class__.__name__,
            fitted_params={
                "indices": fitted_indices,
                "grouping_data": grouping_data,
            },
        )
        data.add_fitted_ml_executor(self.id, state)

        return data

    def execute_predict(self, data: MLExperimentData) -> MLExperimentData:
        """Predict mode: use fitted faiss indices from MLExperimentData."""
        # Load fitted state
        state = data.get_fitted_ml_executor(self.id)
        if state is None:
            raise ValueError(
                f"No fitted state found for {self.__class__.__name__} (id={self.id}). "
                "Run in FIT or FIT_PREDICT mode first."
            )

        fitted_indices = state.fitted_params["indices"]
        grouping_data = state.fitted_params["grouping_data"]

        # Execute matching using fitted indices
        compare_result = self._predict_with_fitted_indices(fitted_indices, grouping_data)

        # Process results
        return self._process_results(data, compare_result, grouping_data)

    def _predict_with_fitted_indices(
        self, fitted_indices: dict[str, FaissExtension], grouping_data: list
    ) -> dict:
        """Use fitted indices to find nearest neighbors."""
        result = {}

        if not self.test_pairs:
            # Standard mode: match test -> control
            ext = fitted_indices["test_to_control"]
            test_result = ext.calc(
                data=grouping_data[0][1],
                test_data=grouping_data[1][1],
                mode="predict"
            )
            result["test"] = self._set_global_match_indexes(test_result, grouping_data[0])

            if self.two_sides:
                ext = fitted_indices["control_to_test"]
                control_result = ext.calc(
                    data=grouping_data[1][1],
                    test_data=grouping_data[0][1],
                    mode="predict"
                )
                result["control"] = self._set_global_match_indexes(control_result, grouping_data[1])
        else:
            # Test pairs mode (ATC): match control -> test
            ext = fitted_indices["control_to_test"]
            control_result = ext.calc(
                data=grouping_data[1][1],
                test_data=grouping_data[0][1],
                mode="predict"
            )
            result["control"] = control_result

            if self.two_sides:
                ext = fitted_indices["test_to_control"]
                test_result = ext.calc(
                    data=grouping_data[0][1],
                    test_data=grouping_data[1][1],
                    mode="predict"
                )
                result["test"] = test_result

        return result

    def _process_results(
        self, data: MLExperimentData, compare_result: dict, grouping_data: list
    ) -> MLExperimentData:
        """Process matching results and save to data."""
        # Count and handle nans
        nans = 0
        for result in compare_result.values():
            nans += (
                sum(result.isna().sum().get_values(row="sum"))
                if self.n_neighbors > 1
                else result.isna().sum()
            )
            result = result.fillna(-1).astype({col: int for col in result.columns})

        if nans > 0:
            warn(
                f"Faiss returned {nans} nans, which were replaced with dummy matches. "
                "Check if the data is suitable for the test.",
                UserWarning,
            )

        # Build matched_indexes dataset
        matched_indexes = Dataset.create_empty()
        for res_k, res_v in compare_result.items():
            group = grouping_data[1][1] if res_k == "test" else grouping_data[0][1]
            t_index_field = res_v.loc[: len(group) - 1]

            # Check for nans
            n_nans = (
                t_index_field.isna().sum().get_values(row="sum")
                if t_index_field.shape[1] > 1
                else [t_index_field.isna().sum()]
            )
            if any(n_nans):
                raise PairsNotFoundError

            # Rename columns
            t_index_field = t_index_field.rename(
                {col: f"indexes_{i}" for i, col in enumerate(t_index_field.columns)}
            )

            # Append to matched_indexes
            matched_indexes = matched_indexes.append(
                Dataset.from_dict(
                    data={
                        col: t_index_field.get_values(column=col)
                        for col in t_index_field.columns
                    },
                    roles={
                        col: AdditionalMatchingRole() for col in t_index_field.columns
                    },
                    index=group.index,
                )
            ).sort()

        # Handle missing indices
        if len(matched_indexes) < len(data.ds) and not self.two_sides:
            matched_indexes = matched_indexes.reindex(data.ds.index, fill_value=-1)
        elif len(matched_indexes) < len(data.ds) and self.two_sides:
            raise PairsNotFoundError

        # Save to CSV for debugging
        matched_indexes.data.to_csv("matched_indexes.csv")

        # Save matched_indexes to data
        return self._set_value(data, matched_indexes, key="matched")

    def _get_fields(self, data: ExperimentData):
        """Get group and feature fields from data."""
        group_field = data.field_search(self.grouping_role)
        target_field = data.field_search(
            self.target_role, search_types=[int, float]
        )
        return group_field, target_field

    @property
    def search_types(self):
        return [int, float]

    @classmethod
    def _set_global_match_indexes(cls, local_indexes: Dataset, data: tuple) -> Dataset:
        """Convert local indexes to global indexes using original data index."""
        if len(local_indexes) == 0:
            return local_indexes

        global_indexes = local_indexes
        for col in local_indexes.columns:
            global_indexes[col] = data[1].index.take(
                local_indexes.get_values(column=col)
            )
        return global_indexes

    def _set_value(
        self, data: ExperimentData, value: Any, key: Any = None
    ) -> ExperimentData:
        """Save matched indexes to ExperimentData."""
        from ..utils import ID_SPLIT_SYMBOL

        for i in range(value.shape[1]):
            data.set_value(
                ExperimentDataEnum.additional_fields,
                f"{self.id}{ID_SPLIT_SYMBOL}{i}",
                value=value.iloc[:, i],
                key=key,
                role=AdditionalMatchingRole(),
            )
        return data

    def _get_grouping_data(self, data: MLExperimentData):
        """Get grouping data from MLExperimentData."""
        group_field, features_fields = self._get_fields(data=data)

        # Check for MahalanobisDistance groups
        distances_keys = data.get_ids(MahalanobisDistance, ExperimentDataEnum.groups)
        if len(distances_keys["MahalanobisDistance"]["groups"]) > 0:
            return list(
                data.groups[distances_keys["MahalanobisDistance"]["groups"][0]].items()
            )

        if group_field[0] in data.groups:
            return list(data.groups[group_field[0]].items())
        else:
            return data.ds.groupby(group_field, fields_list=features_fields)
