from __future__ import annotations

from typing import Any

from ..dataset import Dataset, ExperimentData
from ..dataset.roles import FeatureRole, PreTargetRole, TargetRole
from ..executor import Calculator
from ..utils import ExperimentDataEnum


class CUPACDataSplitter(Calculator):
    """
    Splitter that prepares temporal data structures for CUPAC ML models.
    
    This splitter organizes temporal fields (targets and features with lags) into
    training and prediction structures used by CUPACExecutor for ML model training.
    
    Process:
    1. Groups target and feature fields by their temporal lags
    2. Identifies cofounders (features used for prediction)
    3. Structures data into X_train, Y_train for model training
    4. Creates X_predict for current period adjustment (if applicable)
    
    The results are stored in data.ml["splits"] and consumed by CUPACExecutor.
    """
    
    def __init__(self, key: Any = ""):
        super().__init__(key)
    
    @staticmethod
    def _inner_function(data: ExperimentData) -> dict[str, dict[str, list]]:
        """
        Prepare data for CUPAC by organizing temporal fields into training and prediction structures.

        This method performs complex data organization:
        1. Groups target and feature fields by their temporal lags
        2. Identifies cofounders (features used for prediction)
        3. Structures data into X_train, Y_train for model training
        4. Creates X_predict for current period adjustment (if applicable)

        Args:
            data (ExperimentData): Input experiment data with temporal roles.

        Returns:
            dict: Nested dictionary with structure:
                {target_name: {
                    'X_train': [[feature_cols_at_lag_n], ..., [feature_cols_at_lag_2]],
                    'Y_train': [target_at_lag_n-1, ..., target_at_lag_1],
                    'X_predict': [[feature_cols_at_lag_1]] (optional, only for real targets)
                }}
        """

        def agg_temporal_fields(role, data_obj) -> dict[str, dict]:
            """
            Aggregate fields by their temporal lags.

            Returns:
                dict: {field_name: {lag: field_name_with_lag}} or {field_name: {}}
                      Empty dict means lag=0 or None (current period).
            """
            fields = {}
            searched_fields = data_obj.field_search(
                (
                    [TargetRole(), PreTargetRole()]
                    if isinstance(role, TargetRole)
                    else role
                ),
                search_types=[int, float],
            )

            searched_lags = [
                (
                    field,
                    (
                        data_obj.ds.roles[field].lag
                        if not isinstance(data_obj.ds.roles[field], TargetRole)
                        else 0
                    ),
                )
                for field in searched_fields
            ]
            # Sort, treating None as 0
            sorted_fields_by_lag = sorted(searched_lags, key=lambda x: x[1] if x[1] is not None else 0)
            for field, lag in sorted_fields_by_lag:
                if lag in [None, 0]:
                    fields[field] = {}
                else:
                    if data_obj.ds.roles[field].parent not in fields:
                        fields[data_obj.ds.roles[field].parent] = {}
                    fields[data_obj.ds.roles[field].parent][lag] = field

            return fields

        def agg_train_predict_x(mode: str, lag: int) -> None:
            """
            Aggregate features and targets for a specific lag into training/prediction sets.

            For each cofounder feature, creates a list structure where:
            - First and last lags start new sublists
            - Intermediate lags append to existing sublists
            This groups temporal sequences of the same feature together.
            """
            for i, cofounder in enumerate(cofounders[target]):
                if lag in [1, max_lags[target]]:
                    cupac_data[target][mode].append([features[cofounder][lag]])
                else:
                    cupac_data[target][mode][i].append(cofounder)

            cupac_data[target][mode].append([targets[target][lag]])

        cupac_data = {}
        targets = agg_temporal_fields(TargetRole(), data)
        features = agg_temporal_fields(FeatureRole(), data)

        # Determine cofounders (features used for prediction) for each target
        cofounders = {}
        for target in targets:
            if target in data.ds.columns:
                cofounders[target] = data.ds.roles[target].cofounders
            else:
                # For virtual targets, get cofounders from the earliest lag
                min_lag = min(targets[target].keys())
                cofounders[target] = data.ds.roles[targets[target][min_lag]].cofounders

                if cofounders[target] is None:
                    raise ValueError(
                        f"Cofounders must be defined in the first lag for virtual target '{target}'"
                    )

        # Calculate maximum lag for each target (max across target lags and cofounder feature lags)
        max_lags = {}
        for target, lags in targets.items():
            if lags:
                max_lag = max(lags.keys())
                for feature in cofounders[target]:
                    if features.get(feature):
                        max_lag = max(max(features[feature].keys()), max_lag)
            max_lags[target] = max_lag

        # Build training and prediction structures for each target
        for target in targets.keys():

            cupac_data[target] = {"X_train": [], "Y_train": []}
            # Only real targets (not virtual) need prediction
            if target in data.ds.columns:
                cupac_data[target]["X_predict"] = []

            # Build training data: iterate from max_lag down to 2
            # Each iteration creates X_train entry for lag and Y_train entry for lag-1
            for lag in range(max_lags[target], 1, -1):
                agg_train_predict_x("X_train", lag)
                cupac_data[target]["Y_train"].append(targets[target][lag - 1])

            # Build prediction data for current period (lag=1) if applicable
            if "X_predict" in cupac_data[target].keys():
                agg_train_predict_x("X_predict", 1)

        return cupac_data
    
    def execute(self, data: ExperimentData) -> ExperimentData:
        """
        Execute CUPAC data splitting.
        
        Organizes temporal data and stores result in data.ml["splits"].
        
        Args:
            data: ExperimentData with temporal roles
            
        Returns:
            ExperimentData with splits stored in ml namespace
        """
        # Calculate splits - pass ExperimentData directly
        cupac_splits = self.calc(data=data)
        
        # Store in ExperimentData
        return self._set_value(data, cupac_splits)
    
    def _set_value(self, data: ExperimentData, value) -> ExperimentData:
        """Store splits in ml namespace"""
        return data.set_value(
            ExperimentDataEnum.ml,
            "splits",
            value
        )
