"""Base classes for data splitters in HypEx."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from ..dataset import Dataset, ExperimentData
from ..dataset.ml_data import MLData, MLExperimentData
from ..dataset.roles import PreTargetRole, TargetRole
from ..executor import Executor
from ..utils.enums import MLMode


class Splitter(Executor, ABC):
    """
    Base class for general splitters.

    Splitter contains only generic execution contract and common splitter params.

    Subclasses should implement execute(): main splitting logic.
    """
    
    def __init__(
        self,
        key: Any = "",
    ):
        """
        Initialize Splitter.
        
        Args:
            key: Executor identifier
        """
        super().__init__(key=key)
    
    @abstractmethod
    def execute(self, data: ExperimentData) -> ExperimentData:
        """
        Execute data splitting.
        
        Args:
            data: Input experiment data
            
        Returns:
            ExperimentData with prepared data
        """
        raise NotImplementedError


class MLSplitter(Splitter, ABC):
    """
    Base class for ML splitters used in MLExperiment.

    MLSplitter operates on MLExperimentData only and provides ML-specific helpers:
    - CV fold generation using Dataset API
    - Temporal field aggregation by lag
    - Column aggregation for temporal/lagged ML inputs
    - MLData construction from raw split specs
    """

    def __init__(
        self,
        n_folds: int = 5,
        random_state: Optional[int] = None,
        key: Any = "",
    ):
        """
        Initialize MLSplitter.

        Args:
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            key: Executor identifier
        """
        super().__init__(key=key)
        self.n_folds = n_folds
        self.random_state = random_state

    @abstractmethod
    def execute(
        self,
        data: MLExperimentData,
        mode: MLMode | None = None,
    ) -> MLExperimentData:
        """
        Execute ML data splitting.

        Args:
            data: ML experiment data
            mode: ML execution mode

        Returns:
            MLExperimentData with prepared ML inputs
        """
        raise NotImplementedError
    
    def create_cv_folds(
        self,
        X: Dataset,
        Y: Dataset,
        n_folds: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> Dict[int, Tuple[Dataset, Dataset]]:
        """
        Create cross-validation folds for given data.
        
        Splits data into n_folds train/validation sets using Dataset API only.
        
        Args:
            X: Features dataset
            Y: Target dataset
            n_folds: Number of folds (uses self.n_folds if None)
            random_state: Random seed (uses self.random_state if None)
            
        Returns:
            Dict mapping fold_id to (X_val, Y_val) tuples
        """
        n_folds = n_folds or self.n_folds
        random_state = random_state or self.random_state
        
        # Shuffle data if random_state is provided, using Dataset.sample
        if random_state is not None:
            # Sample 100% of data with random_state to shuffle
            X = X.sample(frac=1.0, random_state=random_state)
            # Ensure Y has the same order as X by sampling with same indices
            Y = Y.iloc[X.data.index.tolist()]
        
        # Get number of samples
        n_samples = X.shape[0]
        
        # Calculate fold size
        fold_size = n_samples // n_folds
        remainder = n_samples % n_folds
        
        # Generate folds using sequential slicing of shuffled data
        cv_folds = {}
        start_idx = 0
        
        for fold_id in range(n_folds):
            # Calculate end index for this fold (distribute remainder across first folds)
            current_fold_size = fold_size + (1 if fold_id < remainder else 0)
            end_idx = start_idx + current_fold_size
            
            # Get validation indices for this fold (using sequential indices on shuffled data)
            val_indices = list(range(start_idx, end_idx))
            
            # Extract validation data using Dataset.iloc
            X_val = X.iloc[val_indices]
            Y_val = Y.iloc[val_indices]
            
            cv_folds[fold_id] = (X_val, Y_val)
            
            start_idx = end_idx
        
        return cv_folds
    
    def add_cv_folds_to_mldata(
        self,
        ml_data: MLData,
        n_folds: Optional[int] = None,
        random_state: Optional[int] = None,
    ) -> MLData:
        """
        Add cross-validation folds to existing MLData.
        
        Args:
            ml_data: MLData object to add folds to
            n_folds: Number of folds (uses self.n_folds if None)
            random_state: Random seed (uses self.random_state if None)
            
        Returns:
            MLData with populated crossval field
        """
        ml_data.crossval = self.create_cv_folds(
            ml_data.X_train,
            ml_data.Y_train,
            n_folds=n_folds,
            random_state=random_state,
        )
        return ml_data

    @staticmethod
    def aggregate_fields_by_lag(
        data: ExperimentData,
        role,
        include_pretarget_for_target: bool = True,
    ) -> dict[str, dict[int, str]]:
        """
        Aggregate fields by temporal lag.

        Returns mapping:
        - current period fields as {field_name: {}}
        - lagged fields grouped by parent: {parent_name: {lag: field_name}}
        """
        fields: dict[str, dict[int, str]] = {}
        search_role = (
            [TargetRole(), PreTargetRole()]
            if include_pretarget_for_target and isinstance(role, TargetRole)
            else role
        )

        searched_fields = data.field_search(search_role, search_types=[int, float])
        searched_lags = [
            (
                field,
                (data.ds.roles[field].lag if not isinstance(data.ds.roles[field], TargetRole) else 0),
            )
            for field in searched_fields
        ]

        sorted_fields_by_lag = sorted(
            searched_lags,
            key=lambda x: x[1] if x[1] is not None else 0,
        )

        for field, lag in sorted_fields_by_lag:
            if lag in [None, 0]:
                fields[field] = {}
                continue

            parent = data.ds.roles[field].parent
            if parent not in fields:
                fields[parent] = {}
            fields[parent][lag] = field

        return fields

    def build_mldata_from_splits(
        self,
        data: MLExperimentData,
        raw_splits: dict[str, dict[str, list]],
        generate_cv_folds: bool = False,
    ) -> MLExperimentData:
        """
        Build and store MLData objects from raw split specification.

        raw_splits format:
            {
                target_name: {
                    "X_train": [...],
                    "Y_train": [...],
                    "X_predict": [...],  # optional
                }
            }
        """
        for target_name, target_splits in raw_splits.items():
            X_train = self.aggregate_columns(data, target_splits["X_train"])
            Y_train = self.aggregate_columns(data, [target_splits["Y_train"]])

            X_predict = None
            if "X_predict" in target_splits:
                X_predict = self.aggregate_columns(data, target_splits["X_predict"])

            ml_data_obj = MLData(X_train=X_train, Y_train=Y_train, X_predict=X_predict)

            if generate_cv_folds:
                ml_data_obj = self.add_cv_folds_to_mldata(
                    ml_data_obj,
                    n_folds=self.n_folds,
                    random_state=self.random_state,
                )

            data.add_ml_data(target_name, ml_data_obj)

        return data
    
    @staticmethod
    def aggregate_columns(
        data: ExperimentData,
        column_spec: list,
    ) -> Dataset:
        """
        Aggregate columns from column specification into a single Dataset.
        
        This method handles two types of column structures:
        1. Single column: [column_name] - directly extracted
        2. Multiple lag columns: [col_lag1, col_lag2, ...] - vertically stacked
        
        Args:
            data: Original ExperimentData with all columns
            column_spec: List of column specifications, where each element is:
                - [single_col_name] for non-temporal columns
                - [col_name_lag1, col_name_lag2, ...] for temporal sequences
                
        Returns:
            Dataset with standardized column names (0, 1, 2, ...)
        """
        res_dataset = None
        column_counter = 0
        
        for column in column_spec:
            if len(column) == 1:
                # Single column case: extract directly
                col_data = data.ds[column[0]]
            else:
                # Multiple lag columns: stack them vertically
                res_lag_column = None
                for lag_column in column:
                    tmp_dataset = data.ds[lag_column]
                    tmp_dataset = tmp_dataset.rename({lag_column: column[0]})
                    if res_lag_column is None:
                        res_lag_column = tmp_dataset
                    else:
                        res_lag_column = res_lag_column.append(
                            tmp_dataset, reset_index=True, axis=0
                        )
                col_data = res_lag_column
            
            # Standardize column names to numeric format for model training
            standard_col_name = f"{column_counter}"
            col_data = col_data.rename(
                {next(iter(col_data.columns)): standard_col_name}
            )
            column_counter += 1
            
            if res_dataset is None:
                res_dataset = col_data
            else:
                res_dataset = res_dataset.add_column(data=col_data)
        
        return res_dataset
