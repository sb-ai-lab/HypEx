from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from ..dataset import Dataset
from ..dataset.ml_data import MLExperimentData
from ..executor.ml_executor import MLExecutor
from ..executor.state import MLExecutorParams
from ..extensions.sklearn import MutualInfoExtension


class FeatureSelectionExecutor(MLExecutor):
    """
    Select top features before model training.

    This executor filters features in MLData (X_train, X_predict) based on
    various selection methods. It runs after data splitting but before
    model selection.

    Supports three execution modes (set by MLExperiment):
    - fit: Calculate feature scores and select features, save selection
    - predict: Apply saved feature selection
    - fit_predict: Calculate and apply selection (default)

    Args:
        method: Feature selection method:
            - "importance": Train a model and select by feature importance
            - "variance": Remove low-variance features
            - "correlation": Remove highly correlated features
            - "mutual_info": Select by mutual information with target
        n_features: Number of features to select:
            - int: Exact number of top features
            - float (0.0-1.0): Fraction of features to keep
            - None: Use threshold instead
        threshold: Minimum score threshold (used if n_features is None):
            - For "importance": minimum importance value
            - For "variance": minimum variance
            - For "correlation": maximum correlation to remove
            - For "mutual_info": minimum mutual info score
        importance_model: Model type for importance-based selection.
            Options: "catboost", "ridge", "linear". Default: "ridge"
        key: Unique identifier for the executor.

    Example:
        >>> # Select top 10 features by importance
        >>> selector = FeatureSelectionExecutor(
        ...     method="importance",
        ...     n_features=10,
        ... )

        >>> # Keep features with variance > 0.01
        >>> selector = FeatureSelectionExecutor(
        ...     method="variance",
        ...     threshold=0.01,
        ... )

        >>> # Remove features with correlation > 0.95
        >>> selector = FeatureSelectionExecutor(
        ...     method="correlation",
        ...     threshold=0.95,
        ... )
    """

    SUPPORTED_METHODS = ("importance", "variance", "correlation", "mutual_info")

    def __init__(
        self,
        method: Literal["importance", "variance", "correlation", "mutual_info"] = "importance",
        n_features: Optional[Union[int, float]] = None,
        threshold: Optional[float] = None,
        importance_model: Literal["catboost", "ridge", "linear"] = "ridge",
        key: Any = "",
    ):
        super().__init__(key=key)

        if method not in self.SUPPORTED_METHODS:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Supported: {self.SUPPORTED_METHODS}"
            )

        if n_features is None and threshold is None:
            # Default: keep top 50% features
            n_features = 0.5

        self.method = method
        self.n_features = n_features
        self.threshold = threshold
        self.importance_model = importance_model

        # Fitted state per target
        self._selected_features: Dict[str, List[str]] = {}

    def execute_fit(self, data: MLExperimentData) -> MLExperimentData:
        """Calculate feature scores and select features for each target."""
        for target_name in data.get_all_targets():
            ml_data_obj = data.get_ml_data(target_name)

            # Calculate scores and select features
            selected = self._select_features(
                X=ml_data_obj.X_train,
                y=ml_data_obj.Y_train,
            )
            self._selected_features[target_name] = selected

        # Save state for predict mode
        state = MLExecutorParams(
            executor_id=self.id,
            executor_class=self.__class__.__name__,
            fitted_params={"selected_features": self._selected_features},
            metadata={
                "method": self.method,
                "n_features": self.n_features,
                "threshold": self.threshold,
            },
        )
        data.add_fitted_ml_executor(self.id, state)

        return data

    def execute_predict(self, data: MLExperimentData) -> MLExperimentData:
        """Apply saved feature selection to data."""
        # Load state if not already loaded
        if not self._selected_features:
            state = data.get_fitted_ml_executor(self.id)
            if state is None:
                raise ValueError(
                    f"FeatureSelectionExecutor (id={self.id}) is in 'predict' mode "
                    "but no fitted state found. Run FIT mode first."
                )
            self._selected_features = state.fitted_params["selected_features"]

        # Apply selection to each target's MLData
        for target_name in data.get_all_targets():
            if target_name not in self._selected_features:
                raise ValueError(
                    f"No feature selection found for target '{target_name}'. "
                    "Make sure FIT was run with this target."
                )

            selected = self._selected_features[target_name]
            ml_data_obj = data.get_ml_data(target_name)

            # Filter X_train
            ml_data_obj.X_train = ml_data_obj.X_train[selected]

            # Filter X_predict if exists
            if ml_data_obj.X_predict is not None:
                ml_data_obj.X_predict = ml_data_obj.X_predict[selected]

        return data

    def _select_features(self, X: Dataset, y: Dataset) -> List[str]:
        """
        Select features based on configured method.

        Returns:
            List of selected feature names
        """
        feature_names = list(X.columns)

        if self.method == "importance":
            scores = self._importance_scores(X, y)
        elif self.method == "variance":
            scores = self._variance_scores(X)
        elif self.method == "correlation":
            return self._correlation_filter(X)
        elif self.method == "mutual_info":
            scores = self._mutual_info_scores(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Select by n_features or threshold
        return self._select_by_scores(feature_names, scores)

    def _select_by_scores(
        self, feature_names: List[str], scores: Dict[str, float]
    ) -> List[str]:
        """Select features based on scores."""
        # Sort by score descending
        sorted_features = sorted(
            scores.items(), key=lambda x: x[1], reverse=True
        )

        if self.n_features is not None:
            # Select by count/fraction
            if isinstance(self.n_features, float) and 0.0 < self.n_features <= 1.0:
                n_select = max(1, int(len(feature_names) * self.n_features))
            else:
                n_select = min(int(self.n_features), len(feature_names))

            selected = [f for f, _ in sorted_features[:n_select]]
        else:
            # Select by threshold
            selected = [f for f, s in sorted_features if s >= self.threshold]
            # Keep at least one feature
            if not selected:
                selected = [sorted_features[0][0]]

        return selected

    def _importance_scores(self, X: Dataset, y: Dataset) -> Dict[str, float]:
        """Calculate feature importance using a model."""
        from .models import MLModel

        model = MLModel.create(self.importance_model)
        model.fit(X, y)

        return model.get_feature_importances()

    def _variance_scores(self, X: Dataset) -> Dict[str, float]:
        """Calculate variance for each feature using Dataset API."""
        scores = {}
        # Use Dataset.var() method
        var_result = X.var()

        # var() returns Dataset with variance values
        for col in X.columns:
            # Get variance value for each column
            var_value = var_result.get_values(column=col)
            if hasattr(var_value, "__iter__") and not isinstance(var_value, str):
                scores[col] = float(var_value[0])
            else:
                scores[col] = float(var_value)

        return scores

    def _correlation_filter(self, X: Dataset) -> List[str]:
        """
        Remove highly correlated features using Dataset API.

        Keeps one feature from each group of correlated features.
        threshold controls maximum allowed correlation.
        """
        threshold = self.threshold if self.threshold is not None else 0.95

        feature_names = list(X.columns)

        # Use Dataset.corr() method
        corr_ds = X.corr()

        # Find features to drop based on correlation
        to_drop = set()

        for i, col_i in enumerate(feature_names):
            if col_i in to_drop:
                continue
            for j, col_j in enumerate(feature_names):
                if j <= i or col_j in to_drop:
                    continue

                # Get correlation value
                corr_value = corr_ds.get_values(column=col_j)
                if hasattr(corr_value, "__getitem__"):
                    corr_val = abs(float(corr_value[i]))
                else:
                    corr_val = abs(float(corr_value))

                if corr_val > threshold:
                    # Drop feature with lower variance
                    var_i = self._get_single_variance(X, col_i)
                    var_j = self._get_single_variance(X, col_j)

                    if var_i >= var_j:
                        to_drop.add(col_j)
                    else:
                        to_drop.add(col_i)
                        break

        # Select remaining features
        selected = [f for f in feature_names if f not in to_drop]

        # Apply n_features limit if specified
        if self.n_features is not None:
            if isinstance(self.n_features, float) and 0.0 < self.n_features <= 1.0:
                n_select = max(1, int(len(feature_names) * self.n_features))
            else:
                n_select = min(int(self.n_features), len(selected))
            selected = selected[:n_select]

        return selected if selected else [feature_names[0]]

    def _get_single_variance(self, X: Dataset, col: str) -> float:
        """Get variance for a single column."""
        col_ds = X[col]
        var_result = col_ds.var()
        var_value = var_result.get_values(column=col)
        if hasattr(var_value, "__iter__") and not isinstance(var_value, str):
            return float(var_value[0])
        return float(var_value)

    def _mutual_info_scores(self, X: Dataset, y: Dataset) -> Dict[str, float]:
        """Calculate mutual information with target using extension."""
        mi_ext = MutualInfoExtension()
        result_ds = mi_ext.calc(X, target=y)

        # Extract scores from result Dataset
        features = result_ds.get_values(column="feature")
        scores_values = result_ds.get_values(column="score")

        return dict(zip(features, scores_values))
