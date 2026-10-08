from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..dataset.dataset import Dataset, SmallDataset
from ..dataset.experiment_data import ExperimentData, ExperimentDataEnum
from ..dataset.roles import (
    AdditionalTargetRole,
    FeatureRole,
    PreTargetRole,
    StatisticRole,
    TargetRole,
)
from ..executor import MLExecutor
from ..extensions.cupac import CupacExtension
from ..utils import ID_SPLIT_SYMBOL
from ..utils.adapter import Adapter
from ..utils.cuped_theta import cuped_theta
from ..utils.models import CUPAC_MODELS


class CUPACExecutor(MLExecutor):
    """
    Executor that applies CUPAC (Control Using Predictions As Covariates) variance reduction technique.

    CUPAC uses machine learning models to predict target values based on historical data,
    then adjusts current targets by removing the predicted variation to reduce variance.

    Args:
        cupac_models (Union[str, Sequence[str], None]): Model(s) to use for prediction.
            If None, all available models will be tried and the best one selected.
        key (Any): Unique identifier for the executor.
        n_folds (int): Number of folds for cross-validation during model selection.
        random_state (Optional[int]): Random seed for reproducibility.
    """

    def __init__(
        self,
        cupac_models: str | Sequence[str] | None = None,
        key: Any = "",
        n_folds: int = 5,
        random_state: int | None = None,
    ):
        super().__init__(target_role=TargetRole(), key=key)
        self.cupac_models = cupac_models
        self.extension = CupacExtension(n_folds, random_state)

    def _validate_models(self) -> None:
        """
        Validate that all specified CUPAC models are supported and available for the current backend.

        Raises:
            ValueError: If any model is not recognized or not available for the current backend.
        """
        wrong_models = []
        if self.cupac_models is None:
            self.cupac_models = list(CUPAC_MODELS.keys())
            return

        self.cupac_models = Adapter.to_list(self.cupac_models)

        for model in self.cupac_models:
            if model.lower() not in CUPAC_MODELS:
                wrong_models.append(model)
            elif CUPAC_MODELS[model] is None:
                raise ValueError(
                    f"Model '{model}' is not available for the current backend"
                )

        if wrong_models:
            raise ValueError(
                f"Wrong or not installed cupac models: {wrong_models}. "
                f"Available models: {list(CUPAC_MODELS.keys())}"
            )

    @staticmethod
    def _prepare_data(data: ExperimentData) -> dict[str, dict[str, list]]:
        """Prepare data for CUPAC by organizing temporal fields into training
        and prediction structures.

        This method performs complex data organization:
        1. Groups target and feature fields by their temporal lags.
        2. Identifies cofounders (features used for prediction).
        3. Structures data into X_train, Y_train for model training.
        4. Creates X_predict for current period adjustment (if applicable).

        Args:
            data: Input experiment data with temporal roles.

        Returns:
            Nested dictionary with structure::

                {target_name: {
                    'X_train': [[feature_cols_at_lag_n], ..., [feature_cols_at_lag_2]],
                    'Y_train': [target_at_lag_n-1, ..., target_at_lag_1],
                    'X_predict': [[feature_cols_at_lag_1]] (optional)
                }}
        """

        def agg_temporal_fields(role, data) -> dict[str, dict]:
            """Aggregate fields by their temporal lags.

            Returns:
                dict: ``{field_name: {lag: field_name_with_lag}}`` or
                ``{field_name: {}}``. Empty dict means lag=0 or None
                (current period).
            """
            fields: dict[str, dict] = {}
            searched_fields = data.field_search(
                (
                    [TargetRole(), PreTargetRole()]
                    if isinstance(role, TargetRole)
                    else role
                ),
                search_types=[int, float],
            )

            # ── BUGFIX: exclude synthetic columns created by a previous
            #    CUPAC run (AdditionalTargetRole inherits TargetRole,
            #    so field_search picks them up). These columns have no
            #    lag metadata and must not be treated as real targets.
            from ..dataset.roles import AdditionalRole

            searched_fields = [
                f
                for f in searched_fields
                if not isinstance(data.ds.roles.get(f), AdditionalRole)
            ]
            # ──────────────────────────────────────────────────────────

            searched_lags = [
                (
                    field,
                    (
                        data.ds.roles[field].lag
                        if not isinstance(data.ds.roles[field], TargetRole)
                        else 0
                    ),
                )
                for field in searched_fields
            ]
            sorted_fields_by_lag = sorted(searched_lags, key=lambda x: x[1])
            for field, lag in sorted_fields_by_lag:
                if lag in [None, 0]:
                    fields[field] = {}
                else:
                    if data.ds.roles[field].parent not in fields:
                        fields[data.ds.roles[field].parent] = {}
                    fields[data.ds.roles[field].parent][lag] = field
            return fields

        def agg_train_predict_x(mode: str, lag: int) -> None:
            """Aggregate features and targets for a specific lag into
            training/prediction sets.

            For each cofounder feature, accumulates lag column names into
            a single list so that all lags are vertically stacked together.
            The target (used as an autoregressive feature) is accumulated
            in the same manner at a fixed position after all cofounders.

            Args:
                mode: Target accumulator key (``"X_train"`` or
                    ``"X_predict"``).
                lag: Current lag being processed.
            """
            for i, cofounder in enumerate(cofounders[target]):
                if lag in [1, max_lags[target]]:
                    # First or last lag → create a new entry for this feature.
                    cupac_data[target][mode].append([features[cofounder][lag]])
                else:
                    # Intermediate lag → append to the existing entry.
                    cupac_data[target][mode][i].append(features[cofounder][lag])

            # ── Target as autoregressive feature ──────────────────────
            # The target entry sits at a fixed index right after all
            # cofounders.  We create it on the FIRST call for a given
            # mode (detected by list length) and append on subsequent
            # calls.  This works for both X_train (first call at
            # lag=max_lags) and X_predict (single call at lag=1).
            target_idx = len(cofounders[target])
            if target_idx >= len(cupac_data[target][mode]):
                # Entry does not exist yet → create it.
                cupac_data[target][mode].append([targets[target][lag]])
            else:
                # Entry already exists → append the new lag column.
                cupac_data[target][mode][target_idx].append(targets[target][lag])

        cupac_data: dict[str, dict[str, list]] = {}
        targets = agg_temporal_fields(TargetRole(), data)
        features = agg_temporal_fields(FeatureRole(), data)

        # Determine cofounders (features used for prediction) for each target
        cofounders: dict[str, list[str]] = {}
        for target in targets:
            if target in data.ds.columns:
                cofounders[target] = data.ds.roles[target].cofounders
            else:
                # For virtual targets, get cofounders from the earliest lag
                min_lag = min(targets[target].keys())
                cofounders[target] = data.ds.roles[targets[target][min_lag]].cofounders
            if cofounders[target] is None:
                raise ValueError(
                    f"Cofounders must be defined in the first lag for "
                    f"virtual target '{target}'"
                )

        # Calculate maximum lag for each target
        max_lags: dict[str, int] = {}
        for target, lags in targets.items():
            if not lags:
                raise ValueError(
                    f"Target '{target}' has no lag periods defined. "
                    f"CUPAC requires at least one historical period. "
                    f"Assign PreTargetRole(lag=N) to historical columns "
                    f"of this target."
                )
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
            for lag in range(max_lags[target], 1, -1):
                agg_train_predict_x("X_train", lag)
                cupac_data[target]["Y_train"].append(targets[target][lag - 1])

            # Build prediction data for current period (lag=1)
            if "X_predict" in cupac_data[target].keys():
                agg_train_predict_x("X_predict", 1)

        return cupac_data

    @classmethod
    def _execute_inner_function(cls) -> None:
        pass

    @classmethod
    def _inner_function(cls) -> None:
        pass

    def calc(
        self, mode: str, model: str | Any, X: Dataset, Y: Dataset | None = None
    ) -> Any:
        if mode == "kfold_fit":
            return self.kfold_fit(model, X, Y)
        elif mode == "fit":
            return self.fit(model, X, Y)
        elif mode == "predict":
            return self.predict(model, X)
        return None

    def kfold_fit(
        self, model: str, X: Dataset, Y: Dataset
    ) -> tuple[float, dict[str, float]]:
        """Run k-fold cross-validation and return variance reduction and feature importances."""
        var_red, feature_importances = self.extension.calc(
            data=X,
            mode="kfold_fit",
            model=model,
            Y=Y,
        )

        return var_red, feature_importances

    def fit(self, model: str, X: Dataset, Y: Dataset) -> Any:
        return self.extension.calc(
            data=X,
            mode="fit",
            model=model,
            Y=Y,
        )

    def predict(self, model: Any, X: Dataset) -> Dataset:
        return self.extension.calc(
            data=X,
            mode="predict",
            model=model,
        )

    @staticmethod
    def _agg_data_from_cupac_data(
        data: ExperimentData, cupac_data_slice: list
    ) -> Dataset:
        res_dataset = None
        column_counter = 0
        for column in cupac_data_slice:
            if len(column) == 1:
                col_data = data.ds[column[0]]
            else:
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

    def execute(self, data: ExperimentData) -> ExperimentData:
        """Execute CUPAC variance reduction on the experiment data.

        Process:
        1. Validate models and prepare temporal data structures.
        2. For each target:
        a. Try all specified models with cross-validation.
        b. Select the model with best variance reduction.
        c. Fit the best model on all training data.
        d. Predict and adjust current target values.
        e. Store adjusted target and variance-reduction metrics.

        Args:
            data: Input data with temporal features and targets.

        Returns:
            Data with CUPAC-adjusted targets and variance reduction reports.
        """
        self._validate_models()

        # ── BUGFIX: remove CUPAC columns from a previous run on the
        #    same Dataset object.  CUPACExecutor is not a transformer,
        #    so Experiment.execute() does NOT deepcopy the data.
        #    Without this cleanup, re-running .execute(data) raises
        #    "Columns with the same name already exist".
        existing_cupac_cols = [
            col
            for col in data.ds.columns
            if col.endswith("_cupac")
            and isinstance(data.ds.roles.get(col), AdditionalTargetRole)
        ]
        if existing_cupac_cols:
            data = data.copy(data=data.ds.drop(columns=existing_cupac_cols))
        # ──────────────────────────────────────────────────────────────

        cupac_data = self._prepare_data(data)

        for target, target_data in cupac_data.items():
            X_train_feature_names = [column[0] for column in target_data["X_train"]]
            X_train = self._agg_data_from_cupac_data(data, target_data["X_train"])
            Y_train = self._agg_data_from_cupac_data(data, [target_data["Y_train"]])

            best_model: str | None = None
            best_var_red: float | None = None
            best_feature_importances: dict[str, float] | None = None

            for model in self.cupac_models:
                var_red, fold_importances = self.calc(
                    mode="kfold_fit",
                    model=model,
                    X=X_train,
                    Y=Y_train,
                )
                if best_var_red is None or var_red > best_var_red:
                    best_model, best_var_red = model, var_red
                    best_feature_importances = {
                        X_train_feature_names[int(col_idx)]: importance
                        for col_idx, importance in fold_importances.items()
                    }

            if best_model is None:
                raise RuntimeError(
                    f"No models were successfully fitted for target '{target}'."
                )

            cupac_variance_reduction_real: float | None = None

            if "X_predict" in target_data:
                fitted_model = self.calc(
                    mode="fit",
                    model=best_model,
                    X=X_train,
                    Y=Y_train,
                )
                X_predict = self._agg_data_from_cupac_data(
                    data,
                    target_data["X_predict"],
                )
                prediction = self.calc(mode="predict", model=fitted_model, X=X_predict)

                theta = cuped_theta(
                    data.ds[target].raw_data.values.flatten(),
                    prediction.raw_data.values.flatten(),
                )
                explained_variation = (prediction - prediction.mean()) * theta
                target_cupac = data.ds[target] - explained_variation
                target_cupac = target_cupac.rename({target: f"{target}_cupac"})

                data = data.set_value(
                    space=ExperimentDataEnum.additional_fields,
                    executor_id=f"{target}_cupac",
                    value=target_cupac,
                    role=AdditionalTargetRole(),
                )

                cupac_variance_reduction_real = (
                    self.extension._calculate_variance_reduction(
                        data.ds[target],
                        target_cupac,
                    )
                )

            # ✅ FIX: store report as SmallDataset via set_value
            report: dict[str, Any] = {
                "cupac_best_model": best_model,
                "cupac_variance_reduction_cv": best_var_red,
                "cupac_variance_reduction_real": cupac_variance_reduction_real,
            }
            report_ds = SmallDataset.from_dict(
                [report],
                roles={
                    "cupac_best_model": StatisticRole(),
                    "cupac_variance_reduction_cv": StatisticRole(float),
                    "cupac_variance_reduction_real": StatisticRole(float),
                },
            )
            data = data.set_value(
                ExperimentDataEnum.analysis_tables,
                f"{self.id}{ID_SPLIT_SYMBOL}{target}",
                report_ds,
            )

            if best_feature_importances:
                imp_ds = SmallDataset.from_dict(
                    [best_feature_importances],
                    roles={k: StatisticRole(float) for k in best_feature_importances},
                )
                data = data.set_value(
                    ExperimentDataEnum.analysis_tables,
                    f"{self.id}{ID_SPLIT_SYMBOL}{target}{ID_SPLIT_SYMBOL}importances",
                    imp_ds,
                )

        return data
