from typing import Any, Optional
import numpy as np
from copy import deepcopy
from ..dataset.dataset import Dataset, ExperimentData
from ..dataset.roles import TargetRole, PreTargetRole, StatisticRole, FeatureRole, AdditionalTargetRole
from ..executor import MLExecutor
from ..utils import ExperimentDataEnum
from ..utils.enums import BackendsEnum

from ..extensions.cupac import CupacExtension

from typing import Union, Sequence
from ..utils.models import CUPAC_MODELS

class CUPACExecutor(MLExecutor):
    def __init__(
        self,
        cupac_models: Union[str, Sequence[str], None] = None,
        key: Any = "",
        n_folds: int = 5,
        random_state: Optional[int] = None,
    ):
        super().__init__(target_role=TargetRole(), key=key)
        self.cupac_models = cupac_models
        self.extension = CupacExtension(n_folds, random_state)

    def _validate_models(self) -> None:
        wrong_models = []
        if self.cupac_models is None:
            self.cupac_models = list(CUPAC_MODELS.keys())
            return
        
        if isinstance(self.cupac_models, str):
            self.cupac_models = [self.cupac_models]
        for model in self.cupac_models:
            if model.lower() not in CUPAC_MODELS:
                wrong_models.append(model)
            elif CUPAC_MODELS[model] is None:
                raise ValueError(f"Model '{model}' is not available for the current backend")
        
        if wrong_models:
            raise ValueError(f"Wrong cupac models: {wrong_models}. Available models: {list(CUPAC_MODELS.keys())}")

    @staticmethod
    def _prepare_data(data: ExperimentData):

        def agg_temporal_fields(role, data):
            fields = {}
            searched_fields = data.field_search([TargetRole(), PreTargetRole()] if isinstance(role, TargetRole) else role, search_types=[int, float])

            searched_lags = [(field, data.ds.roles[field].lag if not isinstance(data.ds.roles[field], TargetRole) else 0) for field in searched_fields]
            sorted_fields_by_lag = sorted(searched_lags, key=lambda x: x[1])
            for field, lag in sorted_fields_by_lag:
                if lag in [None , 0]:
                    fields[field] = {}
                else:
                    if data.ds.roles[field].parent in fields:
                        fields[data.ds.roles[field].parent][lag] = field
                    else:
                        fields[data.ds.roles[field].parent] = {}
                        fields[data.ds.roles[field].parent][lag] = field
            return fields

        def agg_train_predict_x(mode, lag):
            for i, cofounder in enumerate(cofounders[target]):
                if lag in [1, max_lags[target]]:
                    cupac_data[target][mode].append([features[cofounder][lag]])
                else:
                    cupac_data[target][mode][i].append(cofounder)

            cupac_data[target][mode].append([targets[target][lag]])

        cupac_data = {}
        targets = agg_temporal_fields(TargetRole(), data)
        features = agg_temporal_fields(FeatureRole(), data)
        
        cofounders = {}
        for target in targets:
            if target in data.ds.columns:
                cofounders[target] = data.ds.roles[target].cofounders
            else:
                min_lag = min(targets[target].keys())
                cofounders[target] = data.ds.roles[targets[target][min_lag]].cofounders

                if cofounders[target] is None:
                    raise ValueError(f"Cofounders must be defined in the first lag for virtual target '{target}'")                    

        max_lags = {}
        for target, lags in targets.items():
            if lags:
                max_lag = max(lags.keys())
                for feature in cofounders[target]:
                    if feature in features and features[feature]:
                        max_lag = max(max(features[feature].keys()), max_lag)
            max_lags[target] = max_lag
        
        for target in targets.keys():

            cupac_data[target] = {
                'X_train': [],
                'Y_train': []
            }
            if target in data.ds.columns:
                cupac_data[target]['X_predict'] = []

            for lag in range(max_lags[target], 1, -1):
                agg_train_predict_x('X_train', lag)
                cupac_data[target]['Y_train'].append(targets[target][lag - 1])

            if 'X_predict' in cupac_data[target].keys():
                agg_train_predict_x('X_predict', 1)

        return cupac_data

    @classmethod
    def _execute_inner_function():
        pass

    @classmethod
    def _inner_function():
        pass

    def calc(
        self,
        mode,
        model,
        X,
        Y=None
    ):
        if mode == 'kfold_fit':
            return self.kfold_fit(model, X, Y)
        elif mode == 'fit':
            return self.fit(model, X, Y)
        elif mode == 'predict':
            return self.predict(model, X)

    def kfold_fit(self, model, X, Y):
        var_red = self.extension.calc(
            data=X,
            mode='kfold_fit',
            model=model,
            Y=Y,
        )

        return var_red
    
    def fit(self, model, X, Y):
        return self.extension.calc(
            data=X,
            mode='fit',
            model=model,
            Y=Y,
        )

    def predict(self, model, X):
        return self.extension.calc(
            data=X,
            mode='predict',
            model=model,
        )

    @staticmethod
    def _agg_data_from_cupac_data(data, cupac_data_slice):
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
                        res_lag_column = res_lag_column.append(tmp_dataset, reset_index=True, axis=0)
                col_data = res_lag_column
            
            standard_col_name = f"{column_counter}"
            col_data = col_data.rename({list(col_data.columns)[0]: standard_col_name})
            column_counter += 1
            
            if res_dataset is None:
                res_dataset = col_data
            else:
                res_dataset = res_dataset.add_column(data=col_data)
        return res_dataset


    def execute(self, data: ExperimentData) -> ExperimentData:
        self._validate_models()
        cupac_data = CUPACExecutor._prepare_data(data)
        for target in cupac_data.keys():
            X_train = CUPACExecutor._agg_data_from_cupac_data(
                data,
                cupac_data[target]['X_train']
            )
            Y_train = CUPACExecutor._agg_data_from_cupac_data(
                data,
                [cupac_data[target]['Y_train']]
            )
            best_model, best_var_red = None, None

            for model in self.cupac_models:
                var_red = self.calc(
                    mode='kfold_fit',
                    model=model,
                    X=X_train,
                    Y=Y_train
                    )
                if best_var_red is None or var_red > best_var_red:
                    best_model, best_var_red = model, var_red

            if best_model is None:
                raise RuntimeError(f"No models were successfully fitted for target '{target}'. All models failed during training.")

            cupac_variance_reduction_real = None

            if 'X_predict' in cupac_data[target]:
                X_predict = CUPACExecutor._agg_data_from_cupac_data(
                    data,
                    cupac_data[target]['X_predict']
                )

                fitted_model = self.calc(
                    mode='fit',
                    model=best_model,
                    X=X_train,
                    Y=Y_train
                )

                prediction = self.calc(
                    mode='predict',
                    model=fitted_model,
                    X=X_predict
                )

                explained_variation = prediction - prediction.mean()
                target_cupac = data.ds[target] - explained_variation

                target_cupac = target_cupac.rename({target: f"{target}_cupac"})
                data.additional_fields = data.additional_fields.add_column(
                    data=target_cupac,
                    role={f"{target}_cupac": AdditionalTargetRole()}
                )
                cupac_variance_reduction_real = self.extension._calculate_variance_reduction(data.ds[target], target_cupac)

            report = {
                "cupac_best_model": best_model,
                "cupac_variance_reduction_cv": best_var_red,
                "cupac_variance_reduction_real": cupac_variance_reduction_real,
            }
            data.analysis_tables[f"{target}_cupac_report"] = report

        return data