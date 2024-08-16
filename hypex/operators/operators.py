from copy import deepcopy
from typing import Optional, Any, List, Literal, Union, Dict, Tuple

import numpy as np

from hypex.dataset import (
    Dataset,
    ABCRole,
    ExperimentData,
    TargetRole,
    InfoRole,
    FeatureRole,
    AdditionalMatchingRole,
    AdditionalTargetRole,
)
from hypex.operators.abstract import GroupOperator
from hypex.utils.enums import ExperimentDataEnum
from hypex.utils.errors import NoneArgumentError


class SMD(GroupOperator):
    def execute(self, data: ExperimentData) -> ExperimentData:
        pass

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Any:
        test_data = cls._check_test_data(test_data=test_data)
        return (data.mean() + test_data.mean()) / data.std()


class MatchingMetrics(GroupOperator):
    def __init__(
        self,
        grouping_role: Optional[ABCRole] = None,
        target_roles: Union[ABCRole, List[ABCRole], None] = None,
        metric: Optional[Literal["auto", "atc", "att", "ate"]] = None,
        key: Any = "",
    ):
        self.metric = metric or "auto"
        target_roles = target_roles or TargetRole()
        super().__init__(
            grouping_role=grouping_role,
            target_roles=(
                target_roles if isinstance(target_roles, List) else [target_roles]
            ),
            key=key,
        )

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Optional[Dataset] = None,
        target_fields: Optional[List[str]] = None,
        **kwargs,
    ) -> Any:
        if target_fields is None or test_data is None:
            raise NoneArgumentError(
                ["target_fields", "test_data"], "att, atc, ate estimation"
            )
        metric = kwargs.get("metric", "ate")
        itt = test_data[target_fields[0]] - test_data[target_fields[1]]
        itc = data[target_fields[1]] - data[target_fields[0]]
        bias = kwargs.get("bias", {})
        if len(bias) > 0:
            if metric in ["atc", "ate"]:
                itc -= Dataset.from_dict({"test": bias["control"]}, roles={})
            itt += Dataset.from_dict({"control": bias["test"]}, roles={})
        itt = itt.mean()
        itc = itc.mean()
        if metric == "atc":
            return {"ATC": itc}
        if metric == "att":
            return {"ATT": itt}
        len_test, len_control = len(data), len(test_data)
        return {
            "ATT": itt,
            "ATC": itc,
            "ATE": (itt * len_test + itc * len_control) / (len_test + len_control),
        }

    @staticmethod
    def _define_metric(
        grouping_data: List[Tuple[Union[int, str], Dataset]], target_fields
    ):
        if grouping_data[0][1][target_fields[1]].isna().sum() == len(
            grouping_data[0][1]
        ):
            return "att"
        else:
            return (
                "atc"
                if grouping_data[0][1][target_fields[1]].isna().sum()
                == len(grouping_data[1][1])
                else "ate"
            )

    @classmethod
    def _execute_inner_function(
        cls, grouping_data, target_fields: Optional[List[str]] = None, **kwargs
    ) -> Dict:
        metric = kwargs.get("metric", "auto")
        if target_fields is None or len(target_fields) != 2:
            raise ValueError(
                "This operator works with 2 targets, but got {}".format(
                    len(target_fields) if target_fields else None
                )
            )
        if metric == "auto":
            metric = cls._define_metric(grouping_data, target_fields)
        return cls._inner_function(
            data=grouping_data[0][1],
            test_data=grouping_data[1][1],
            target_fields=target_fields,
            metric=metric,
            bias=kwargs.get("bias_estimation", None),
        )

    def _prepare_new_target(self, data: ExperimentData, t_data: Dataset) -> Dataset:
        indexes = self._field_searching(data, AdditionalMatchingRole())
        if len(indexes) == 0:
            raise ValueError(f"No indexes were found")
        new_target = data.ds.search_columns(TargetRole())[0]
        indexes = data.additional_fields[indexes[0]]
        indexes.index = t_data.index
        filtered_field = indexes.drop(
            indexes[indexes[indexes.columns[0]] == -1], axis=0
        )
        matched_data = data.ds.loc[
            list(map(lambda x: x[0], filtered_field.get_values()))
        ][new_target].rename(
            {new_target: new_target + "_matched" for _ in data.ds.columns}
        )
        matched_data.index = filtered_field.index
        return matched_data

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, target_fields = self._get_fields(data=data)
        bias = (
            data.variables[data.get_one_id(Bias, ExperimentDataEnum.variables)]
            if len(
                data.get_ids(Bias, ExperimentDataEnum.variables)["Bias"]["variables"]
            )
            > 0
            else None
        )
        t_data = deepcopy(data.ds)
        if len(target_fields) != 2:
            matched_data = self._prepare_new_target(data, t_data)
            target_fields += [matched_data.search_columns(TargetRole())[0]]
            data.set_value(
                ExperimentDataEnum.additional_fields,
                self.id,
                matched_data,
                role=AdditionalTargetRole(),
            )
            t_data = t_data.add_column(
                matched_data.reindex(t_data.index),
                role={target_fields[1]: TargetRole()},
            )
        self.key = str(
            target_fields[0] if len(target_fields) == 1 else (target_fields or "")
        )
        if (
            not target_fields and data.ds.tmp_roles
        ):  # если колонка не подходит для теста, то тагет будет пустой, но если есть темп роли, то это нормальное поведение
            return data

        compare_result = self.calc(
            data=t_data,
            group_field=group_field,
            target_fields=target_fields,
            metric=self.metric,
            bias_estimation=bias,
        )
        return self._set_value(data, compare_result)


class Bias(GroupOperator):
    def __init__(
        self,
        grouping_role: Optional[ABCRole] = None,
        target_roles: Optional[List[ABCRole]] = None,
        key: Any = "",
    ):
        super().__init__(
            grouping_role=grouping_role, target_roles=target_roles, key=key
        )

    @staticmethod
    def calc_coefficients(X: Dataset, Y: Dataset) -> List[float]:
        X_l = Dataset.create_empty(roles={"temp": InfoRole()}, index=X.index).fillna(1)
        return np.linalg.lstsq(
            X_l.append(X, axis=1).data.values, Y.data.values, rcond=-1
        )[0][1:]

    @staticmethod
    def calc_bias(
        X: Dataset, X_matched: Dataset, coefficients: List[float]
    ) -> List[float]:
        return [
            (j - i).dot(coefficients)[0]
            for i, j in zip(X.data.values, X_matched.data.values)
        ]

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Optional[Dataset] = None,
        target_fields: Optional[List[str]] = None,
        features_fields: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict:
        if target_fields is None or features_fields is None or test_data is None:
            raise NoneArgumentError(
                ["target_fields", "features_fields", "test_data"], "bias_estimation"
            )
        test_result = cls.calc_bias(
            test_data[features_fields[: len(features_fields) // 2]],
            test_data[features_fields[len(features_fields) // 2 :]],
            cls.calc_coefficients(
                test_data[features_fields[len(features_fields) // 2 :]],
                test_data[target_fields[1]],
            ),
        )
        if data[target_fields[1]].isna().sum() > 0:
            return {"test": test_result}
        return {
            "test": test_result,
            "control": cls.calc_bias(
                data[features_fields[: len(features_fields) // 2]],
                data[features_fields[len(features_fields) // 2 :]],
                cls.calc_coefficients(
                    data[features_fields[len(features_fields) // 2 :]],
                    data[target_fields[1]],
                ),
            ),
        }

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_fields: Optional[List[str]] = None,
        features_fields: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict:
        return cls._inner_function(
            grouping_data[0][1],
            test_data=grouping_data[1][1],
            target_fields=target_fields,
            features_fields=features_fields,
            **kwargs,
        )

    def _prepare_data(self, data: ExperimentData, t_data: Dataset) -> Dataset:
        indexes = data.additional_fields[
            self._field_searching(data, AdditionalMatchingRole())[0]
        ]
        indexes.index = t_data.index
        filtered_field = indexes.drop(
            indexes[indexes[indexes.columns[0]] == -1], axis=0
        )
        matched_data = data.ds.loc[
            list(map(lambda x: x[0], filtered_field.get_values()))
        ].rename({i: i + "_matched" for i in data.ds.columns})
        matched_data.index = filtered_field.index
        return matched_data

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, target_fields = self._get_fields(data)
        t_data = deepcopy(data.ds)
        if len(target_fields) < 2:
            matched_data = self._prepare_data(data, t_data)
            target_fields += [matched_data.search_columns(TargetRole())[0]]
            t_data = t_data.append(matched_data.reindex(t_data.index), axis=1)
        self.key = str(
            target_fields[0] if len(target_fields) == 1 else (target_fields or "")
        )
        if (
            not target_fields and data.ds.tmp_roles
        ):  # если колонка не подходит для теста, то тагет будет пустой, но если есть темп роли, то это нормальное поведение
            return data

        compare_result = self.calc(
            data=t_data,
            group_field=group_field,
            target_fields=target_fields,
            features_fields=t_data.search_columns(
                FeatureRole(), search_types=[int, float]
            ),
        )
        return self._set_value(data, compare_result)
