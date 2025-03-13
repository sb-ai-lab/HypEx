from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

import numpy as np

from ..dataset import (
    ABCRole,
    AdditionalMatchingRole,
    AdditionalTargetRole,
    Dataset,
    ExperimentData,
    FeatureRole,
    InfoRole,
    TargetRole,
)
from ..extensions.scipy_stats import NormCDF
from ..utils.enums import ExperimentDataEnum
from ..utils.errors import NoneArgumentError
from .abstract import GroupOperator


class SMD(GroupOperator):
    def execute(self, data: ExperimentData) -> ExperimentData:
        pass

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Dataset | None = None, **kwargs
    ) -> Any:
        test_data = cls._check_test_data(test_data=test_data)
        return (data.mean() + test_data.mean()) / data.std()


class MatchingMetrics(GroupOperator):
    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        target_roles: ABCRole | list[ABCRole] | None = None,
        metric: Literal["auto", "atc", "att", "ate"] | None = None,
        key: Any = "",
    ):
        self.metric = metric or "auto"
        self.__scaled_counts = {}
        target_roles = target_roles or TargetRole()
        super().__init__(
            grouping_role=grouping_role,
            target_roles=(
                target_roles if isinstance(target_roles, list) else [target_roles]
            ),
            key=key,
        )

    def _calc_scaled_counts(self, matches, group):
        s_counts = [x[0] for x in matches.value_counts()["count"].get_values()]
        extra_counts = [0 for _ in range(len(matches) - len(s_counts))]
        self.__scaled_counts[group] = s_counts + extra_counts

    @staticmethod
    def _calc_vars(value):
        var = 0 if value[value.columns[0]].isna().sum() > 0 else value.var()
        return [var for _ in range(len(value))]

    @staticmethod
    def _calc_se(var_c, var_t, scaled_counts, is_ate=False):
        n_c, n_t = len(var_c), len(var_t)
        if not is_ate:
            weights_c = n_c / n_t * np.array(scaled_counts)
            weights_t = np.ones(n_t)
        else:
            n = n_c + n_t
            weights_c = (n_c / n) * np.array(scaled_counts["control"])
            weights_t = (n_t / n) * np.array(scaled_counts["test"])

        return np.sqrt(
            (weights_t**2 * var_t).sum() / n_t**2
            + (weights_c**2 * var_c).sum() / n_c**2
        )

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Dataset | None = None,
        target_fields: list[str] | None = None,
        **kwargs,
    ) -> Any:
        if target_fields is None or test_data is None:
            raise NoneArgumentError(
                ["target_fields", "test_data"], "att, atc, ate estimation"
            )
        metric = kwargs.get("metric", "ate")
        scaled_counts = kwargs.get("scaled_counts")
        itt = test_data[target_fields[0]] - test_data[target_fields[1]]
        itc = data[target_fields[1]] - data[target_fields[0]]
        bias = kwargs.get("bias", {})
        if bias and len(bias) > 0:
            if metric in ["atc", "ate"]:
                itc -= Dataset.from_dict(
                    {"test": bias["control"]}, roles={}, index=itc.index
                )
            if metric in ["att", "ate"]:
                itt += Dataset.from_dict(
                    {"control": bias["test"]}, roles={}, index=itt.index
                )
        var_t = cls._calc_vars(itt)
        var_c = cls._calc_vars(itc)
        itt_se = cls._calc_se(var_c, var_t, scaled_counts["control"])
        itc_se = cls._calc_se(var_t, var_c, scaled_counts["test"])
        itt = itt.mean()
        itc = itc.mean()
        p_val_itt = (
            NormCDF()
            .calc(
                Dataset.from_dict(
                    {"value": [itt / itt_se]}, roles={"value": InfoRole()}
                )
            )
            .get_values()[0][0]
        )
        p_val_itc = (
            NormCDF()
            .calc(
                Dataset.from_dict(
                    {"value": [itc / itc_se]}, roles={"value": InfoRole()}
                )
            )
            .get_values()[0][0]
        )
        if metric == "atc":
            return {
                "ATC": [
                    itc,
                    itc_se,
                    p_val_itc,
                    itc - 1.96 * itc_se,
                    itc + 1.96 * itc_se,
                ]
            }
        if metric == "att":
            return {
                "ATT": [
                    itt,
                    itt_se,
                    p_val_itt,
                    itt - 1.96 * itt_se,
                    itt + 1.96 * itt_se,
                ]
            }
        len_test, len_control = len(data), len(test_data)
        ate = (itt * len_test + itc * len_control) / (len_test + len_control)
        ate_se = cls._calc_se(var_c, var_t, scaled_counts, is_ate=True)
        p_val_ate = (
            NormCDF()
            .calc(
                Dataset.from_dict(
                    {"value": [ate / ate_se]}, roles={"value": InfoRole()}
                )
            )
            .get_values()[0][0]
        )
        return {
            "ATT": [itt, itt_se, p_val_itt, itt - 1.96 * itt_se, itt + 1.96 * itt_se],
            "ATC": [itc, itc_se, p_val_itc, itc - 1.96 * itc_se, itc + 1.96 * itc_se],
            "ATE": [ate, ate_se, p_val_ate, ate - 1.96 * ate_se, ate + 1.96 * ate_se],
        }

    @classmethod
    def _execute_inner_function(
        cls, grouping_data, target_fields: list[str] | None = None, **kwargs
    ) -> dict:
        metric = kwargs.get("metric", "ate")
        if target_fields is None or len(target_fields) != 2:
            raise ValueError(
                f"This operator works with 2 targets, but got {len(target_fields) if target_fields else None}"
            )
        return cls._inner_function(
            data=grouping_data[0][1],
            test_data=grouping_data[1][1],
            target_fields=target_fields,
            metric=metric,
            bias=kwargs.get("bias_estimation", None),
            scaled_counts=kwargs.get("scaled_counts"),
        )

    def _prepare_new_target(
        self, data: ExperimentData, t_data: Dataset, group_field: str
    ) -> Dataset:
        indexes = data.field_search(AdditionalMatchingRole())
        if len(indexes) == 0:
            raise ValueError("No indexes were found")
        new_target = data.ds.search_columns(TargetRole())[0]
        indexes = data.additional_fields[indexes[0]]
        indexes.index = t_data.index
        grouped_data = data.ds.groupby(group_field)
        control_indexes = indexes.loc[grouped_data[0][1].index]
        test_indexes = indexes.loc[grouped_data[1][1].index]
        self._calc_scaled_counts(control_indexes, "control")
        self._calc_scaled_counts(test_indexes, "test")
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
            matched_data = self._prepare_new_target(data, t_data, group_field)
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
        ):  # if the column is not suitable for the test, then the target will be empty, but if there is a role tempo, then this is normal behavior
            return data

        compare_result = self.calc(
            data=t_data,
            group_field=group_field,
            target_fields=target_fields,
            metric=self.metric,
            bias_estimation=bias,
            scaled_counts=self.__scaled_counts,
        )
        return self._set_value(data, compare_result)


class Bias(GroupOperator):
    def __init__(
        self,
        grouping_role: ABCRole | None = None,
        target_roles: list[ABCRole] | None = None,
        key: Any = "",
    ):
        super().__init__(
            grouping_role=grouping_role, target_roles=target_roles, key=key
        )

    @staticmethod
    def calc_coefficients(X: Dataset, Y: Dataset) -> list[float]:
        X_l = Dataset.create_empty(roles={"temp": InfoRole()}, index=X.index).fillna(1)
        X = X_l.append(X, axis=1).data.values
        return np.linalg.lstsq(X, Y.data.values, rcond=-1)[0][1:]

    @staticmethod
    def calc_bias(
        X: Dataset, X_matched: Dataset, coefficients: list[float]
    ) -> list[float]:
        return [
            (j - i).dot(coefficients)[0]
            for i, j in zip(X.data.values, X_matched.data.values)
        ]

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Dataset | None = None,
        target_fields: list[str] | None = None,
        features_fields: list[str] | None = None,
        **kwargs,
    ) -> dict:
        if target_fields is None or features_fields is None or test_data is None:
            raise NoneArgumentError(
                ["target_fields", "features_fields", "test_data"], "bias_estimation"
            )
        if data[target_fields[1]].isna().sum() > 0:
            return {
                "test": cls.calc_bias(
                    test_data[features_fields[: len(features_fields) // 2]],
                    test_data[features_fields[len(features_fields) // 2 :]],
                    cls.calc_coefficients(
                        test_data[features_fields[len(features_fields) // 2 :]],
                        test_data[target_fields[1]],
                    ),
                )
            }
        if test_data[target_fields[1]].isna().sum() > 0:
            return {
                "control": cls.calc_bias(
                    data[features_fields[: len(features_fields) // 2]],
                    data[features_fields[len(features_fields) // 2 :]],
                    cls.calc_coefficients(
                        data[features_fields[len(features_fields) // 2 :]],
                        data[target_fields[1]],
                    ),
                )
            }
        return {
            "test": cls.calc_bias(
                test_data[features_fields[: len(features_fields) // 2]],
                test_data[features_fields[len(features_fields) // 2 :]],
                cls.calc_coefficients(
                    test_data[features_fields[len(features_fields) // 2 :]],
                    test_data[target_fields[1]],
                ),
            ),
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
        target_fields: list[str] | None = None,
        features_fields: list[str] | None = None,
        **kwargs,
    ) -> dict:
        return cls._inner_function(
            grouping_data[0][1],
            test_data=grouping_data[1][1],
            target_fields=target_fields,
            features_fields=features_fields,
            **kwargs,
        )

    def _prepare_data(self, data: ExperimentData, t_data: Dataset) -> Dataset:
        indexes = data.additional_fields[data.field_search(AdditionalMatchingRole())[0]]
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
        ):  # if the column is not suitable for the test, then the target will be empty, but if there is a role tempo, then this is normal behavior
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
