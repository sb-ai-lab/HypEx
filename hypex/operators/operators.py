from copy import deepcopy
from typing import Optional, Any, List, Literal, Union, Dict

from plotly.figure_factory import np

from hypex.dataset import (
    Dataset,
    ABCRole,
    ExperimentData,
    TargetRole,
    InfoRole,
    FeatureRole,
)
from hypex.ml.faiss import FaissNearestNeighbors
from hypex.operators.abstract import GroupOperator
from hypex.utils import SpaceEnum
from hypex.utils.enums import ExperimentDataEnum
from hypex.utils.errors import NoneArgumentError, NotFoundInExperimentDataError


class SMD(GroupOperator):
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
        space: SpaceEnum = SpaceEnum.auto,
        metric: Optional[Literal["auto", "atc", "att", "ate"]] = None,
        key: Any = "",
    ):
        self.metric = metric or "auto"
        super().__init__(
            grouping_role=grouping_role, target_roles=target_roles, space=space, key=key
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, target_fields = self._get_fields(data=data)
        bias = (
            data.variables[data.get_one_id(Bias, ExperimentDataEnum.variables)]
            if data.get_ids(Bias, ExperimentDataEnum.variables)["Bias"]["variables"]
            != []
            else None
        )
        t_data = deepcopy(data.ds)
        distances_keys = data.get_ids(FaissNearestNeighbors, ExperimentDataEnum.groups)
        if len(target_fields) != 2:
            if len(distances_keys["FaissNearestNeighbors"]["groups"]) > 0:
                target_fields += data.groups[
                    distances_keys["FaissNearestNeighbors"]["groups"][0]
                ]["matched_df"].search_columns(self.target_roles)
            else:
                raise ValueError
        if target_fields[1] not in t_data.columns:
            t_data = t_data.add_column(
                data.groups[distances_keys["FaissNearestNeighbors"]["groups"][0]][
                    "matched_df"
                ][target_fields[1]],
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
            if len(
                grouping_data[0][1][grouping_data[0][1][target_fields[1]] == -1]
            ) == len(grouping_data[0][1]):
                metric = "att"
            else:
                metric = (
                    "atс"
                    if len(
                        grouping_data[1][1][grouping_data[1][1][target_fields[1]] == -1]
                    )
                    == len(grouping_data[1][1])
                    else "ate"
                )
        return cls._inner_function(
            data=grouping_data[0][1],
            test_data=grouping_data[1][1],
            target_fields=target_fields,
            metric=metric,
            bias=kwargs.get("bias_estimation", None),
        )

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Optional[Dataset] = None,
        target_fields: Optional[List[str]] = None,
        **kwargs
    ) -> Any:
        metric = kwargs.get("metric", "ate")
        itt = test_data[target_fields[0]] - test_data[target_fields[1]]
        itc = data[target_fields[1]] - data[target_fields[0]]
        if kwargs.get("bias", False):
            itc += Dataset.from_dict({"test": kwargs.get("bias")["test"]}, roles={})
            itt -= Dataset.from_dict(
                {"control": kwargs.get("bias")["control"]}, roles={}
            )
        itt = itt.mean()
        itc = itc.mean()
        if metric == "atc":
            return {"ATC": itc}
        if metric == "att":
            return {"ATC": itt}
        len_test, len_control = len(data), len(test_data)
        return {
            "ATT": itt,
            "ATC": itc,
            "ATE": (itt * len_test + itc * len_control) / (len_test + len_control),
        }


class Bias(GroupOperator):
    def __init__(
        self,
        grouping_role: Optional[ABCRole] = None,
        target_roles: Optional[List[ABCRole]] = None,
        space: SpaceEnum = SpaceEnum.auto,
        data_space: ExperimentDataEnum = ExperimentDataEnum.groups,
        executor_name: str = "FaissNearestNeighbors",
        key: Any = "",
    ):
        super().__init__(
            grouping_role=grouping_role, target_roles=target_roles, space=space, key=key
        )
        self.data_space = data_space
        self.executor_name = executor_name

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, target_fields = self._get_fields(data)
        t_data = deepcopy(data.ds)
        distances_keys = data.get_ids(self.executor_name, self.data_space)
        if (
            len(distances_keys[self.executor_name]["groups"]) > 0
        ):  # добавить поиск в analysis_tables
            matched_data_key = list(
                data.groups[distances_keys[self.executor_name]["groups"][0]].keys()
            )[0]
            matched_data = data.groups[distances_keys[self.executor_name]["groups"][0]][
                matched_data_key
            ]
            target_fields += matched_data.search_columns(
                self.target_roles
            )  # нужно проверять, что таргетов не 2
        else:
            raise NotFoundInExperimentDataError(self.executor_name)
        self.key = str(
            target_fields[0] if len(target_fields) == 1 else (target_fields or "")
        )
        if (
            not target_fields and data.ds.tmp_roles
        ):  # если колонка не подходит для теста, то тагет будет пустой, но если есть темп роли, то это нормальное поведение
            return data

        compare_result = self.calc(
            data=t_data.append(matched_data, axis=1),
            group_field=group_field,
            target_fields=target_fields,
            features_fields=t_data.search_columns(
                FeatureRole(), search_types=[int, float]
            )
            + matched_data.search_columns(FeatureRole(), search_types=[int, float]),
        )
        return self._set_value(data, compare_result)

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_fields: Optional[List[str]] = None,
        features_fields: Optional[List[str]] = None,
        **kwargs
    ) -> Dict:
        return cls._inner_function(
            grouping_data[0][1],
            test_data=grouping_data[1][1],
            target_fields=target_fields,
            features_fields=features_fields,
            **kwargs
        )

    @classmethod
    def _inner_function(
        cls,
        data: Dataset,
        test_data: Optional[Dataset] = None,
        target_fields: Optional[List[str]] = None,
        features_fields: Optional[List[str]] = None,
        **kwargs
    ) -> Dict:
        if target_fields is None or features_fields is None or test_data is None:
            raise NoneArgumentError(
                ["target_fields", "features_fields", "test_data"], "bias_estimation"
            )
        return {
            "test": cls.calc_bias(
                test_data[features_fields[: len(features_fields) // 2]],
                test_data[features_fields[len(features_fields) // 2 :]],
                cls.calc_coefficients(
                    data[features_fields[len(features_fields) // 2 :]],
                    data[target_fields[1]],
                ),
            ),
            "control": cls.calc_bias(
                data[features_fields[: len(features_fields) // 2]],
                data[features_fields[len(features_fields) // 2 :]],
                cls.calc_coefficients(
                    test_data[features_fields[len(features_fields) // 2 :]],
                    test_data[target_fields[1]],
                ),
            ),
        }

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
