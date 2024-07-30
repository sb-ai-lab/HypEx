from copy import deepcopy
from typing import Optional, Any, List, Literal, Union, Dict

from hypex.dataset import (
    Dataset,
    ABCRole,
    ExperimentData,
    TargetRole,
    InfoRole,
)
from hypex.ml.faiss import FaissNearestNeighbors
from hypex.operators.abstract import GroupOperator
from hypex.utils import SpaceEnum
from hypex.utils.enums import ExperimentDataEnum


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
        bias = None
        # bias = (
        #     data.additional_fields[
        #         data.get_one_id(Bias, ExperimentDataEnum.additional_fields)
        #     ]
        #     if data.get_one_id(Bias, ExperimentDataEnum.additional_fields)
        #     else None
        # )
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
                metric = "atc"
            else:
                metric = (
                    "att"
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
            bias=kwargs.get("bias", None),
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
        itt = (data[target_fields[0]] - data[target_fields[1]]).mean()
        itc = (test_data[target_fields[0]] - test_data[target_fields[1]]).mean()
        if kwargs.get("bias", False):
            itt -= kwargs.get("bias")["treated"]
            itc += kwargs.get("bias")["control"]
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

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, target_fields = self._get_fields(data)
        t_data = deepcopy(data.ds)
        distances_keys = data.get_ids(FaissNearestNeighbors, ExperimentDataEnum.groups)
        if len(target_fields) != 2:
            if len(distances_keys["FaissNearestNeighbors"]["groups"]) > 0:
                target_fields += data.groups[
                    distances_keys["FaissNearestNeighbors"]["groups"][0]
                ]["matched_df"].search_columns(self.target_roles)
            else:
                raise ValueError
        matched_data = data.groups[
            distances_keys["FaissNearestNeighbors"]["groups"][0]
        ]["matched_df"]
        self.key = str(
            target_fields[0] if len(target_fields) == 1 else (target_fields or "")
        )
        if (
            not target_fields and data.ds.tmp_roles
        ):  # если колонка не подходит для теста, то тагет будет пустой, но если есть темп роли, то это нормальное поведение
            return data

        compare_result = self.calc(
            data=t_data,
            matched_data=matched_data,
            group_field=group_field,
        )
        return self._set_value(data, compare_result)

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_fields: Optional[List[str]] = None,
        matched_data: Optional[Dataset] = None,
        **kwargs
    ) -> Dict:
        if matched_data is None:
            raise NameError("No matched data found")
        return cls._inner_function(grouping_data[0][1], grouping_data[1][1], **kwargs)

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Any:
        bias_c = cls.calc_bias(
            data,
            test_data,
            cls.calc_coefficients(data, Y_t, X_t),
        )
        bias_t = cls.calc_bias(
            X_t,
            X_c,
            matches_t,
            cls.calc_coefficients(grouping_data[1][1], Y_c, X_c),
        )
        return {"treated": bias_t, "control": bias_c}

    @staticmethod
    def calc_coefficients(data, matched_data: Dataset):
        X = Dataset.create_empty(
            roles={"term": InfoRole()}, index=range(len(data))
        ).fillna(1)
        X = X.append(matched_data)
        return np.linalg.lstsq(X.data.values, matched_data.data.values)[0][
            1:
        ]  # don't need intercept coef

    @staticmethod
    def calc_bias(X, X_matched, coefs):
        return [(X_j - X_i).dot(coefs) for X_i, X_j in zip(X, X_matched)]
