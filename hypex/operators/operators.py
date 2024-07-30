from copy import deepcopy
from typing import Optional, Any, List, Literal, Union, Dict

from hypex.dataset import Dataset, ABCRole, ExperimentData, MatchingRole, TargetRole
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
        t_data = deepcopy(data.ds)
        if len(target_fields) != 2: 
            distances_keys = data.get_ids(FaissNearestNeighbors, ExperimentDataEnum.groups)
            if len(distances_keys["FaissNearestNeighbors"]["groups"]) > 0:
                target_fields += data.groups[distances_keys["FaissNearestNeighbors"]["groups"][0]]["matched_df"].search_columns(
                    self.target_roles)
            else: 
                raise ValueError
        if target_fields[1] not in t_data.columns:
            t_data = t_data.add_column(
                data.groups[distances_keys["FaissNearestNeighbors"]["groups"][0]]["matched_df"][target_fields[1]],
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
                grouping_data[0][1][grouping_data[0][1][target_fields[1]] == 0]
            ) == len(grouping_data[0][1]):
                metric = "atc"
            else:
                metric = (
                    "att"
                    if len(
                        grouping_data[1][1][grouping_data[1][1][target_fields[1]] == 0]
                    )
                    == len(grouping_data[1][1])
                    else "ate"
                )
        att = (
            grouping_data[0][1][target_fields[0]]
            - grouping_data[0][1][target_fields[1]]
        ).mean()
        if metric == "att":
            return {"ATT": att}
        atc = (
            grouping_data[1][1][target_fields[0]]
            - grouping_data[1][1][target_fields[1]]
        ).mean()
        if metric == "atc":
            return {"ATC": atc}
        len_test, len_control = len(grouping_data[0][1]), len(grouping_data[1][1])
        return {
            "ATT": att,
            "ATC": atc,
            "ATE": (att * len_test + atc * len_control) / (len_test + len_control),
        }

    @classmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Any:
        raise NotImplementedError
