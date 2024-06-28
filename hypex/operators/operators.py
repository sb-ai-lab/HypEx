from copy import deepcopy
from typing import Optional, Any, List, Literal, Union, Dict

from hypex.dataset import Dataset, ABCRole, ExperimentData, MatchingRole
from hypex.operators.abstract import GroupOperator
from hypex.utils import SpaceEnum, FieldKeyTypes


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
        self.key = str(
            target_fields[0] if len(target_fields) == 1 else (target_fields or "")
        )
        if (
            not target_fields and data.ds.tmp_roles
        ):  # если колонка не подходит для теста, то тагет будет пустой, но если есть темп роли, то это нормальное поведение
            return data
        if group_field[0] in data.groups:  # TODO: to recheck if this is a correct check
            grouping_data = list(data.groups[group_field[0]].items())
        else:
            grouping_data = data.ds.groupby(group_field)
        t_data = deepcopy(data.ds)
        if len(target_fields) < 2:
            index_fields = data.additional_fields.search_columns(MatchingRole())
            if not index_fields:
                raise Exception()
            for i in range(len(index_fields)):
                index_field = data.additional_fields[index_fields[i]].fillna(-1)
                # TODO фильтр -1, дроп, iloc, таргет столбец, присоединить

            # target_fields += indexes
            # t_data = t_data.add_column(
            #     data.additional_fields[target_field[0]],
            #     role={target_field[0]: MatchingRole()},
            # )
        compare_result = self.calc(
            data=t_data,
            group_field=group_field,
            grouping_data=grouping_data,
            target_fields=target_fields,
            metric=self.metric,
        )
        return self._set_value(data, compare_result)

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_fields: Optional[List[FieldKeyTypes]] = None,
        **kwargs
    ) -> Dict:
        metric = kwargs.get("metric", "auto")
        if target_fields is None or len(target_fields) != 2:
            raise ValueError("Нужно дописать ошибку")
        if metric == "auto":
            if grouping_data[0][target_fields[1]].isna().sum() > 0:
                metric = "atc" if grouping_data[0][0] < grouping_data[1][0] else "att"
            else:
                metric = "ate"
        if metric in ["atc", "att"]:
            return {
                metric.upper(): (
                    grouping_data[0][1][target_fields[0]]
                    - grouping_data[0][1][target_fields[1]]
                ).mean()
            }
        else:
            att = (
                grouping_data[0][1][target_fields[0]]
                - grouping_data[0][1][target_fields[1]]
            ).mean()
            atc = (
                grouping_data[1][1][target_fields[0]]
                - grouping_data[1][1][target_fields[1]]
            ).mean()
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
