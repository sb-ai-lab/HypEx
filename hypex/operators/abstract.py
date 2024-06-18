from abc import abstractmethod
from typing import Any, Dict, List, Optional

from hypex.dataset import (
    Dataset,
    ExperimentData,
)
from hypex.executor import GroupCalculator
from hypex.utils import (
    ExperimentDataEnum,
    FieldKeyTypes,
    AbstractMethodError,
)


class GroupOperator(GroupCalculator):

    @classmethod
    @abstractmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Any:
        raise AbstractMethodError

    @classmethod
    def _execute_inner_function(
        cls,
        grouping_data,
        target_fields: Optional[List[FieldKeyTypes]] = None,
        **kwargs,
    ) -> Dict:
        if len(target_fields) != 2:
            raise ValueError
        result = {}
        for group, group_data in grouping_data:
            result[group[0]] = cls._inner_function(
                data=group_data[target_fields[0]],
                test_data=group_data[target_fields[1]],
                **kwargs,
            )
        return result

    def _set_value(
        self, data: ExperimentData, value: Optional[Dict] = None, key: Any = None
    ) -> ExperimentData:
        data.set_value(
            ExperimentDataEnum.variables,
            self.id,
            str(self.__class__.__name__),
            value,
        )
        return data
