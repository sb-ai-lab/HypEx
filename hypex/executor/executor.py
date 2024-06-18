from abc import abstractmethod, ABC
from typing import Any, Dict, List, Optional, Sequence, Union

from hypex.dataset import (
    ABCRole,
    Dataset,
    ExperimentData,
    GroupingRole,
)
from hypex.utils import (
    ComparisonNotSuitableFieldError,
    FieldKeyTypes,
    NoColumnsError,
    SpaceEnum,
    AbstractMethodError,
    ID_SPLIT_SYMBOL,
)
from hypex.utils.adapter import Adapter


class Executor(ABC):
    def __init__(
        self,
        key: Any = "",
    ):
        self._id: str = ""
        self._params_hash = ""

        self.key: Any = key
        self._generate_id()

    def check_and_setattr(self, params: Dict[str, Any]):
        for key, value in params.items():
            if key in self.__dir__():
                setattr(self, key, value)

    def _generate_params_hash(self):
        self._params_hash = ""

    def _generate_id(self):
        self._generate_params_hash()
        self._id = ID_SPLIT_SYMBOL.join(
            [
                self.__class__.__name__,
                self._params_hash.replace(ID_SPLIT_SYMBOL, "|"),
                str(self._key).replace(ID_SPLIT_SYMBOL, "|"),
            ]
        )

    def set_params(self, params: SetParamsDictTypes) -> None:
        if isinstance(list(params)[0], str):
            self.check_and_setattr(params)
        elif isinstance(list(params)[0], type):
            for executor_class, class_params in params.items():
                if isinstance(self, executor_class):
                    self.check_and_setattr(class_params)
        else:
            raise ValueError(
                "params must be a dict of str to dict or a dict of class to dict"
            )
        self._generate_id()

    @property
    def id(self) -> str:
        return self._id

    @property
    def key(self) -> Any:
        return self._key

    @key.setter
    def key(self, value: Any):
        self._key = value
        self._generate_id()

    @property
    def params_hash(self) -> str:
        return self._params_hash

    @property
    def id_for_name(self) -> str:
        return self.id.replace(ID_SPLIT_SYMBOL, "_")

    @property
    def _is_transformer(self) -> bool:
        return False

    def _set_value(
        self, data: ExperimentData, value: Any, key: Any = None
    ) -> ExperimentData:
        # defined in order to avoid  unnecessary redefinition in classes like transformer
        return data

    @abstractmethod
    def execute(self, data: ExperimentData) -> ExperimentData:
        raise AbstractMethodError


class Calculator(Executor, ABC):
    @classmethod
    def calc(cls, data: Dataset, **kwargs):
        return cls._inner_function(data, **kwargs)

    @staticmethod
    @abstractmethod
    def _inner_function(data: Dataset, **kwargs) -> Any:
        raise AbstractMethodError


class GroupCalculator(Calculator):
    def __init__(
        self,
        grouping_role: Optional[ABCRole] = None,
        target_roles: Optional[List[ABCRole]] = None,
        space: SpaceEnum = SpaceEnum.auto,
        search_types: Union[type, List[type], None] = None,
        key: Any = "",
    ):
        self.grouping_role = grouping_role or GroupingRole()
        self.space = space
        self.__additional_mode = space == SpaceEnum.additional
        self._search_types = (
            search_types
            if isinstance(search_types, type) or search_types is None
            else [search_types]
        )
        self.target_roles = target_roles or []
        super().__init__(key=key)

    def _field_searching(
        self, data: ExperimentData, field, tmp_role: bool = False, search_types=None
    ):
        searched_field = []
        if self.space in [SpaceEnum.auto, SpaceEnum.data]:
            searched_field = data.ds.search_columns(
                field, tmp_role=tmp_role, search_types=search_types
            )
        if (
            self.space in [SpaceEnum.auto, SpaceEnum.additional]
            and searched_field == []
            and isinstance(data, ExperimentData)
        ):
            searched_field = data.additional_fields.search_columns(
                field, tmp_role=tmp_role, search_types=search_types
            )
            self.__additional_mode = True
        if len(searched_field) == 0:
            raise NoColumnsError(field)
        return searched_field

    def _get_fields(self, data: ExperimentData):
        group_field = self._field_searching(data, self.grouping_role)
        target_fields = self._field_searching(
            data, self.target_roles, search_types=self._search_types
        )
        return group_field, target_fields

    @staticmethod
    def _check_test_data(test_data: Optional[Dataset] = None) -> Dataset:
        if test_data is None:
            raise ValueError("test_data is needed for comparison")
        return test_data

    def _get_grouping_data(self, data: ExperimentData):
        group_field, target_fields = self._get_fields(data)
        if (
            not target_fields and data.ds.tmp_roles
        ):  # если колонка не подходит для теста, то тагет будет пустой, но если есть темп роли, то это нормальное поведение
            return data
        grouping_data = None
        if group_field[0] in data.groups:  # TODO: to recheck if this is a correct check
            grouping_data = list(data.groups[group_field[0]].items())
        return group_field, target_fields, grouping_data

    @staticmethod
    def _field_arg_universalization(
        field: Union[Sequence[FieldKeyTypes], FieldKeyTypes, None]
    ) -> List[FieldKeyTypes]:
        if not field:
            raise NoColumnsError(field)
        elif isinstance(field, FieldKeyTypes):
            return [field]
        return list(field)

    @classmethod
    @abstractmethod
    def _inner_function(
        cls, data: Dataset, test_data: Optional[Dataset] = None, **kwargs
    ) -> Any:
        raise AbstractMethodError

    @classmethod
    @abstractmethod
    def _execute_inner_function(cls, grouping_data, **kwargs) -> Dict:
        raise AbstractMethodError

    @classmethod
    def calc(
        cls,
        data: Dataset,
        group_field: Union[Sequence[FieldKeyTypes], FieldKeyTypes, None] = None,
        grouping_data: Optional[Dict[FieldKeyTypes, Dataset]] = None,
        target_fields: Optional[List[FieldKeyTypes]] = None,
        **kwargs,
    ) -> Dict:
        group_field = Adapter.to_list(group_field)

        if grouping_data is None:
            grouping_data = data.groupby(group_field)
        if len(grouping_data) > 1:
            grouping_data[0][1].tmp_roles = data.tmp_roles
        else:
            raise ComparisonNotSuitableFieldError(group_field)
        return cls._execute_inner_function(
            grouping_data, target_fields=target_fields, old_data=data, **kwargs
        )

    def execute(self, data: ExperimentData) -> ExperimentData:
        group_field, target_fields, grouping_data = self._get_grouping_data(data)
        compare_result = self.calc(
            data=data.ds,
            group_field=group_field,
            target_fields=target_fields,
            grouping_data=grouping_data,
        )
        return self._set_value(data, compare_result)
