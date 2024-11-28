from typing import Any, Union, Iterable

from ..dataset import Dataset
from ..utils import BackendsEnum
from ..executor.executor_state import DatasetSpace, ExecutorState


class ExperimentData:
    def __init__(self, data: Dataset):
        self._data = {
            "data": data,
        }
        self._index = {
            "additional_fields": {},
            "variables": {},
            "groups": {},
            "analysis_tables": {},
        }

    @property
    def ds(self):
        """
        Returns the dataset property.

        Returns:
            The dataset property.
        """
        return self._data["data"]

    @staticmethod
    def create_empty(
        roles=None, backend=BackendsEnum.pandas, index=None
    ) -> "ExperimentData":
        """Creates an empty ExperimentData object"""
        ds = Dataset.create_empty(backend, roles, index)
        return ExperimentData(ds)

    def set_value(
        self,
        executor_state: ExecutorState,
        value: Any,
    ) -> "ExperimentData":
        def refresh_index_structure():
            if executor_state.save_space is None:
                raise ValueError(
                    "The ExecutorState must contain an indication of the save_space"
                )
            if executor_state.executor not in self._index:  # add executor in index
                self._index[executor_state.save_space][
                    executor_state.executor
                ] = executor_state.get_dict_for_index()
            elif (
                executor_state.parameters
                not in self._index[executor_state.save_space][executor_state.executor]
            ):  # add executor parameters in index
                self._index[executor_state.save_space][executor_state.executor][
                    executor_state.parameters
                ].update(executor_state.get_dict_for_index())

        refresh_index_structure()
        self._data[str(executor_state)] = value
        self._index[executor_state.save_space][executor_state.executor][
            executor_state.parameters
        ][executor_state.key] = str(executor_state)
        return self

    def search_state(
        self,
        space: Union[Iterable[DatasetSpace], DatasetSpace, None] = None,
        executor: Union[type, str, None] = None,
    ):
        pass

    def get_ids(
        self,
        classes: Union[type, Iterable[type], str, Iterable[str]],
        searched_space: Union[
            ExperimentDataEnum, Iterable[ExperimentDataEnum], None
        ] = None,
        key: Optional[str] = None,
    ) -> Dict[str, Dict[str, List[str]]]:
        def check_id(id_: str, class_: str) -> bool:
            result = id_[: id_.find(ID_SPLIT_SYMBOL)] == class_

            if result and key is not None:
                result = id_[id_.rfind(ID_SPLIT_SYMBOL) + 1 :] == key
            return result

        spaces = {
            ExperimentDataEnum.additional_fields: self.additional_fields.columns,
            ExperimentDataEnum.analysis_tables: self.analysis_tables.keys(),
            ExperimentDataEnum.groups: self.groups.keys(),
            ExperimentDataEnum.variables: self.variables.keys(),
        }
        classes = [
            c.__name__ if isinstance(c, type) else c for c in Adapter.to_list(classes)
        ]
        searched_space = (
            Adapter.to_list(searched_space) if searched_space else list(spaces.keys())
        )

        return {
            class_: {
                space.value: [
                    str(id_) for id_ in spaces[space] if check_id(id_, class_)
                ]
                for space in searched_space
            }
            for class_ in classes
        }

    def get_one_id(
        self,
        class_: Union[type, str],
        space: ExperimentDataEnum,
        key: Optional[str] = None,
    ) -> str:
        class_ = class_ if isinstance(class_, str) else class_.__name__
        result = self.get_ids(class_, space, key)
        if (class_ not in result) or (not len(result[class_][space.value])):
            raise NotFoundInExperimentDataError(class_)
        return result[class_][space.value][0]

    def copy(self, data: Optional[Dataset] = None) -> "ExperimentData":
        result = deepcopy(self)
        if data is not None:
            result._data = data
        return result

    def field_search(
        self,
        roles: Union[ABCRole, Iterable[ABCRole]],
        tmp_role: bool = False,
        search_types=None,
    ) -> List[str]:
        searched_field = []
        roles = Adapter.to_list(roles)
        field_in_additional = [
            role for role in roles if isinstance(role, AdditionalRole)
        ]
        field_in_data = [role for role in roles if role not in field_in_additional]
        if field_in_data:
            searched_field += self.ds.search_columns(
                field_in_data, tmp_role=tmp_role, search_types=search_types
            )
        if field_in_additional and isinstance(self, ExperimentData):
            searched_field += self.additional_fields.search_columns(
                field_in_additional, tmp_role=tmp_role, search_types=search_types
            )
        return searched_field

    def field_data_search(
        self,
        roles: Union[ABCRole, Iterable[ABCRole]],
        tmp_role: bool = False,
        search_types=None,
    ) -> Dataset:
        searched_data: Dataset = (
            Dataset.create_empty()
        )  # TODO: backend check to be added
        roles = Adapter.to_list(roles)
        roles_columns_map = {
            role: self.field_search(role, tmp_role, search_types) for role in roles
        }
        for role, columns in roles_columns_map.items():
            for column in columns:
                t_data = (
                    self.additional_fields[column]
                    if isinstance(role, AdditionalRole)
                    else self.ds[column]
                )
                searched_data = searched_data.add_column(
                    data=t_data, role={column: role}
                )
        return searched_data
