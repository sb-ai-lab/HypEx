class ExperimentData:
    def __init__(self, data: Dataset):
        self._data = data
        self._data = {
            "data": data,
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

    def check_hash(self, executor_id: int, space: ExperimentDataEnum) -> bool:
        if space == ExperimentDataEnum.additional_fields:
            return executor_id in self.additional_fields.columns
        elif space == ExperimentDataEnum.variables:
            return executor_id in self.variables.keys()
        elif space == ExperimentDataEnum.analysis_tables:
            return executor_id in self.analysis_tables
        else:
            return any(self.check_hash(executor_id, s) for s in ExperimentDataEnum)

    def set_value(
        self,
        space: ExperimentDataEnum,
        executor_id: Union[str, Dict[str, str]],
        value: Any,
        key: Optional[str] = None,
        role=None,
    ) -> "ExperimentData":
        if space == ExperimentDataEnum.additional_fields:
            if not isinstance(value, Dataset) or len(value.columns) == 1:
                self.additional_fields = self.additional_fields.add_column(
                    data=value, role={executor_id: role}
                )
            else:
                rename_dict = (
                    {value.columns[0]: executor_id}
                    if isinstance(executor_id, str)
                    else executor_id
                )
                value = value.rename(names=rename_dict)
                self.additional_fields = self.additional_fields.merge(
                    right=value, left_index=True, right_index=True
                )
        elif space == ExperimentDataEnum.analysis_tables:
            self.analysis_tables[executor_id] = value
        elif space == ExperimentDataEnum.variables:
            if executor_id in self.variables:
                self.variables[executor_id][key] = value
            elif isinstance(value, Dict):
                self.variables[executor_id] = value
            else:
                self.variables[executor_id] = {key: value}
        elif space == ExperimentDataEnum.groups:
            if executor_id not in self.groups:
                self.groups[executor_id] = {key: value}
            else:
                self.groups[executor_id][key] = value
        return self

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
