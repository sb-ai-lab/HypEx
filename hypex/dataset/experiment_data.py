"""ExperimentData: container for datasets, variables, and analysis artifacts.

Provides a unified interface for managing experiment-related data structures,
including main datasets, additional fields, variables, grouped datasets, and
precomputed analysis tables. Supports role-based column search and ID mapping.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

try:
    from typing import Self  # Python >= 3.11
except ImportError:
    from typing_extensions import Self  # Python < 3.11

from .dataset import Dataset, SmallDataset
from ..utils import (
    BackendsEnum,
    ExperimentDataEnum,
    ID_SPLIT_SYMBOL,
    NotFoundInExperimentDataError,
)
from ..utils.adapter import Adapter
from .roles import AdditionalRole, ABCRole

logger = logging.getLogger(__name__)

_SUPPORTED_SPACES = frozenset(
    {
        ExperimentDataEnum.additional_fields,
        ExperimentDataEnum.analysis_tables,
        ExperimentDataEnum.groups,
        ExperimentDataEnum.variables,
    }
)


class ExperimentData:
    """Container for experiment-related data structures.

    Holds the main dataset, auxiliary fields, scalar variables, grouped datasets,
    and precomputed analysis tables. Provides role-based search and unified
    ID resolution across all internal spaces.

    Attributes:
        _data: Primary dataset wrapper.
        additional_fields: Auxiliary dataset for extra columns.
        variables: Nested dict for scalar experiment parameters.
        groups: Nested dict for categorized dataset collections.
        analysis_tables: Dict mapping IDs to precomputed SmallDataset results.
        id_name_mapping: Optional human-readable name overrides for IDs.
    """

    def __init__(self, data: Dataset | SmallDataset) -> None:
        """Initialize ExperimentData with a base dataset.

        Creates an empty dataset for additional fields aligned with the main
        dataset's index and backend, and initializes all internal containers.

        Args:
            data: The primary dataset to wrap. Must be a Dataset or SmallDataset instance.
        """
        self._data: Dataset | SmallDataset = data
        self.additional_fields: Dataset | SmallDataset = data.create_empty(
            index=data.index, backend=data.backend_type, session=data.session
        )
        self.variables: dict[str, dict[str, int | float]] = {}
        self.groups: dict[str, dict[str, Dataset]] = {}
        self.analysis_tables: dict[str, SmallDataset] = {}
        self.id_name_mapping: dict[str, str] = {}

    @property
    def ds(self) -> Dataset | SmallDataset:
        """Get the primary dataset.

        Returns:
            The main Dataset or SmallDataset instance managed by this container.
        """
        return self._data

    @staticmethod
    def create_empty(
        roles: dict[str, ABCRole] | None = None,
        backend: BackendsEnum = BackendsEnum.pandas,
        index: Any | None = None,
    ) -> Self:
        """Create an empty ExperimentData instance.

        Args:
            roles: Optional semantic roles mapping for the empty dataset columns.
            backend: Data processing backend (pandas or spark). Defaults to pandas.
            index: Optional index labels for the empty dataset.

        Returns:
            A new ExperimentData instance containing an empty dataset.
        """
        idx = index.index if isinstance(index, Dataset) else index
        ds = Dataset.create_empty(backend=backend, roles=roles, index=idx)
        return ExperimentData(ds)

    @staticmethod
    def _normalize_executor_id(executor_id: str | dict[str, str] | list) -> str:
        """Normalize executor identifier to a string.

        Handles various input formats (dict with single key, list with single item,
        or raw string/int) and returns a consistent string identifier.

        Args:
            executor_id: Identifier in string, dict, or list format.

        Returns:
            Normalized string identifier.
        """
        if isinstance(executor_id, dict):
            return next(iter(executor_id))
        if isinstance(executor_id, list):
            return executor_id[0]
        return str(executor_id)

    @staticmethod
    def _normalize_role(role: Any) -> Any:
        """Extract role value from wrapper containers.

        Unwraps single-element lists or single-value dictionaries to return
        the underlying role object directly.

        Args:
            role: Role instance, list, or dict wrapper.

        Returns:
            The unwrapped role value.
        """
        if isinstance(role, list):
            return role[0]
        if isinstance(role, dict):
            return next(iter(role.values()))
        return role

    @staticmethod
    def _parse_id_for_search(id_str: str) -> tuple[str, str | None]:
        """Safely parse composite ID strings using the configured split symbol.

        Uses str.partition() to avoid index errors when the separator is absent.

        Args:
            id_str: ID string potentially containing the split symbol.

        Returns:
            Tuple of (prefix, suffix). Suffix is None if the split symbol is not found.
        """
        prefix, sep, suffix = id_str.partition(ID_SPLIT_SYMBOL)
        return prefix, suffix if sep else None

    def check_hash(self, executor_id: int | str, space: ExperimentDataEnum) -> bool:
        """Check if an executor ID exists in a specified data space.

        Args:
            executor_id: Identifier to search for.
            space: Target ExperimentDataEnum space to check.

        Returns:
            True if the ID exists in the specified space, False otherwise.
            Automatically searches all supported spaces if an unknown space is provided.
        """
        exec_id_str = str(executor_id)
        if space == ExperimentDataEnum.additional_fields:
            return exec_id_str in self.additional_fields.columns
        if space == ExperimentDataEnum.variables:
            return exec_id_str in self.variables
        if space == ExperimentDataEnum.analysis_tables:
            return exec_id_str in self.analysis_tables

        # Safe fallback: prevents recursion on unknown enum members
        if space not in _SUPPORTED_SPACES:
            return False
        return any(self.check_hash(executor_id, s) for s in _SUPPORTED_SPACES)

    def set_value(
        self,
        space: ExperimentDataEnum,
        executor_id: str | dict[str, str],
        value: Any,
        key: str | None = None,
        role: ABCRole | None = None,
    ) -> Self:
        """Store a value in the specified experiment data space.

        Routes the value to the appropriate internal container based on the
        target space enum. Normalizes executor_id and handles role extraction.

        Args:
            space: Target storage space (e.g., additional_fields, variables).
            executor_id: Unique identifier for the stored value.
            value: Data to store (Dataset, SmallDataset, scalar, or dict).
            key: Optional sub-key for nested storage (variables, groups).
            role: Optional semantic role for column assignment in additional_fields.

        Returns:
            Self for method chaining.
        """
        exec_id = self._normalize_executor_id(executor_id)

        if space == ExperimentDataEnum.additional_fields:
            return self._set_additional_fields(exec_id, value, role)
        if space == ExperimentDataEnum.analysis_tables:
            return self._set_analysis_tables(exec_id, value)
        if space == ExperimentDataEnum.variables:
            return self._set_variables(exec_id, value, key)
        if space == ExperimentDataEnum.groups:
            return self._set_groups(exec_id, value, key)

        raise ValueError(f"Unknown space: {space}")

    def _set_additional_fields(self, exec_id: str, value: Any, role: Any) -> Self:
        """Handle storage in the additional_fields space.

        Adds columns to the auxiliary dataset, handling single-column datasets,
        raw data sequences, and multi-column merges with index alignment.

        Args:
            exec_id: Normalized column/identifier name.
            value: Data to add as a column or merge.
            role: Semantic role assigned to the new column.

        Returns:
            Self for method chaining.
        """
        if not isinstance(value, Dataset):
            self.additional_fields = self.additional_fields.add_column(
                data=value, role={exec_id: role}
            )
            return self

        if len(value.columns) == 1:
            normalized_role = self._normalize_role(role)
            self.additional_fields = self.additional_fields.add_column(
                data=value, role={exec_id: normalized_role}
            )
            return self

        rename_dict = {value.columns[0]: exec_id}
        self.additional_fields = self.additional_fields.merge(
            right=value.rename(names=rename_dict), left_index=True, right_index=True
        )
        return self

    def _set_analysis_tables(self, exec_id: str, value: Any) -> Self:
        """Handle storage in the analysis_tables space.

        Converts Dataset instances to SmallDataset if necessary, validates
        the input type, and stores the result. Logs operations for debugging.

        Args:
            exec_id: Identifier for the analysis table.
            value: SmallDataset or Dataset to store.

        Returns:
            Self for method chaining.

        Raises:
            TypeError: If value cannot be converted to SmallDataset.
        """
        logger.debug(
            "set_value ENTRY | executor_id=%s | value_type=%s",
            exec_id,
            type(value).__name__,
        )
        logger.debug(
            "set_value | executor_id=%s | keys_before=%s",
            exec_id,
            list(self.analysis_tables.keys()),
        )

        if isinstance(value, Dataset):
            value = value.to_small_dataset()
        if not isinstance(value, SmallDataset):
            raise TypeError(f"Wrong value {value} for converting to SmallDataset")

        self.analysis_tables[exec_id] = value
        logger.debug("set_value | keys_after=%s", list(self.analysis_tables.keys()))
        return self

    def _set_variables(self, exec_id: str, value: Any, key: str | None) -> Self:
        """Handle storage in the variables space.

        Manages scalar experiment variables, supporting nested dict updates
        and requiring explicit keys for non-dict values.

        Args:
            exec_id: Variable group identifier.
            value: Scalar value or dict of key-value pairs.
            key: Sub-key for nested assignment (required for scalars).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If key is missing when required.
        """
        if exec_id in self.variables:
            if key is None:
                raise ValueError("key is required when updating existing variable")
            self.variables[exec_id][key] = value
        elif isinstance(value, dict):
            self.variables[exec_id] = value
        else:
            if key is None:
                raise ValueError("key is required for new scalar variable")
            self.variables[exec_id] = {key: value}
        return self

    def _set_groups(self, exec_id: str, value: Any, key: str | None) -> Self:
        """Handle storage in the groups space.

        Stores datasets under group identifiers with explicit sub-keys.

        Args:
            exec_id: Group identifier.
            value: Dataset to store in the group.
            key: Sub-key within the group (required).

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If key is not provided.
        """
        if key is None:
            raise ValueError("key is required for groups")
        self.groups.setdefault(exec_id, {})[key] = value
        return self

    def get_ids(
        self,
        classes: type | Iterable[type] | str | Iterable[str],
        searched_space: ExperimentDataEnum | Iterable[ExperimentDataEnum] | None = None,
        key: str | None = None,
    ) -> dict[str, dict[str, list[str]]]:
        """Search for IDs matching class names across experiment spaces.

        Parses composite IDs using the configured split symbol and filters
        by class name prefix and optional suffix key.

        Args:
            classes: Class type(s) or name string(s) to search for.
            searched_space: Specific space(s) to search. If None, searches all supported spaces.
            key: Optional suffix filter for IDs.

        Returns:
            Nested dictionary mapping class names to spaces to lists of matching IDs.
        """
        target_classes = [
            c.__name__ if isinstance(c, type) else c for c in Adapter.to_list(classes)
        ]
        # Safe fallback: only iterate over spaces we actually manage
        spaces_to_search = (
            Adapter.to_list(searched_space)
            if searched_space is not None
            else list(_SUPPORTED_SPACES)
        )

        spaces_map = {
            ExperimentDataEnum.additional_fields: self.additional_fields.columns,
            ExperimentDataEnum.analysis_tables: self.analysis_tables.keys(),
            ExperimentDataEnum.groups: self.groups.keys(),
            ExperimentDataEnum.variables: self.variables.keys(),
        }

        result = {}
        for cls_name in target_classes:
            cls_matches = {}
            for space in spaces_to_search:
                # Skip unsupported enum members gracefully
                if space not in spaces_map:
                    cls_matches[space.value] = []
                    continue

                matched = []
                for raw_id in spaces_map[space]:
                    prefix, suffix = self._parse_id_for_search(str(raw_id))
                    if prefix == cls_name and (key is None or suffix == key):
                        matched.append(str(raw_id))
                cls_matches[space.value] = matched
            result[cls_name] = cls_matches
        return result

    def get_one_id(
        self,
        class_: type | str,
        space: ExperimentDataEnum,
        key: str | None = None,
    ) -> str:
        """Retrieve a single matching ID or raise an error if not found.

        Args:
            class_: Class type or name to search for.
            space: Target space to search in.
            key: Optional suffix filter.

        Returns:
            The first matching ID string.

        Raises:
            NotFoundInExperimentDataError: If no matching ID is found.
        """
        cls_name = class_ if isinstance(class_, str) else class_.__name__
        matches = self.get_ids(cls_name, space, key)
        ids = matches.get(cls_name, {}).get(space.value, [])
        if not ids:
            raise NotFoundInExperimentDataError(cls_name)
        return ids[0]

    def copy(self, data: Dataset | SmallDataset | None = None) -> Self:
        """Create a deep copy of this ExperimentData instance.

        Args:
            data: Optional replacement dataset for the primary data field.

        Returns:
            A new ExperimentData instance with deep-copied internal state.
        """
        result = deepcopy(self)
        if data is not None:
            result._data = data
        return result

    def field_search(
        self,
        roles: ABCRole | Iterable[ABCRole],
        tmp_role: bool = False,
        search_types: list[type] | None = None,
    ) -> list[str]:
        """Search for column names matching specified semantic roles.

        Separates search into main dataset and additional fields based on
        role type (AdditionalRole vs others).

        Args:
            roles: Role instance(s) to match.
            tmp_role: If True, searches in temporary roles instead of permanent ones.
            search_types: Optional list of Python types to filter roles by.

        Returns:
            List of column names matching the criteria.
        """
        roles_list = Adapter.to_list(roles)
        additional_roles = [r for r in roles_list if isinstance(r, AdditionalRole)]
        data_roles = [r for r in roles_list if r not in additional_roles]

        found = []
        if data_roles:
            found.extend(
                self.ds.search_columns(
                    data_roles, tmp_role=tmp_role, search_types=search_types
                )
            )
        if additional_roles:
            found.extend(
                self.additional_fields.search_columns(
                    additional_roles, tmp_role=tmp_role, search_types=search_types
                )
            )
        return found

    def field_data_search(
        self,
        roles: ABCRole | Iterable[ABCRole],
        tmp_role: bool = False,
        search_types: list[type] | None = None,
    ) -> Dataset:
        """Build a new dataset containing columns matching specified roles.

        Extracts data from both main and additional datasets based on role
        matching, preserving original roles in the output.

        Args:
            roles: Role instance(s) to match.
            tmp_role: If True, searches in temporary roles.
            search_types: Optional type filter for role matching.

        Returns:
            New Dataset instance containing only the matched columns.
        """
        roles_list = Adapter.to_list(roles)
        role_columns = {
            role: self.field_search(role, tmp_role, search_types) for role in roles_list
        }

        searched = Dataset.create_empty(
            index=self._data.index,
            backend=self._data.backend_type,
            session=self._data.session,
        )
        for role, cols in role_columns.items():
            for col in cols:
                src = (
                    self.additional_fields
                    if isinstance(role, AdditionalRole)
                    else self.ds
                )
                searched = searched.add_column(data=src[col], role={col: role})
        return searched
