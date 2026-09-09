"""ExperimentData: container for datasets, variables, and analysis artifacts.

Provides a unified interface for managing experiment-related data structures,
including main datasets, additional fields, variables, grouped datasets, and
precomputed analysis tables. Supports role-based column search and ID mapping.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any, Literal

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
from ..utils.logger import logger
from .roles import AdditionalRole, ABCRole, DefaultRole, DisabledRole

_SUPPORTED_SPACES = frozenset(
    {
        ExperimentDataEnum.additional_fields,
        ExperimentDataEnum.analysis_tables,
        ExperimentDataEnum.groups,
        ExperimentDataEnum.variables,
    }
)

@logger.log_methods(log_args=False, log_result=False, private=True, static=True)
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
        self.variables: dict[str, dict[str, int | float]] = {}
        self.groups: dict[str, dict[str, Dataset]] = {}
        self.analysis_tables: dict[str, SmallDataset] = {}
        self.id_name_mapping: dict[str, str] = {}

        self._initial_cols = deepcopy(self._data.columns)

    @property
    def initial_ds(self) -> Dataset | SmallDataset:
        return self._data[self._initial_cols]

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
        Extracts the first part (class name) and the last part (key).

        Args:
            id_str: ID string potentially containing the split symbol.

        Returns:
            Tuple of (prefix, suffix). Suffix is None if the split symbol is not found.
        """
        parts = id_str.split(ID_SPLIT_SYMBOL)
        prefix = parts[0]
        suffix = parts[-1] if len(parts) > 1 else None
        return prefix, suffix

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
            return exec_id_str in self._data.columns
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

        Writes columns directly into self._data (the main dataset) instead of
        a separate additional_fields dataset. The property-shim `additional_fields`
        will provide a filtered view for backwards compatibility.

        Args:
            exec_id: Normalized column/identifier name.
            value: Data to add as a column or merge.
            role: Semantic role assigned to the new column.

        Returns:
            Self for method chaining.
        """
        normalized_role = self._normalize_role(role)

        storage_level = self._data.get_storage_level() or "MEMORY_AND_DISK"
        was_persisted = self._data.is_persisted

        if was_persisted:
            self._data.unpersist()


        if not isinstance(value, Dataset):
            # Raw data (list, scalar, etc.) — add as a single column
            new_data = self._data.add_column(
                data=value, 
                role={exec_id: normalized_role}
            )
        elif len(value.columns) == 1:
            # Single-column Dataset — extract the column and add with exec_id as name
            new_data = self._data.add_column(
                data=value[value.columns[0]], 
                role={exec_id: normalized_role}
            )
            # return self
        else:
            # Multi-column Dataset — rename all columns to avoid naming collisions
            rename_dict = {col: f"{exec_id}_{col}" for col in value.columns}
            renamed_value = value.rename(names=rename_dict)
            new_data = self._data.merge(
                right=renamed_value,
                left_index=True,
                right_index=True
            )
            # Apply roles: the first column gets the normalized_role, others keep their original roles
            for i, col in enumerate(value.columns):
                new_col_name = f"{exec_id}_{col}"
                if i == 0:
                    new_data.roles[new_col_name] = normalized_role
                else:
                    new_data.roles[new_col_name] = value.roles.get(col, DefaultRole())
        new_data.persist(storage_level=storage_level, action="none")
        if was_persisted:
            self._data.unpersist()
        self._data = new_data
        return self
    
    @property
    def additional_fields(self) -> Dataset | SmallDataset:
        """Backwards-compatible view of ds filtered to AdditionalRole columns.
        
        Returns a new Dataset containing only columns whose role is a subclass
        of AdditionalRole. This shim enables incremental migration: writers
        in Wave 3 can continue using data.additional_fields syntax while
        actual storage moves to ds.
        
        WARNING: This is a READ-ONLY view. Writes through this property
        will modify a copy and NOT the underlying ds.
        """
        # additional_cols = [
        #     col for col, role in self._data.roles.items()
        #     if isinstance(role, AdditionalRole)
        # ]
        additional_cols = list(set(self._data.columns) - set(self._initial_cols))
        if not additional_cols:
            return self._data.create_empty(
                backend=self._data.backend_type,
                session=self._data.session,
                roles={}
            )
        view = self._data[additional_cols]
        view.roles = {c: self._data.roles[c] for c in additional_cols}
        return view

    def cleanup_additional(self) -> Self:
        """Remove all columns with AdditionalRole-derived roles from ds.
        
        Called at the end of Output.extract() to ensure that public-facing
        experiment results do not leak internal/synthetic columns.
        """
        if self._data.is_persisted:
            self._data.unpersist()

        cols_to_drop = [
            col for col, role in self._data.roles.items()
            if isinstance(role, AdditionalRole)
        ]
        cols_to_enable = self._data.search_columns(DisabledRole())
        if cols_to_drop:
            self._data = self._data.drop(columns=cols_to_drop)

        if cols_to_enable:
            self._data = self._data.replace_roles(
                new_roles_map={
                    col: self._data.roles[col].initial_role for col in cols_to_enable
                }
            )
        return self
    
    def _clean_ds_for_iteration(self) -> Dataset | SmallDataset:
        """Return a copy of ds with AdditionalRole columns removed.
        
        This restores the iteration isolation that was previously provided
        by the separate additional_fields dataset. Used by ParamsExperiment,
        CycledExperiment, and GroupExperiment to ensure each iteration
        starts with a clean dataset (no leftover synthetic columns from
        previous iterations).
        
        Returns:
            A new Dataset without AdditionalRole columns.
        """
        additional_cols = [
            col for col, role in self._data.roles.items()
            if isinstance(role, AdditionalRole)
        ]
        if not additional_cols:
            return self._data

        cleaned = self._data.drop(columns=additional_cols)

        # truncate the computational graph (DAG) in Spark,
        # to avoid exponential slowdown at each iteration.
        if cleaned.backend_type == BackendsEnum.spark:
            cleaned.checkpoint(eager=True)

        return cleaned

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
        if isinstance(value, Dataset):
            value = value.to_small_dataset()

        self.analysis_tables[exec_id] = value
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
        space: Literal["all", "ds", "additional_fields"] = "all",
    ) -> list[str]:
        """Search for column names matching specified semantic roles.
        
        After the refactor, ALL columns live in self.ds 
        (including AdditionalRole columns), so we search only there.
        """
        space_dict = {
            "all": self.ds,
            "ds": self.ds[self._initial_cols],
            "additional_fields": self.additional_fields,
        }
        search_space = space_dict[space]
        roles_list = Adapter.to_list(roles)
        return search_space.search_columns(
            roles_list, tmp_role=tmp_role, search_types=search_types
        )

    def field_data_search(
        self,
        roles: ABCRole | Iterable[ABCRole],
        tmp_role: bool = False,
        search_types: list[type] | None = None,
        space: Literal["all", "ds", "additional_fields"] = "all",
    ) -> Dataset:
        """Build a new dataset containing only columns that match the specified roles.

        Instead of creating an empty dataset with the full 60M-row index
        (which forces a ``toPandas()`` call on the driver and triggers
        executor checkpoint reads), this method performs a pure column
        projection via ``self._data[cols]``. This is an O(1) Spark
        transformation that never materializes the index and never reads
        from local_checkpoint.

        Args:
            roles: A single role or iterable of roles to search for.
            tmp_role: Whether to search in temporary roles instead of
                permanent ones. Defaults to ``False``.
            search_types: Optional list of Python types to additionally
                filter columns by.

        Returns:
            A new ``Dataset`` containing only the matched columns with
            the requested roles applied. Returns an empty dataset (without
            materialized index) if no columns match.
        """
        roles_list = Adapter.to_list(roles)
        role_columns = {
            role: self.field_search(role, tmp_role, search_types, space)
            for role in roles_list
        }

        # Collect unique columns preserving first-seen order and map each
        # column to the role that requested it.
        cols: list[str] = []
        new_roles: dict[str, ABCRole] = {}
        for role, found_cols in role_columns.items():
            for col in found_cols:
                if col not in cols:
                    cols.append(col)
                    new_roles[col] = role

        if not cols:
            # Return a truly empty dataset WITHOUT evaluating the 60M index.
            # Passing index=None prevents SparkNavigation.create_empty from
            # calling ps.DataFrame(index=...) which triggers toPandas().
            return Dataset.create_empty(
                roles={},
                backend=self._data.backend_type,
                session=self._data.session,
            )

        # Pure column projection: a lazy Spark transformation that does not
        # trigger any action, does not read from checkpoint, and does not
        # ship the index to the driver.
        subset_ds = self._data[cols]

        # Apply the requested roles to the projected subset.
        for col, role in new_roles.items():
            subset_ds.roles[col] = role

        return subset_ds
