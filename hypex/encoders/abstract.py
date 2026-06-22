from __future__ import annotations
from collections.abc import Sequence
from typing import Any
from ..dataset import Dataset, ExperimentData, FeatureRole, DisableRole
from ..executor import Calculator
from ..utils import (
    NAME_BORDER_SYMBOL,
    AbstractMethodError,
    CategoricalTypes,
    ExperimentDataEnum,
)


class Encoder(Calculator):
    """Base class for all categorical feature encoders in the HypEx library.

    Provides the common interface and execution flow for encoding categorical
    columns into numeric representations (e.g., one-hot, ordinal, target
    encoding). Subclasses must implement the ``_inner_function`` method to
    define the actual encoding logic for a specific backend (Pandas, Spark).

    The encoding pipeline consists of the following steps:
        1. Discover all columns in the dataset that match ``target_roles``
           and have a categorical data type (``search_types = [CategoricalTypes]``).
        2. Delegate the encoding computation to a backend-specific extension
           resolved via ``backend_factory``.
        3. Mark the original categorical columns as "disabled" by replacing
           their roles with ``DisableRole``. This prevents them from being
           picked up by downstream operators (e.g., ``MahalanobisDistance``,
           ``FaissNearestNeighbors``) while preserving the ability to restore
           them later via ``initial_role``.
        4. Store the encoded columns in ``ExperimentData.additional_fields``
           with appropriately derived roles (typically ``AdditionalFeatureRole``).

    Inherits from:
        Calculator: The base class for stateless calculation executors.

    Examples:
        .. code-block:: python

            # Typically used as a base class for concrete encoders
            from hypex.encoders import DummyEncoder
            encoder = DummyEncoder(target_roles=FeatureRole())
            encoded_data = encoder.execute(experiment_data)

    Args:
        target_roles (str | Sequence[str] | None, optional): The role(s) that
            identify which columns should be encoded. Defaults to ``FeatureRole()``.
        key (Any, optional): Optional identifier for the encoder instance.
            Defaults to "".

    Attributes:
        target_roles (ABCRole): The role used to search for columns to encode.
        search_types (list[type]): Data types considered categorical (``[str]``).

    See Also:
        DummyEncoder: Concrete encoder that creates one-hot encoded variables.
        DisableRole: Role assigned to original columns after encoding.
    """

    def __init__(
        self,
        target_roles: str | Sequence[str] | None = None,
        key: Any = "",
    ):
        """Initialize the encoder.

        Args:
            target_roles (str | Sequence[str] | None, optional): The role(s)
                identifying columns to encode. Defaults to ``FeatureRole()``.
            key (Any, optional): Optional identifier for the encoder instance.
                Defaults to "".
        """
        self.target_roles = target_roles or FeatureRole()
        self._key = key
        super().__init__(key)

    @property
    def __is_encoder(self):
        """Marker property used by the experiment pipeline to detect encoders.

        Returns:
            bool: Always ``True`` for encoder instances.
        """
        return True

    @property
    def search_types(self):
        """Data types considered categorical and eligible for encoding.

        Returns:
            list[type]: A list containing ``CategoricalTypes`` (``str``).
        """
        return [CategoricalTypes]

    def _get_ids(self, col_name):
        """Generate a unique executor ID for a specific encoded column.

        The ID is constructed by wrapping the column name with
        ``NAME_BORDER_SYMBOL`` delimiters, producing identifiers like
        ``┆feat_cat_B┆``.

        Args:
            col_name (str): The name of the column being encoded.

        Returns:
            str: The generated executor ID for this column.
        """
        self.key = f"{NAME_BORDER_SYMBOL}{col_name}{NAME_BORDER_SYMBOL}"
        return self.id

    def _ids_to_names(self, col_names: list[str]):
        """Map a list of column names to their corresponding executor IDs.

        Args:
            col_names (list[str]): The names of the encoded columns.

        Returns:
            dict[str, str]: A mapping ``{col_name: executor_id}`` for each
                encoded column.
        """
        return {col_name: self._get_ids(col_name) for col_name in col_names}

    @staticmethod
    def _inner_function(data: Dataset, **kwargs) -> Dataset:
        """Core encoding logic to be implemented by subclasses.

        Args:
            data (Dataset): The input dataset containing columns to encode.
            **kwargs: Additional keyword arguments (e.g., ``target_cols``).

        Returns:
            Dataset: A new dataset containing the encoded columns.

        Raises:
            AbstractMethodError: If not overridden by a subclass.
        """
        raise AbstractMethodError

    def _set_value(
        self, data: ExperimentData, value: Dataset, key=None
    ) -> ExperimentData:
        """Store the encoded columns in the experiment data.

        Saves the encoded dataset into ``ExperimentData.additional_fields``
        under the encoder's executor ID, preserving the roles of the encoded
        columns.

        Args:
            data (ExperimentData): The experiment data to update.
            value (Dataset): The dataset containing encoded columns.
            key (Any, optional): Optional key for the stored value.
                Defaults to None.

        Returns:
            ExperimentData: The updated experiment data with encoded columns
                stored in ``additional_fields``.
        """
        return data.set_value(
            space=ExperimentDataEnum.additional_fields,
            executor_id=self.id,
            value=value,
            role=value.roles,
        )

    @staticmethod
    def _disable_target_cols(data: ExperimentData, target_cols: list[str]) -> ExperimentData:
        """Mark original categorical columns as disabled.

        Replaces the roles of the specified columns with ``DisableRole``,
        preserving the original role in ``DisableRole.initial_role``. This
        prevents the original columns from being processed by downstream
        operators while keeping them available for potential restoration.

        Args:
            data (ExperimentData): The experiment data containing the columns.
            target_cols (list[str]): The names of the columns to disable.

        Returns:
            ExperimentData: The experiment data with updated roles.
        """
        disable_roles = {col:  DisableRole(initial_role=data.ds.roles[col]) for col in target_cols}
        data.ds.replace_roles(new_roles_map=disable_roles)
        return data

    def execute(self, data: ExperimentData) -> ExperimentData:
        """Execute the encoding pipeline on the experiment data.

        Orchestrates the full encoding process:
            1. Search for columns matching ``target_roles`` with categorical types.
            2. Compute the encoded representation via ``calc`` (which delegates
               to the backend-specific ``_inner_function``).
            3. Disable the original categorical columns by assigning them
               ``DisableRole``.
            4. Store the encoded columns in ``additional_fields``.

        Args:
            data (ExperimentData): The experiment data to encode.

        Returns:
            ExperimentData: The updated experiment data with encoded columns
                in ``additional_fields`` and original columns marked as
                ``DisableRole``.
        """
        target_cols = data.ds.search_columns(
            roles=self.target_roles, search_types=self.search_types
        )
        if not target_cols:
            return data
        result = self.calc(data=data.ds, target_cols=target_cols)
        data = self._disable_target_cols(data, target_cols)
        return self._set_value(
            data=data,
            value=result,
            key=self.key,
        )