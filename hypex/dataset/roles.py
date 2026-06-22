from __future__ import annotations

from abc import ABC
from copy import copy

from ..utils import (
    CategoricalTypes,
    DefaultRoleTypes,
    FeatureRoleTypes,
    RoleNameType,
    TargetRoleTypes,
)


class ABCRole(ABC):
    """Abstract base class for semantic column roles in datasets.

    Defines the interface for role identification, data typing, and
    transformation. Subclasses represent specific semantic meanings
    of columns (e.g., target, feature, grouping).
    """
    _role_name: RoleNameType = "Abstract"

    def __init__(self, data_type: DefaultRoleTypes | None = None):
        """Initializes a role with an optional data type.

        Args:
            data_type: The expected Python type for the column data.
                If ``None``, the type is typically inferred from the dataset.
        """
        self.data_type = data_type

    def __copy__(self, data_type: DefaultRoleTypes | None = None):
        """Creates a shallow copy of the role with an optional updated data type.

        Args:
            data_type: The data type for the copied role. Defaults to
                the current role's data type if ``None``.

        Returns:
            A new instance of the same role class with the specified data type.
        """
        return type(self)(data_type or self.data_type)

    @property
    def role_name(self) -> str:
        """Returns the canonical name of the role.

        Returns:
            The role name string (e.g., ``'Target'``, ``'Feature'``).
        """
        return self._role_name

    def __repr__(self) -> str:
        """Returns a string representation of the role.

        Returns:
            A string in the format ``'RoleName(data_type)'``.
        """
        return f"{self._role_name}({self.data_type})"

    def astype(self, data_type: DefaultRoleTypes | None = None) -> ABCRole:
        """Creates a new role instance with the specified data type.

        Args:
            data_type: The target data type for the new role. If ``None``,
                the current data type is preserved.

        Returns:
            A new role instance with the updated data type.
        """
        role = copy(self)
        role.data_type = data_type
        return role

    def asadditional(self, data_type: DefaultRoleTypes | None = None) -> ABCRole:
        """Returns an ``Additional`` variant of the current role.

        Searches the global role registry for a class that inherits from both
        the current role's class and ``AdditionalRole``. If found, returns a
        new instance of that class. Otherwise, returns a copy of the current role.

        Args:
            data_type: The data type for the returned role. Defaults to
                the current role's data type if ``None``.

        Returns:
            An ``AdditionalRole`` variant or a copy of the current role.
        """
        data_type = data_type or self.data_type
        for role_type in list(default_roles.values()):
            if isinstance(role_type, self.__class__) and isinstance(
                role_type, AdditionalRole
            ):
                return role_type.__class__(data_type)
        return self.__class__(data_type)


class LagRole(ABCRole):
    """Base class for roles that support temporal metadata (parent, lag).

    Used for time-series or panel data where columns represent lagged
    values or have temporal dependencies on other fields.
    """

    def __init__(
        self,
        data_type: DefaultRoleTypes | None = None,
        parent: str | None = None,
        lag: int | None = None,
    ):
        """Initializes a lag role with temporal metadata.

        Args:
            data_type: The expected Python type for the column data.
            parent: The name of the original/parent column this lag derives from.
            lag: The number of periods this column is lagged by.
        """
        super().__init__(data_type)
        self.parent = parent
        self.lag = lag

    def __repr__(self) -> str:
        """Returns a string representation including temporal metadata.

        Returns:
            A formatted string containing the role name and any non-``None``
            attributes (``data_type``, ``parent``, ``lag``).
        """
        parts = []
        if self.data_type is not None:
            parts.append(f"data_type={self.data_type}")
        if self.parent is not None:
            parts.append(f"parent='{self.parent}'")
        if self.lag is not None:
            parts.append(f"lag={self.lag}")
        return (
            f"{self._role_name}({', '.join(parts)})"
            if parts
            else f"{self._role_name}()"
        )


class IndexRole(ABCRole):
    """Semantic role for index or identifier columns."""
    _role_name: RoleNameType = "Index"


class InfoRole(ABCRole):
    """Semantic role for auxiliary or informational columns not used in statistical analysis."""
    _role_name: RoleNameType = "Info"

class DisableRole(ABCRole):
    """Semantic role for columns that have been "disabled" after preprocessing."""
    _role_name: RoleNameType = "Disable"

    def __init__(self, initial_role: ABCRole | None = None, data_type = None):
        super().__init__(
            initial_role.data_type 
            if initial_role 
            else data_type
        )
        self._initial_role = initial_role

    @property
    def initial_role(self):
        return self._initial_role

class StratificationRole(ABCRole):
    """Semantic role for columns used to stratify data during sampling or splitting.

    Typically represents categorical variables that ensure balanced group
    representation across experimental splits.
    """
    _role_name: RoleNameType = "Stratification"

    def __init__(self, data_type: CategoricalTypes | None = None):
        super().__init__(data_type)


class GroupingRole(ABCRole):
    """Semantic role for columns that define logical groups or segments in the dataset.

    Used in group-by operations, matching algorithms, or comparative analysis.
    """
    _role_name: RoleNameType = "Grouping"

    def __init__(self, data_type: CategoricalTypes | None = None):
        """Initializes a grouping role.

        Args:
            data_type: The expected categorical data type.
        """
        super().__init__(data_type)


class TreatmentRole(ABCRole):
    """Semantic role for columns indicating treatment assignment or group membership."""
    _role_name: RoleNameType = "Treatment"


class TargetRole(ABCRole):
    """Semantic role for primary outcome or target variable columns.

    Represents the metric being analyzed, predicted, or compared in an experiment.
    """
    _role_name: RoleNameType = "Target"

    def __init__(
        self,
        data_type: TargetRoleTypes | None = None,
        cofounders: list[str] | None = None,
    ):
        """Initializes a target role with optional confounder references.

        Args:
            data_type: The expected Python type for the target variable.
            cofounders: List of column names identified as confounders
                affecting the target variable.
        """
        super().__init__(data_type=data_type)
        self.cofounders = cofounders if cofounders is not None else []


class FeatureRole(LagRole):
    """Semantic role for independent variable or predictor columns.

    Inherits from ``LagRole`` to support temporal feature engineering
    and lag-based dependencies.
    """
    _role_name: RoleNameType = "Feature"

    def __init__(
        self,
        data_type: FeatureRoleTypes | None = None,
        parent: str | None = None,
        lag: int | None = None,
    ):
        """Initializes a feature role.

        Args:
            data_type: The expected Python type for the feature.
            parent: Name of the parent column if this is a derived/lagged feature.
            lag: Number of periods lagged, if applicable.
        """
        super().__init__(data_type=data_type, parent=parent, lag=lag)


class PreTargetRole(LagRole):
    """Semantic role for pre-treatment or baseline target measurements.

    Used to adjust for baseline differences or apply CUPED-like variance reduction.
    Inherits from ``LagRole`` to handle temporal dependencies.
    """
    _role_name: RoleNameType = "PreTarget"

    def __init__(
        self,
        data_type: TargetRoleTypes | None = None,
        parent: str | None = None,
        lag: int | None = None,
        cofounders: list[str] | None = None,
    ):
        """Initializes a pre-target role.

        Args:
            data_type: The expected Python type for the pre-target variable.
            parent: Name of the original target column.
            lag: Number of periods lagged.
            cofounders: List of column names identified as confounders.
        """
        super().__init__(data_type=data_type, parent=parent, lag=lag)
        self.cofounders = cofounders if cofounders is not None else []


class StatisticRole(ABCRole):
    """Semantic role for columns containing computed statistics or aggregated results."""
    _role_name: RoleNameType = "Statistic"


class ResumeRole(ABCRole):
    """Semantic role for columns containing summary or report-level metrics."""
    _role_name = "Resume"


class FilterRole(ABCRole):
    """Semantic role for boolean mask columns used for row filtering."""
    _role_name: RoleNameType = "Filter"


class ConstGroupRole(ABCRole):
    """Semantic role for columns defining fixed, non-randomized group assignments."""
    _role_name: RoleNameType = "ConstGroup"


# ___________________________________________________________________________________________
class TempRole(ABCRole):
    """Base semantic role for temporary or transient columns during intermediate data processing."""
    _role_name: RoleNameType = "Temp"


class TempTreatmentRole(TempRole, TreatmentRole):
    """Temporary role for treatment columns during intermediate processing steps."""
    _role_name: RoleNameType = "TempTreatment"


class TempTargetRole(TempRole, TargetRole):
    """Temporary role for target columns during intermediate processing steps."""
    _role_name: RoleNameType = "TempTarget"


class TempGroupingRole(TempRole, GroupingRole):
    """Temporary role for grouping columns during intermediate processing steps."""
    _role_name: RoleNameType = "TempGrouping"


class DefaultRole(ABCRole):
    """Fallback semantic role assigned to columns with unspecified or unknown meanings."""
    _role_name: RoleNameType = "Default"


class ReportRole(ABCRole):
    """Semantic role for columns specifically intended for reporting or output generation."""
    _role_name: RoleNameType = "Report"


# ___________________________________________________________________________________________
class AdditionalRole(ABCRole):
    """Base semantic role for derived, auxiliary, or supplementary columns."""
    _role_name: RoleNameType = "Additional"


class AdditionalTreatmentRole(AdditionalRole, TreatmentRole):
    """Derived role for supplementary treatment assignment or group indicator columns."""
    _role_name: RoleNameType = "AdditionalTreatment"


class AdditionalGroupingRole(AdditionalRole, GroupingRole):
    """Derived role for supplementary grouping or segmentation columns."""
    _role_name: RoleNameType = "AdditionalGrouping"


class AdditionalTargetRole(AdditionalRole, TargetRole):
    """Derived role for auxiliary or transformed target variable columns."""
    _role_name: RoleNameType = "AdditionalTarget"


class AdditionalFeatureRole(AdditionalRole, FeatureRole):
    """Derived role for supplementary or engineered predictor columns."""
    _role_name: RoleNameType = "AdditionalFeature"


class AdditionalPreTargetRole(AdditionalRole, PreTargetRole):
    """Derived role for supplementary pre-treatment or baseline measurement columns."""
    _role_name: RoleNameType = "AdditionalPreTarget"


class AdditionalMatchingRole(AdditionalRole):
    """Derived role for columns storing matching indices or nearest-neighbor results."""
    _role_name: RoleNameType = "AdditionalMatching"
    
class AdditionalVarianceReductionRole(AdditionalRole, StatisticRole):
    """For CUPED variance reductions."""
    _role_name: RoleNameType = "AdditionalVarianceReduction"

class AdditionalStatisticRole(AdditionalRole, StatisticRole):
    _role_name: RoleNameType = "AdditionalStatistic"


default_roles: dict[RoleNameType, ABCRole] = {
    "info": InfoRole(),
    "default": DefaultRole(),
    "feature": FeatureRole(),
    "treatment": TreatmentRole(),
    "grouping": GroupingRole(),
    "target": TargetRole(),
    "pretarget": PreTargetRole(),
    "stratification": StratificationRole(),
    "statistic": StatisticRole(),
    "index": IndexRole(),
    "filter": FilterRole(),
    "constgroup": ConstGroupRole(),
    "additionaltreatment": AdditionalTreatmentRole(),
    "additionalgrouping": AdditionalGroupingRole(),
    "additionaltarget": AdditionalTargetRole(),
    "additionalfeature": AdditionalFeatureRole(),
    "additionalpretarget": AdditionalPreTargetRole(),
    "additionalvariancereduction": AdditionalVarianceReductionRole(),
}
