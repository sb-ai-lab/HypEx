from abc import ABC
from typing import Optional, Dict, Union
from copy import deepcopy

from ..utils import (
    TargetRoleTypes,
    DefaultRoleTypes,
    CategoricalTypes,
    RoleNameType,
)


class ABCRole(ABC):
    """Base abstract class for all roles.

    Base class for defining different roles within a data set. Each role corresponds to a specific
    type of data handling or processing functionality. This class is abstract and should be
    subclassed to implement specific roles.

    Attributes:
        _role_name (RoleNameType): The default name of the role, intended to be overridden in subclasses.
        data_type (Optional[DefaultRoleTypes]): The type of data associated with this role,
            which can be any defined type or None. This specifies how data should be treated 
            or processed within the given role context.

    Examples:
        Create a custom role by subclassing ABCRole:
        >>> class CustomRole(ABCRole):
        ...     _role_name = "Custom"
        ...     def __init__(self, data_type=None):
        ...         super().__init__(data_type)
        
        >>> role = CustomRole(data_type="float64")
        >>> print(role.role_name)
        Custom
        >>> print(role.data_type)
        float64
    """

    _role_name: RoleNameType = "Abstract"

    def __init__(self, data_type: Optional[DefaultRoleTypes] = None):
        """Initializes a new instance of ABCRole with an optional data type.

        Args:
            data_type (Optional[DefaultRoleTypes], optional): Specifies the data type associated 
                with the role. This can influence how data is validated, processed, or interpreted. 
                Defaults to None.

        Examples:
            >>> role = ABCRole(data_type="int64")
            >>> print(role.data_type)
            int64
        """
        self.data_type = data_type

    @property
    def role_name(self) -> str:
        """Retrieves the name of the role.

        Returns:
            str: A string that represents the name of the role. This is typically a descriptive
                label that identifies the role's purpose or function in data handling.

        Examples:
            >>> role = ABCRole()
            >>> print(role.role_name)
            Abstract
        """
        return self._role_name

    def __repr__(self) -> str:
        """Provides a string representation of the role, which includes its name and data type.

        Returns:
            str: A string representation of the role, combining the role's name with its data type,
                formatted for clarity and easy understanding. For example, "Abstract(None)" if
                no data type is specified.

        Examples:
            >>> role = ABCRole(data_type="float64")
            >>> print(repr(role))
            Abstract(float64)
        """
        return f"{self._role_name}({self.data_type})"


class InfoRole(ABCRole):
    """Role for information column.

    Represents informational data that doesn't directly affect the analysis but may be useful
    for reporting, debugging, or tracking purposes.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'Info'.

    Examples:
        >>> info_role = InfoRole(data_type="object")
        >>> print(info_role.role_name)
        Info
        >>> print(info_role.data_type)
        object
    """

    _role_name: RoleNameType = "Info"


class StratificationRole(ABCRole):
    """Role for column that need to be stratified.

    Used to ensure that different treatment groups are statistically comparable by balancing
    them based on the stratification variables. This helps in reducing bias in the treatment
    effect estimation by equalizing covariates that could influence the outcome.

    Args:
        data_type (Optional[CategoricalTypes]): Specifies the data type for the stratification feature.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'Stratification'.

    Examples:
        >>> strat_role = StratificationRole(data_type="category")
        >>> print(strat_role.role_name)
        Stratification
        >>> print(strat_role.data_type)
        category
    """

    _role_name: RoleNameType = "Stratification"

    def __init__(self, data_type: Optional[CategoricalTypes] = None):
        super().__init__(data_type)


class GroupingRole(ABCRole):
    """Role for column that need to be grouped by categorical variables.

    Facilitates comparisons within homogenous subgroups. This role is used to ensure that
    analyses or comparisons are made within defined categories to maintain validity and relevance,
    particularly when examining effects that may vary by group.

    Args:
        data_type (Optional[CategoricalTypes]): The type of the categorical data used for grouping.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'Grouping'.

    Examples:
        >>> group_role = GroupingRole(data_type="category") 
        >>> print(group_role.role_name)
        Grouping
        >>> print(group_role.data_type)
        category
    """

    _role_name: RoleNameType = "Grouping"

    def __init__(self, data_type: Optional[CategoricalTypes] = None):
        super().__init__(data_type)


class TreatmentRole(ABCRole):
    """Role for column that shows treatment.

    Designates the treatment assignment of subjects in an experimental study. This role is critical
    for identifying which group a subject belongs to, facilitating the assessment of the treatment
    effects across different conditions.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'Treatment'.

    Examples:
        >>> treat_role = TreatmentRole()
        >>> print(treat_role.role_name)
        Treatment
    """

    _role_name: RoleNameType = "Treatment"


class TargetRole(ABCRole):
    """Role for column that shows target.

    Represents the primary outcome variable of interest in an analysis or study. Target roles
    are crucial for defining what the analysis aims to predict or explain.

    Args:
        data_type (Optional[TargetRoleTypes]): The type of the target data, which influences
            how the data is processed or analyzed.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'Target'.

    Examples:
        >>> target_role = TargetRole(data_type="continuous")
        >>> print(target_role.role_name)
        Target
        >>> print(target_role.data_type)
        continuous
    """

    _role_name: RoleNameType = "Target"

    def __init__(self, data_type: Optional[TargetRoleTypes] = None):
        super().__init__(data_type)


class FeatureRole(ABCRole):
    """Roles for column that shows feature.

    Used for features that directly influence the outcome of the analysis. These are key variables
    that algorithms use to learn patterns or make decisions.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'Feature'.

    Examples:
        >>> feature_role = FeatureRole()
        >>> print(feature_role.role_name)
        Feature
    """

    _role_name: RoleNameType = "Feature"


class PreTargetRole(ABCRole):
    """Roles for column that shows Pre target.

    Represents the value of the target variable at a previous time point. This is often required
    for algorithms that need to consider historical outcomes to make predictions or assessments.

    Args:
        data_type (Optional[TargetRoleTypes]): The type of the pre-target data.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'PreTarget'.

    Examples:
        >>> pre_target_role = PreTargetRole(data_type="continuous")
        >>> print(pre_target_role.role_name)
        PreTarget
        >>> print(pre_target_role.data_type)
        continuous
    """

    _role_name: RoleNameType = "PreTarget"

    def __init__(self, data_type: Optional[TargetRoleTypes] = None):
        super().__init__(data_type)


class StatisticRole(ABCRole):
    """Role for statistic column.

    Designated for columns that are used to compute statistical parameters or metrics.
    This role facilitates the identification and processing of data that contributes to
    statistical analysis, ensuring that relevant calculations are performed accurately.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'Statistic'.

    Examples:
        >>> stat_role = StatisticRole()
        >>> print(stat_role.role_name)
        Statistic
    """

    _role_name: RoleNameType = "Statistic"


class ResumeRole(ABCRole):
    """Role for resume column.

    Used for columns that contain summary or aggregated information.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'Resume'.

    Examples:
        >>> resume_role = ResumeRole()
        >>> print(resume_role.role_name)
        Resume
    """
    _role_name = "Resume"


class FilterRole(ABCRole):
    """Role for filter column.

    Used for columns that contain filtering criteria or conditions.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'Filter'.

    Examples:
        >>> filter_role = FilterRole()
        >>> print(filter_role.role_name)
        Filter
    """
    _role_name: RoleNameType = "Filter"


class ConstGroupRole(ABCRole):
    """Role for constant group column.

    Used for columns that contain constant or unchanging group identifiers.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'ConstGroup'.

    Examples:
        >>> const_group_role = ConstGroupRole()
        >>> print(const_group_role.role_name)
        ConstGroup
    """
    _role_name: RoleNameType = "ConstGroup"


# ___________________________________________________________________________________________
class TempRole(ABCRole):
    """Base role for temporary columns.

    Used as a base class for all temporary roles in the system.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'Temp'.

    Examples:
        >>> temp_role = TempRole()
        >>> print(temp_role.role_name)
        Temp
    """
    _role_name: RoleNameType = "Temp"


class TempTreatmentRole(TempRole, TreatmentRole):
    """Role for temp treatment column.

    A temporary role used for transient treatment assignments in executables or during
    experimental processing phases. This role can help manage dynamic treatment group
    assignments that are not final but necessary for intermediate calculations.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'TempTreatment'.

    Examples:
        >>> temp_treat_role = TempTreatmentRole()
        >>> print(temp_treat_role.role_name)
        TempTreatment
    """

    _role_name: RoleNameType = "TempTreatment"


class TempTargetRole(TempRole, TargetRole):
    """Role for temp target column.

    A temporary role for target variables used in intermediate stages of data processing
    or during certain executions in the analysis pipeline. It facilitates operations on
    target data that may not be final or are used for testing scenarios.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'TempTarget'.

    Examples:
        >>> temp_target_role = TempTargetRole()
        >>> print(temp_target_role.role_name)
        TempTarget
    """

    _role_name: RoleNameType = "TempTarget"


class TempGroupingRole(TempRole, GroupingRole):
    """Role for temp grouping column.

    Used temporarily to manage groups in the context of execution cursors or during
    specific phases of data manipulation. This role assists in keeping track of groups
    that are in flux or under evaluation in analytical processes.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'TempGrouping'.

    Examples:
        >>> temp_group_role = TempGroupingRole()
        >>> print(temp_group_role.role_name)
        TempGrouping
    """

    _role_name: RoleNameType = "TempGrouping"


class DefaultRole(ABCRole):
    """The default role for newly created columns or any column for which another role has not been defined.

    This role is assigned automatically to columns that don't have a specific role designation.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'Default'.

    Examples:
        >>> default_role = DefaultRole()
        >>> print(default_role.role_name)
        Default
    """

    _role_name: RoleNameType = "Default"


class ReportRole(ABCRole):
    """Role for report columns.

    Used for columns that contain information specifically formatted or intended for reporting purposes.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'Report'.

    Examples:
        >>> report_role = ReportRole()
        >>> print(report_role.role_name)
        Report
    """

    _role_name: RoleNameType = "Report"


# ___________________________________________________________________________________________
class AdditionalRole(ABCRole):
    """Base role for additional columns.

    Used as a base class for all additional roles that extend the basic role functionality.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'Additional'.

    Examples:
        >>> additional_role = AdditionalRole()
        >>> print(additional_role.role_name)
        Additional
    """

    _role_name: RoleNameType = "Additional"


class AdditionalTreatmentRole(AdditionalRole):
    """Role for additional treatment columns.

    Used for supplementary treatment-related columns that extend beyond the primary treatment designation.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'AdditionalTreatment'.

    Examples:
        >>> additional_treat_role = AdditionalTreatmentRole()
        >>> print(additional_treat_role.role_name)
        AdditionalTreatment
    """

    _role_name: RoleNameType = "AdditionalTreatment"


class AdditionalGroupingRole(AdditionalRole):
    """Role for additional grouping columns.

    Used for supplementary grouping columns that provide additional categorization beyond primary grouping.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'AdditionalGrouping'.

    Examples:
        >>> additional_group_role = AdditionalGroupingRole()
        >>> print(additional_group_role.role_name)
        AdditionalGrouping
    """

    _role_name: RoleNameType = "AdditionalGrouping"


class AdditionalTargetRole(AdditionalRole):
    """Role for additional target columns.

    Used for supplementary target variables that are of interest beyond the primary target.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'AdditionalTarget'.

    Examples:
        >>> additional_target_role = AdditionalTargetRole()
        >>> print(additional_target_role.role_name)
        AdditionalTarget
    """

    _role_name: RoleNameType = "AdditionalTarget"


class AdditionalPreTargetRole(AdditionalRole):
    """Role for additional pre-target columns.

    Used for supplementary pre-target variables that provide historical context beyond the primary pre-target.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'AdditionalPreTarget'.

    Examples:
        >>> additional_pre_target_role = AdditionalPreTargetRole()
        >>> print(additional_pre_target_role.role_name)
        AdditionalPreTarget
    """

    _role_name: RoleNameType = "AdditionalPreTarget"


class AdditionalMatchingRole(AdditionalRole):
    """Role for additional matching columns.

    Used for supplementary columns that assist in matching or pairing observations.

    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'AdditionalMatching'.

    Examples:
        >>> additional_matching_role = AdditionalMatchingRole()
        >>> print(additional_matching_role.role_name)
        AdditionalMatching
    """

    _role_name: RoleNameType = "AdditionalMatching"


default_roles: Dict[RoleNameType, ABCRole] = {
    "info": InfoRole(),
    "default": DefaultRole(),
    "feature": FeatureRole(),
    "treatment": TreatmentRole(),
    "grouping": GroupingRole(),
    "target": TargetRole(),
    "pretarget": PreTargetRole(),
    "stratification": StratificationRole(),
    "statistic": StatisticRole(),
    "filter": FilterRole(),
    "additionaltreatment": AdditionalTreatmentRole(),
    "additionalgrouping": AdditionalGroupingRole(),
    "additionaltarget": AdditionalTargetRole(),
    "additionalpretarget": AdditionalPreTargetRole(),
}
