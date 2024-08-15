from abc import ABC
from typing import Optional, Dict

from hypex.utils import (
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
                                                which can be any defined type or None. This
                                                specifies how data should be treated or processed
                                                within the given role context.
    """

    _role_name: RoleNameType = "Abstract"

    def __init__(self, data_type: Optional[DefaultRoleTypes] = None):
        """Initializes a new instance of ABCRole with an optional data type.
        Attributes:
            data_type (Optional[DefaultRoleTypes], optional): Specifies the data type associated with the role.
                                                              This can influence how data is validated, processed,
                                                              or interpreted. Defaults to None.
        """
        self.data_type = data_type

    @property
    def role_name(self) -> str:
        """Retrieves the name of the role.
        Returns:
            RoleNameType: A string that represents the name of the role. This is typically a descriptive
                 label that identifies the role's purpose or function in data handling.
        """
        return self._role_name

    def __repr__(self) -> str:
        """Provides a string representation of the role, which includes its name and data type.
        Returns:
            str: A string representation of the role, combining the role's name with its data type,
                 formatted for clarity and easy understanding. For example, "Abstract(None)" if
                 no data type is specified.
        """
        return f"{self._role_name}({self.data_type})"


class InfoRole(ABCRole):
    """Role for information column.
    Represents informational data that doesn't directly affect the analysis but may be useful
    for reporting, debugging, or tracking purposes.
    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'Info'.
    """

    _role_name: RoleNameType = "Info"


class StratificationRole(ABCRole):
    """Role for column that need to be stratified.
    Used to ensure that different treatment groups are statistically comparable by balancing
    them based on the stratification variables. This helps in reducing bias in the treatment
    effect estimation by equalizing covariates that could influence the outcome.
    Attributes:
        data_type (Optional[CategoricalTypes]): Specifies the data type for the stratification feature.
    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'Stratification'.
    """

    _role_name: RoleNameType = "Stratification"

    def __init__(self, data_type: Optional[CategoricalTypes] = None):
        super().__init__(data_type)


class GroupingRole(ABCRole):
    """Role for column that need to be grouped by categorical variables.
    Facilitates comparisons within homogenous subgroups. This role is used to ensure that
    analyses or comparisons are made within defined categories to maintain validity and relevance,
    particularly when examining effects that may vary by group.
    Attributes:
        data_type (Optional[CategoricalTypes]): The type of the categorical data used for grouping.
    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'Grouping'.
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
    """

    _role_name: RoleNameType = "Treatment"


class TargetRole(ABCRole):
    """Role for column that shows target.
    Represents the primary outcome variable of interest in an analysis or study. Target roles
    are crucial for defining what the analysis aims to predict or explain.
    Attributes:
        data_type (Optional[TargetRoleTypes]): The type of the target data, which influences
                                               how the data is processed or analyzed.
    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'Target'.
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
    """

    _role_name: RoleNameType = "Feature"


class PreTargetRole(ABCRole):
    """Roles for column that shows Pre target.
    Represents the value of the target variable at a previous time point. This is often required
    for algorithms that need to consider historical outcomes to make predictions or assessments.
    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'PreTarget'.
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
    """

    _role_name: RoleNameType = "Statistic"


class ResumeRole(ABCRole):
    _role_name = "Resume"


class FilterRole(ABCRole):
    _role_name: RoleNameType = "Filter"


class TempTreatmentRole(TreatmentRole):
    """Role for temp treatment column.
    A temporary role used for transient treatment assignments in executables or during
    experimental processing phases. This role can help manage dynamic treatment group
    assignments that are not final but necessary for intermediate calculations.
    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'TempTreatment'.
    """

    _role_name: RoleNameType = "TempTreatment"


class TempTargetRole(TargetRole):
    """Role for temp target column.
    A temporary role for target variables used in intermediate stages of data processing
    or during certain executions in the analysis pipeline. It facilitates operations on
    target data that may not be final or are used for testing scenarios.
    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'TempTarget'.
    """

    _role_name: RoleNameType = "TempTarget"


class TempGroupingRole(GroupingRole):
    """Role for temp grouping column.
    Used temporarily to manage groups in the context of execution cursors or during
    specific phases of data manipulation. This role assists in keeping track of groups
    that are in flux or under evaluation in analytical processes.
    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'TempGrouping'.
    """

    _role_name: RoleNameType = "TempGrouping"


class DefaultRole(ABCRole):
    """The default role for a  newly created columns or any column for which another role has not been defined.
    Attributes:
        _role_name (RoleNameType): A name identifying the role as 'Default'.
    """

    _role_name: RoleNameType = "Default"


class ReportRole(ABCRole):
    """ """

    _role_name: RoleNameType = "Report"


class AdditionalRole(ABCRole):
    """ """

    _role_name: RoleNameType = "Additional"


class AdditionalTreatmentRole(AdditionalRole):
    """ """

    _role_name: RoleNameType = "AdditionalTreatment"


class AdditionalGroupingRole(AdditionalRole):
    """ """

    _role_name: RoleNameType = "AdditionalGrouping"


class AdditionalTargetRole(AdditionalRole):
    """ """

    _role_name: RoleNameType = "AdditionalTarget"


class AdditionalPreTargetRole(AdditionalRole):
    """ """

    _role_name: RoleNameType = "AdditionalPreTarget"


class AdditionalMatchingRole(AdditionalRole):

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
    "additionalpretarget": AdditionalPreTargetRole(),
}
