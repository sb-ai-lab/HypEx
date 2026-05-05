from .analyzers.aa import OneAAStatAnalyzer
from .comparators import Chi2Test, GroupDifference, GroupSizes, KSTest, TTest
from .dataset import ABCRole, TargetRole, TreatmentRole, FeatureRole
from .experiments.base import Experiment, OnRoleExperiment
from .ui.base import ExperimentShell
from .ui.homo import HomoOutput


def _build_homogeneity_experiment(role: ABCRole) -> Experiment:
    return Experiment(
        executors=[
            OnRoleExperiment(
                executors=[
                    GroupSizes(grouping_role=TreatmentRole(), compare_by="groups"),
                    GroupDifference(grouping_role=TreatmentRole(), compare_by="groups"),
                    TTest(grouping_role=TreatmentRole(), compare_by="groups"),
                    KSTest(grouping_role=TreatmentRole(), compare_by="groups"),
                    Chi2Test(grouping_role=TreatmentRole(), compare_by="groups"),
                ],
                role=role,
            ),
            OneAAStatAnalyzer(),
        ]
    )


HOMOGENEITY_TEST = _build_homogeneity_experiment(TargetRole())


class HomogeneityTest(ExperimentShell):
    """A class for conducting homogeneity tests between the groups.

    This class provides functionality to test whether treatment and control groups are
    homogeneous across target variables using multiple statistical tests including t-test,
    Kolmogorov-Smirnov test, and chi-square test.

    The class runs the following analyses:
        - Group size comparisons
        - Group differences
        - T-test for continuous variables
        - KS-test for distribution comparisons
        - Chi-square test for categorical variables
        - AA statistics analysis

    Args:
        role (ABCRole, optional): The dataset role to run the homogeneity test on.
            Defaults to TargetRole(). Pass any ABCRole subclass (e.g. FeatureRole())
            to test columns with that role instead.

    Examples
    --------
    .. code-block:: python

        # Basic homogeneity test on target columns
        homo_test = HomogeneityTest()
        results = homo_test.execute(data)

        # Run on feature columns instead
        from hypex.dataset import FeatureRole
        homo_test = HomogeneityTest(role=FeatureRole())
        results = homo_test.execute(data)

        # Running test on dataset with roles
        from hypex.dataset import Dataset, TargetRole, TreatmentRole
        ds = Dataset(
            roles={
                'treatment': TreatmentRole(),
                'outcome': TargetRole()
            },
            data=df
        )
        homo_test = HomogeneityTest()
        results = homo_test.execute(ds)
    """

    def __init__(self, role: ABCRole | None = None):
        """Initialize HomogeneityTest with an optional role override."""
        experiment = (
            HOMOGENEITY_TEST if role is None else _build_homogeneity_experiment(role)
        )
        super().__init__(
            experiment=experiment,
            output=HomoOutput(),
        )
