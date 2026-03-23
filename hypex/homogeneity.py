from .analyzers.aa import OneAAStatAnalyzer
from .comparators import GroupChi2Test, GroupDifference, GroupSizes, GroupKSTest, GroupTTest
from .dataset import TargetRole, TreatmentRole
from .experiments.base import Experiment, OnRoleExperiment
from .ui.base import ExperimentShell
from .ui.homo import HomoOutput

HOMOGENEITY_TEST = Experiment(
    executors=[
        OnRoleExperiment(
            executors=[
                GroupSizes(grouping_role=TreatmentRole(), compare_by="groups"),
                GroupDifference(grouping_role=TreatmentRole(), compare_by="groups"),
                GroupTTest(grouping_role=TreatmentRole(), compare_by="groups"),
                GroupKSTest(grouping_role=TreatmentRole(), compare_by="groups"),
                GroupChi2Test(grouping_role=TreatmentRole(), compare_by="groups"),
            ],
            role=TargetRole(),
        ),
        OneAAStatAnalyzer(),
    ]
)


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

    Examples
    --------
    .. code-block:: python

        # Basic homogeneity test
        homo_test = HomogeneityTest()
        results = homo_test.execute(data)

        # Accessing specific test results
        homo_test = HomogeneityTest()
        results = homo_test.execute(data)
        output = results.resume

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

    def __init__(self):
        """Initialize HomogeneityTest with default experiment and output configurations."""
        super().__init__(
            experiment=HOMOGENEITY_TEST,
            output=HomoOutput(),
        )
