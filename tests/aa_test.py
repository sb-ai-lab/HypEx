from hypex import AATest
from hypex.dataset import (
    Dataset,
    InfoRole,
    TreatmentRole,
    TargetRole,
    StratificationRole,
)
from hypex.splitters import AASplitter, AASplitterWithStratification
from hypex.utils.test_wrappers import ShellTest


class AAShellTest(ShellTest):
    shell_class = AATest

    def create_dataset(self):
        self.data = Dataset(
            data=self.data,
            roles={
                "user_id": InfoRole(int),
                "treat": TreatmentRole(int),
                "pre_spends": TargetRole(),
                "post_spends": TargetRole(),
                "gender": StratificationRole(str),
            },
        )

    # structure tests--------------------------------------------------------------

    def default_experiment_structure_test(self):
        shell = self.shell_class()
        self.assertEqual(
            2000, len(shell.experiment.executors[0].params[AASplitter]["random_state"])
        )
        self.assertEqual(
            0.5, shell._experiment.executors[0].params[AASplitter]["control_size"][0]
        )
        self.assertIsInstance(
            shell.experiment.executors[0].executors[0].executors[0], AASplitter
        )

    def set_experiment_structure_test(self, n_iterations=100, control_size=0.3):
        shell = self.shell_class(n_iterations=n_iterations, control_size=control_size)
        self.assertEqual(
            n_iterations,
            len(shell.experiment.executors[0].params[AASplitter]["random_state"]),
        )
        self.assertEqual(
            control_size,
            shell._experiment.executors[0].params[AASplitter]["control_size"][0],
        )

    def set_rs_experiment_structure_test(self, random_states=range(100)):
        shell = self.shell_class(random_states=random_states)
        self.assertEqual(
            random_states,
            shell.experiment.executors[0].params[AASplitter]["random_state"],
        )

    def set_stratification_experiment_structure_test(self):
        shell = self.shell_class(stratification=True)
        self.assertIsInstance(
            shell.experiment.executors[0].executors[0].executors[0],
            AASplitterWithStratification,
        )

    def test_experiment_structure(self):
        self.default_experiment_structure_test()
        self.set_experiment_structure_test()
        self.set_rs_experiment_structure_test()
        self.set_stratification_experiment_structure_test()

    # --------------------------------------------------------------
    # running tests--------------------------------------------------------------

    def test_shell_output_structure(self):
        self.shell = self.shell_class(n_iterations=10)
        super().test_shell_output_structure()
        self.check_output_structure(
            self.output,
            ["best_split", "experiments", "aa_score", "best_split_statistic"],
        )
