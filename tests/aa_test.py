from hypex import AATest
from hypex.dataset import (
    Dataset,
    InfoRole,
    TreatmentRole,
    TargetRole,
    StratificationRole,
)
from hypex.splitters import AASplitter, AASplitterWithStratification
from hypex.ui.aa import AAOutput
from hypex.utils.test_wrappers import ShellTest


class AAShellTest(ShellTest):
    shell_class = AATest
    output: AAOutput

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
    def check_default_experiment_structure(self):
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

    def check_set_experiment_structure(self, n_iterations=100, control_size=0.3):
        shell = self.shell_class(n_iterations=n_iterations, control_size=control_size)
        self.assertEqual(
            n_iterations,
            len(shell.experiment.executors[0].params[AASplitter]["random_state"]),
        )
        self.assertEqual(
            control_size,
            shell._experiment.executors[0].params[AASplitter]["control_size"][0],
        )

    def check_set_rs_experiment_structure(self, random_states=range(100)):
        shell = self.shell_class(random_states=random_states)
        self.assertEqual(
            random_states,
            shell.experiment.executors[0].params[AASplitter]["random_state"],
        )

    def check_set_stratification_experiment_structure(self):
        shell = self.shell_class(stratification=True)
        self.assertIsInstance(
            shell.experiment.executors[0].executors[0].executors[0],
            AASplitterWithStratification,
        )

    def test_experiment_structure(self):
        self.check_default_experiment_structure()
        self.check_set_experiment_structure()
        self.check_set_rs_experiment_structure()
        self.check_set_stratification_experiment_structure()

    # --------------------------------------------------------------
    # running tests--------------------------------------------------------------

    def check_resume_structure(self):
        resume = self.output.resume
        features = self.experiment_data.ds.search_columns(TargetRole())
        self.assertTrue(all(f in resume["feature"] for f in features))

    def check_experiments(self):
        n_iterations = len(
            self.shell.experiment.executors[0].params[AASplitter]["random_state"]
        )
        self.assertEqual(len(self.output.experiments), n_iterations)

    def test_shell_output_structure(self):
        self.shell = self.shell_class(n_iterations=10)
        super().test_shell_output_structure()
        self.check_output_structure(
            self.output,
            ["best_split", "experiments", "aa_score", "best_split_statistic"],
        )
        self.check_resume_structure()
        self.check_experiments()
