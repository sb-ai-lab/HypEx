from hypex import AATest
from hypex.dataset import (
    Dataset,
    InfoRole,
    TreatmentRole,
    TargetRole,
    StratificationRole,
)
from hypex.splitters import AASplitter
from hypex.utils.test_wrappers import ShellTest


class AAShellTest(ShellTest):
    shell = AATest

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

    def default_structure_test(self):
        shell = self.shell()
        self.assertEqual(
            2000, len(shell.experiment.executors[0].params[AASplitter]["random_state"])
        )
        self.assertEqual(
            0.5, shell._experiment.executors[0].params[AASplitter]["control_size"][0]
        )

    def test_experiment_structure(self):
        self.default_structure_test()

    def common_test(self):
        result = self.shell()
