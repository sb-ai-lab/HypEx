import unittest as ut
from typing import Optional, List

import pandas as pd

from hypex.dataset import ExperimentData
from hypex.experiments import Experiment
from hypex.ui.base import ExperimentShell, Output
from hypex.utils import ExperimentDataEnum
from hypex.utils.tutorial_data_creation import create_test_data


class BaseTest(ut.TestCase):
    base_data_path: Optional[str]

    def create_dataset(self):
        pass

    @staticmethod
    def _default_data_generation(**kwargs):
        return create_test_data(**kwargs)

    def setUp(self):
        self.data = (
            pd.read_csv(self.base_data_path)
            if self.base_data_path
            else create_test_data()
        )
        self.create_dataset()


class ExperimentTest(BaseTest):
    experiment: Optional[Experiment] = None

    def setUp(self):
        super().setUp()
        self.experiment_data = ExperimentData(self.data)

    def execute_experiment(self):
        self.experiment_data = self.experiment.execute(self.experiment_data)

    def check_experiment_key(self, key: str, space: ExperimentDataEnum):
        self.assertIn(key, self.experiment_data.get_spaces_key_dict()[space])


class ShellTest(ExperimentTest):
    shell_class: type
    output: Output
    shell: ExperimentShell

    def create_experiment(self, **kwargs):
        return self.shell_class.create_experiment(**kwargs)

    def execute_shell(self):
        return self.shell_class.execute(self.experiment_data)

    def test_experiment_structure(self):
        self.assertTrue(False)  # add assertion here

    def check_output_structure(self, output, attributes: List[str]):
        return self.assertTrue(
            all(
                hasattr(output, a) and output.__getattribute__(a) is not None
                for a in attributes
            )
        )

    def test_shell_output_structure(self):
        self.output = self.shell.execute(self.experiment_data)
        self.check_output_structure(self.output, ["resume"])
