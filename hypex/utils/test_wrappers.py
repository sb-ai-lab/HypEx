import unittest as ut
from typing import Optional, List

import pandas as pd

from hypex.dataset import ExperimentData
from hypex.experiments import Experiment
from hypex.ui.base import ExperimentShell
from hypex.utils import ExperimentDataEnum
from hypex.utils.tutorial_data_creation import create_test_data


class BaseTest(ut.TestCase):
    base_data_path: Optional[str] = "C:\Projects\HypEx\data.csv"

    def createDataset(self):
        pass

    def setUp(self):
        self.data = (
            pd.read_csv(self.base_data_path)
            if self.base_data_path
            else create_test_data()
        )
        self.createDataset()


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
    shell: ExperimentShell

    def create_experiment(self, **kwargs):
        return self.shell.create_experiment(**kwargs)

    def execute_shell(self):
        return self.shell.execute(self.experiment_data)

    def test_experiment_structure(self):
        self.assertEqual(True, False)  # add assertion here

    def check_output_structure(self, output, attributes: List[str]):
        return self.assertEqual(all(hasattr(output, a) for a in attributes), True)
