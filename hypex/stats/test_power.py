from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ks_2samp, norm
from sklearn.utils import shuffle
from statsmodels.stats.power import TTestIndPower

from hypex.experiment.base import Experiment

def split_splited_data(splitted_data:) -> Dict:
    """Splits a pandas DataFrame into two separate dataframes based on a specified group field.

    Args:
        splitted_data:
            The input dataframe to be split

    Returns:
        A dictionary containing two dataframes, 'test' and 'control', where 'test' contains rows where the
        group field is 'test', and 'control' contains rows where the group field is 'control'.
    """
    return {
        "control": splitted_data[splitted_data["group"] == "control"],
        "test": splitted_data[splitted_data["group"] == "test"],
    }

class StatMdeBySize(Experiment):
    mde: float

    def __init__(self, reliability: float, power: float):
        self.reliability = reliability
        self.power = power

    def execute(self):
        split_data = split_splited_data(data)
        control_group = split_data["control"]
        test_group = split_data["test"]

        m = norm.ppf(1 - (1 - reliability) / 2) + norm.ppf(power)

        n_test, n_control = len(test_group), len(control_group)
        proportion = n_test / (n_test + n_control)
        p = np.sqrt(1 / (proportion * (1 - proportion)))

        var_test, var_control = np.var(test_group, ddof=1), np.var(
            control_group, ddof=1
        )
        s = np.sqrt(var_test / n_test + var_control / n_control)

        self.mde = p * m * s
        return self.mde


class StatSampleSizeByMde(Experiment):
    sample_size: int

    def __init__(self, mde: float, significance: float, power: float):
        self.mde = mde
        self.significance = significance
        self.power = power

    def execute(self, data):
        split_data = split_splited_data(data)
        control_group = split_data["control"]
        test_group = split_data["test"]

        control_std = control_group.std()
        test_std = test_group.std()

        test_proportion = len(test_group) / (len(test_group) + len(control_group))
        control_proportion = 1 - test_proportion

        d = ((norm.ppf(1 - significance / 2) + norm.ppf(power)) / mde) ** 2
        s = test_std**2 / test_proportion + control_std**2 / control_proportion
        self.sample_size = int(d * s)
        return self.sample_size

class StatPowerByTTestInd(Experiment):
    power: float

    def __init__(self, sample_size: int, significance: float):
        self.sample_size = sample_size
        self.significance = significance

    def execute(self, data):
        split_data = split_splited_data(data)
        control_size = len(split_data["control"])
        test_size = len(split_data["test"])

        analysis = TTestIndPower()
        ratio = test_size / control_size
        return analysis.power(
            effect_size=effect_size,
            nobs1=test_size,
            ratio=ratio,
            alpha=significance,
        )
