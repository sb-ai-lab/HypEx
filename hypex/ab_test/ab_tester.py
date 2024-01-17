"""Module for AB tests"""
from typing import Dict
import pandas as pd
from IPython.display import display
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu


class ABTest:
    """
       A class for conducting A/B testing using statistical methods.

       This class provides methods for splitting data into test and control groups,
       calculating various metrics to compare these groups, and computing p-values
       to assess the statistical significance of the observed differences.

       Attributes:
           calc_difference_method (str): Method used for calculating the difference
               between test and control groups. Options include 'all' (default),
               'ate' (average treatment effect), 'diff_in_diff' (difference in
               differences), and 'cuped' (Controlled-Experiment using Pre-Experiment
               Data).
           calc_p_value_method (str): Method used for calculating the p-value.
               Options include 'all' (default), 't-test', and 'mann_whitney'.
           results (dict or None): Stores the results of the executed tests. Each
               key (like 'size', 'difference', 'p-value') maps to a corresponding
               result.

       Methods:
           split_ab(data, group_field):
               Splits a DataFrame into test and control groups based on a group field.
           cuped(test_data, control_data, target_field, target_field_before):
               Calculates the Controlled-Experiment using Pre-Experiment Data (CUPED)
               metric.
           diff_in_diff(test_data, control_data, target_field, target_field_before):
               Computes the Difference in Differences (DiD) metric.
           calc_difference(splitted_data, target_field, target_field_before):
               Calculates the difference in the target field between test and control.
           calc_p_value(splitted_data, target_field):
               Calculates the p-value for the difference between test and control groups.
           execute(data, target_field, group_field, target_field_before):
               Executes the A/B test, splitting the data and calculating size,
               difference, and p-value.
           show_beautiful_result():
               Displays the results in an easy-to-read format.

       Example:
        >>> model = ABTest()
        >>> results = model.execute(
        >>>     data=data_ab,
        >>>     target_field='post_spends',
        >>>     target_field_before='pre_spends',
        >>>     group_field='group'
        >>>)
        >>> results
       """

    def __init__(
            self,
            calc_difference_method: str = "all",
            calc_p_value_method: str = "all",
    ):
        """Initializes the ABTest class.

        Args:
            calc_difference_method:
                The method used to calculate the difference:
                    'all' [default] - all metrics
                    'ate' - basic difference in means of targets in test and control group
                    'diff_in_diff' - difference in difference value,
                                     performs pre-post analysis (required values of target before pilot)
                    'cuped' - Controlled-Experiment using Pre-Experiment Data value,
                              performs pre-post analysis (required values of target before pilot)
            calc_p_value_method:
                The method used to calculate the p-value. Defaults to 'all'
        """
        self.calc_difference_method = calc_difference_method
        self.calc_p_value_method = calc_p_value_method
        self.results = None

    @staticmethod
    def split_ab(data: pd.DataFrame, group_field: str) -> Dict[str, pd.DataFrame]:
        """Splits a pandas DataFrame into two separate dataframes based on a specified group field.

        Args:
            data:
                The input dataframe to be split
            group_field:
                The column name representing the group field

        Returns:
            A dictionary containing two dataframes, 'test' and 'control', where 'test' contains rows where the
            group field is 'test', and 'control' contains rows where the group field is 'control'.
        """
        return {
            "test": data[data[group_field] == "test"],
            "control": data[data[group_field] == "control"],
        }

    @staticmethod
    def cuped(
            test_data: pd.DataFrame,
            control_data: pd.DataFrame,
            target_field: str,
            target_field_before: str,
    ) -> float:
        """Counts CUPED (Controlled-Experiment using Pre-Experiment Data) in absolute values.

        Metric uses pre-post analysis of target, uses to minimize variance of effect:
        ATE = mean(test_cuped) - mean(control_cuped)
        , where
            test_cuped = target__test - theta * target_before__test
            control_cuped = target__control - theta * target_before__control
                , where
                theta = (cov_test + cov_control) / (var_test + var_control)
                    , where
                    cov_test = cov(target__test, target_before__test)
                    cov_control = cov(target__control, target_before__control)
                    var_test = var(target_before__test)
                    var_control = var(target_before__control)

        Args:
            test_data:
                Input data of test group
                Should include target before and after pilot
            control_data:
                Input data of control group
                Should include target before and after pilot
            target_field:
                Column name of target after pilot
            target_field_before:
                Column name of target before pilot

        Returns:
                Named tuple with p-value, effect, ci_length, left_bound and right_bound
        """
        control = control_data[target_field]
        control_before = control_data[target_field_before]
        test = test_data[target_field]
        test_before = test_data[target_field_before]

        theta = (
                        np.cov(control, control_before)[0, 1] + np.cov(test, test_before)[0, 1]
                ) / (np.var(control_before) + np.var(test_before))

        control_cuped = control - theta * control_before
        test_cuped = test - theta * test_before

        mean_control = np.mean(control_cuped)
        mean_test = np.mean(test_cuped)

        return mean_test - mean_control

    @staticmethod
    def diff_in_diff(
            test_data: pd.DataFrame,
            control_data: pd.DataFrame,
            target_field: str,
            target_field_before: str,
    ) -> float:
        """Counts Difference in Difference.

        Metric uses pre-post analysis and counts difference in means in data before and after pilot:
        ATE = (y_test_after - y_control_after) - (y_test_before - y_control_before)

        Args:
            test_data:
                Input data of test group
            control_data:
                Input data of control group
            target_field:
                Column name of target after pilot
            target_field_before:
                Column name of target before pilot

        Returns:
            Value of difference in difference
        """
        mean_test = np.mean(test_data[target_field])
        mean_control = np.mean(control_data[target_field])

        mean_test_before = np.mean(test_data[target_field_before])
        mean_control_before = np.mean(control_data[target_field_before])
        return (mean_test - mean_control) - (mean_test_before - mean_control_before)

    def calc_difference(
            self,
            splitted_data: Dict[str, pd.DataFrame],
            target_field: str,
            target_field_before: str = None,
    ) -> Dict[str, float]:
        """Calculates the difference between the target field values of the 'test' and 'control' dataframes.

        Args:
            splitted_data:
                A dictionary containing the 'test' and 'control' dataframes
            target_field:
                The name of the target field contains data after pilot
            target_field_before:
                The name of the target field contains data before pilot

        Returns:
            A dictionary containing the difference between the target field
            values of the 'test' and 'control' dataframes
        """
        result = {}
        if (
                self.calc_difference_method in {"all", "diff_in_diff", "cuped"}
                and target_field_before is None
        ):
            raise ValueError(
                "For calculation metrics 'cuped' or 'diff_in_diff' field 'target_field_before' is required.\n"
                "Metric 'ate'(=diff-in-means) can be used without 'target_field_before'"
            )
        if self.calc_difference_method in {"all", "ate"}:
            result["ate"] = (
                    splitted_data["test"][target_field].values
                    - splitted_data["control"][target_field].values
            ).mean()

        if self.calc_difference_method in {"all", "cuped"}:
            result["cuped"] = self.cuped(
                test_data=splitted_data["test"],
                control_data=splitted_data["control"],
                target_field=target_field,
                target_field_before=target_field_before,
            )

        if self.calc_difference_method in {"all", "diff_in_diff"}:
            result["diff_in_diff"] = self.diff_in_diff(
                test_data=splitted_data["test"],
                control_data=splitted_data["control"],
                target_field=target_field,
                target_field_before=target_field_before,
            )

        return result

    def calc_p_value(
            self, splitted_data: Dict[str, pd.DataFrame], target_field: str
    ) -> Dict[str, float]:
        """Calculates the p-value for a given dataset.

        Args:
            splitted_data:
                A dictionary containing the split data, where the keys are 'test' and 'control'
                and the values are pandas DataFrames
            target_field:
                The name of the target field
        Returns:
            A dictionary containing the calculated p-values, where the keys are 't-test' and 'mann_whitney'
            and the values are the corresponding p-values
        """
        result = {}
        if self.calc_p_value_method in {"all", "t-test"}:
            result["t-test"] = ttest_ind(
                splitted_data["test"][target_field],
                splitted_data["control"][target_field],
            ).pvalue

        if self.calc_p_value_method in {"all", "mann_whitney"}:
            result["mann_whitney"] = mannwhitneyu(
                splitted_data["test"][target_field],
                splitted_data["control"][target_field],
            ).pvalue

        return result

    def execute(
            self,
            data: pd.DataFrame,
            target_field: str,
            group_field: str,
            target_field_before: str = None,
    ) -> Dict[str, Dict[str, float]]:
        """Splits the input data based on the group field and calculates the size, difference, and p-value.

        Parameters:
            data:
                Input data as a pandas DataFrame
            target_field:
                Target field to be analyzed
            group_field:
                Field used to split the data into groups
            target_field_before:
                Target field without treatment to be analyzed

        Returns:
            A dictionary containing the size, difference, and p-value of the split data
                'size': A dictionary with the sizes of the test and control groups
                'difference': A dictionary with the calculated differences between the groups
                'p-value': A dictionary with the calculated p-values for each group
        """
        splitted_data = self.split_ab(data, group_field)

        results = {
            "size": {
                "test": len(splitted_data["test"]),
                "control": len(splitted_data["control"]),
            },
            "difference": self.calc_difference(
                splitted_data, target_field, target_field_before
            ),
            "p-value": self.calc_p_value(splitted_data, target_field),
        }

        self.results = results

        return results

    def show_beautiful_result(self):
        """Shows results of 'execute' function - dict as dataframes."""
        for k in self.results.keys():
            display(pd.DataFrame(self.results[k], index=[k]).T)
