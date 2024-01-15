"""Module for AA tests"""
import warnings
from itertools import combinations
from pathlib import Path
from typing import Iterable, Union, Optional, Dict, Any, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind, ks_2samp, norm
from sklearn.utils import shuffle
from statsmodels.stats.power import TTestIndPower
from tqdm.auto import tqdm


def merge_groups(
    control_group: Union[Iterable[pd.DataFrame], pd.DataFrame],
    test_group: Union[Iterable[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    """Merges test and control groups in one DataFrame and creates column "group".
    Column "group" contains of "test" and "control" values.

    Args:
        control_group:
            Data of control group
        test_group:
            Data of target group
    Returns:
        merged_data:
            Contacted DataFrame
    """
    control_group.loc[:, "group"] = "control"
    test_group.loc[:, "group"] = "test"

    return pd.concat([test_group, control_group], ignore_index=True)


def split_splited_data(splitted_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
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


def calc_mde(
    test_group: pd.Series,
    control_group: pd.Series,
    reliability: float = 0.95,
    power: float = 0.8,
) -> float:
    """Calculates the minimum detectable effect (MDE) for a given test and control groups.

    Args:
        test_group:
            The test group as a pandas Series
        control_group:
            The control group as a pandas Series
        reliability:
            The reliability of the test
        power:
            The power of the test

    Returns:
        The minimum detectable effect
    """

    m = norm.ppf(1 - (1 - reliability) / 2) + norm.ppf(power)

    n_test, n_control = len(test_group), len(control_group)
    proportion = n_test / (n_test + n_control)
    p = np.sqrt(1 / (proportion * (1 - proportion)))

    var_test, var_control = np.var(test_group, ddof=1), np.var(control_group, ddof=1)
    s = np.sqrt(var_test / n_test + var_control / n_control)

    return p * m * s


def calc_sample_size(
    control_group: pd.Series,
    test_group: pd.Series,
    mde,
    significance: float = 0.05,
    power: float = 0.8,
) -> float:
    """Calculates sample size of dataframe depends on mde and power.

    Args:
        control_group:
            Numpy Series with data from control group
        test_group:
            Numpy Series with data from test group
        mde:
            Minimum detectable effect
        significance:
            Level of significance (alpha)
        power:
            Power of criterion

    Returns:
        Sample_size of dataframe
    """
    if isinstance(mde, Iterable):
        z_alpha = norm.ppf((2 - significance) / 2)
        z_beta = norm.ppf(power)

        p1 = mde[0]
        p2 = mde[1]

        return (
            (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (p1 - p2) ** 2
        )
    else:
        control_std = control_group.std()
        test_std = test_group.std()

        test_proportion = len(test_group) / (len(test_group) + len(control_group))
        control_proportion = 1 - test_proportion

        d = ((norm.ppf(1 - significance / 2) + norm.ppf(power)) / mde) ** 2
        s = test_std**2 / test_proportion + control_std**2 / control_proportion
        return d * s


def calc_power(
    effect_size: float,
    control_size: float,
    test_size: float,
    significance: float = 0.05,
) -> float:
    """Statistical power calculations for t-test for two independent sample and known significance.

    Args:
        effect_size:
            Size of the effect
        control_size:
            Size of control group
        test_size:
            Size of test group
        significance:
            Level of significance (alpha)

    Returns:
        Statistical power
    """
    analysis = TTestIndPower()
    ratio = test_size / control_size
    return analysis.power(
        effect_size=effect_size,
        nobs1=test_size,
        ratio=ratio,
        alpha=significance,
    )


class AATest:
    """
    A class for conducting AA testing (random split testing) to assess the
    statistical uniform of two samples.

    AA testing is used to validate that the splitting mechanism of an A/B test
    is unbiased and random. This class supports various statistical methods to
    evaluate the equivalence of two randomly split samples in terms of various
    metrics.

    Attributes:
        target_fields (Union[Iterable[str], str]): Target column names to analyze. This fields should be numeric.
        group_cols (Union[str, Iterable[str]]): Column names used for grouping. This fields should be categorical. It's a field for stratification. Stratification - the way to divide groups with equal number of categories in each of them.
        info_cols (Union[str, Iterable[str]]): Column names for additional information.
        quant_field (str): Name of the column for quantization. This fields should be categorical. A quantum is a category that passes entirely into one of the groups.
        mode (str): Mode of the AA test. Options are 'simple' and 'balanced'. 'simple' - naively splits groups in half. 'balanced' - separation with quantum balancing at each step (only used if a quantization field is specified.
        alpha (float): Level of significance for statistical tests.

    Methods:
        columns_labeling(data):
            Classifies columns in the input DataFrame as target or group columns.
        __simple_mode(data, random_state, test_size):
            Internal method to create a simple random split of the data.
        split(data, random_state, test_size):
            Splits the dataset into test and control groups.
        _postprep_data(data, spit_indexes):
            Combines the index markup obtained at the split step.
        calc_ab_delta(a_mean, b_mean, mode):
            Calculates the difference between averages of two samples.
        sampling_metrics(data, random_state, test_size):
            Computes various metrics for a single random split of the data.
        calc_uniform_tests(data, test_size, iterations, file_name, experiment_write_mode, split_write_mode, write_step, pbar):
            Runs multiple iterations of AA tests to find a uniform distribution.
        features_p_value_distribution(experiment_results, figsize, bin_step):
            Plots the distribution of p-values for each feature.
        aa_score(experiment_results):
            Computes the average score for passed tests in AA testing.
        uniform_tests_interpretation(experiment_results):
            Analyzes and plots the results of uniform tests.
        num_feature_uniform_analysis(control_data, test_data, plot_set):
            Analyzes and plots numerical feature distributions in control and test data.
        cat_feature_uniform_analysis(control_data, test_data):
            Analyzes and plots categorical feature distributions in control and test data.
        experiment_result_transform(experiment):
            Transforms the result of an experiment into a readable format.
        split_analysis(splited_data):
            Analyzes split data for both target and group columns.
        get_resume(aa_score, best_experiment_stat):
            Formats the final results of AA testing for clarity.
        process(data, optimize_groups, iterations, show_plots, test_size, pbar):
            Main method to perform the complete AA test process, including optimization, testing, and result presentation.

    Example:
        >>> aa_test = AATest(target_fields=["metric1", "metric2"], group_cols=["group"], info_cols=["info1", "info2"])
        >>> results = aa_test.process(data, optimize_groups=True, iterations=1000, show_plots=True)
    """

    def __init__(
        self,
        target_fields: Union[Iterable[str], str] = None,
        group_cols: Union[str, Iterable[str]] = None,
        info_cols: Union[str, Iterable[str]] = None,
        quant_field: str = None,
        mode: str = "simple",
        alpha: float = 0.05,
    ):
        """Initialize the AATest class.

        Args:
            target_fields:
                List or str with target columns. This fields should be numeric.
            group_cols:
                List or str with columns for grouping. This fields should be categorical. It's a field for stratification. Stratification - the way to divide groups with equal number of categories in each of them.
            info_cols:
                List or str with informational columns
            quant_field:
                String with name of column for quantization. This fields should be categorical. A quantum is a category that passes entirely into one of the groups.
            mode:
                Mode of the AA-test
                Available modes:
                    * simple - naively splits groups in half
                    * balanced - separation with quantum balancing at each step (only used if a quantization field is specified)
            alpha:
                Level of significance
        """
        self.target_fields = (
            [target_fields] if isinstance(target_fields, str) else target_fields
        )
        self.group_cols = (
            [group_cols] if isinstance(group_cols, str) else group_cols
        ) or []
        self.info_cols = [info_cols] if isinstance(info_cols, str) else info_cols
        self.quant_field = quant_field
        self.mode = mode
        self.alpha = alpha

    def columns_labeling(self, data: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Label columns as target columns and group columns.

        Args:
            data:
                Input dataframe

        Returns:
            Dictionary with list of target columns and group columns

        """
        return {
            "target_field": list(
                data.select_dtypes(include="number").columns.drop(
                    self.info_cols, errors="ignore"
                )
            ),
            "group_col": list(
                data.select_dtypes(include="object").columns.drop(
                    self.info_cols, errors="ignore"
                )
            ),
        }

    def __simple_mode(
        self, data: pd.DataFrame, random_state: int = None, test_size: float = 0.5
    ) -> Dict:
        """Separates data on A and B samples within simple mode.
        Separation performed to divide groups of equal sizes - equal amount of records
        or equal amount of groups in each sample.

        Args:
            data:
                Input data
            random_state:
                Seed of random

        Returns:
            Test and control samples of indexes dictionary
        """
        result = {"test_indexes": [], "control_indexes": []}

        if self.quant_field:
            random_ids = shuffle(
                data[self.quant_field].unique(), random_state=random_state
            )
            edge = int(len(random_ids) * test_size)
            result["test_indexes"] = list(
                data[data[self.quant_field].isin(random_ids[:edge])].index
            )
            result["control_indexes"] = list(
                data[data[self.quant_field].isin(random_ids[edge:])].index
            )

        else:
            addition_indexes = list(shuffle(data.index, random_state=random_state))
            edge = int(len(addition_indexes) * test_size)
            result["test_indexes"] = addition_indexes[:edge]
            result["control_indexes"] = addition_indexes[edge:]

        return result

    def split(
        self, data: pd.DataFrame, random_state: int = None, test_size: float = 0.5
    ) -> Dict:
        """Divides sample on two groups.

        Args:
            data:
                Input data
            random_state:
                Seed of random - one integer to fix split
            test_size:
                Proportion of the test group

        Returns:
            Dict of indexes with division on test and control group
        """
        result = {"test_indexes": [], "control_indexes": []}

        if self.group_cols:
            groups = data.groupby(self.group_cols)
            for _, gd in groups:
                if self.mode not in ("balanced", "simple"):
                    warnings.warn(
                        f"The mode '{self.mode}' is not supported for group division. Implemented mode 'simple'."
                    )
                    self.mode = "simple"

                if self.mode == "simple":
                    t_result = self.__simple_mode(gd, random_state, test_size)
                    result["test_indexes"] += t_result["test_indexes"]
                    result["control_indexes"] += t_result["control_indexes"]

                elif self.mode == "balanced":
                    if self.quant_field:
                        random_ids = shuffle(
                            gd[self.quant_field].unique(), random_state=random_state
                        )
                        addition_indexes = list(
                            gd[gd[self.quant_field].isin(random_ids)].index
                        )
                    else:
                        addition_indexes = list(
                            shuffle(gd.index, random_state=random_state)
                        )

                    if len(result["control_indexes"]) > len(result["test_indexes"]):
                        result["test_indexes"] += addition_indexes
                    else:
                        result["control_indexes"] += addition_indexes

        else:
            if self.mode != "simple":
                warnings.warn(
                    f"The mode '{self.mode}' is not supported for regular division. "
                    f"Implemented mode 'simple'."
                )

            t_result = self.__simple_mode(data, random_state, test_size)
            result["test_indexes"] = t_result["test_indexes"]
            result["control_indexes"] = t_result["control_indexes"]

        result["test_indexes"] = list(set(result["test_indexes"]))
        result["control_indexes"] = list(set(result["control_indexes"]))

        return result

    @staticmethod
    def _postprep_data(data, spit_indexes: Dict = None) -> pd.DataFrame:
        """Prepares data to show user.
        Adds info_cols and decode binary variables.

        Args:
            data:
                Input data
            spit_indexes:
                Dict of indexes with separation on test and control group

        Returns:
            Separated initial data with column "group"
        """
        test = data.loc[spit_indexes["test_indexes"]]
        control = data.loc[spit_indexes["control_indexes"]]
        data = merge_groups(control, test)

        return data

    @staticmethod
    def calc_ab_delta(a_mean: float, b_mean: float, mode: str = "percentile")->float:
        """Calculates target delta between A and B groups.

        Args:
            a_mean:
                Average of target in one group
            b_mean:
                Average of target in another group
            mode:
                Type of expected result:
                    * 'percentile' - percentage exceeding the average in group A compared to group B
                    * 'absolute' - absolute value of difference between B and A group
                    * 'relative' - percent in format of number (absolute) exceeding the average in group A compared to group B

        Returns:
            Delta between groups as percent or absolute value
        """
        if mode == "percentile":
            return (1 - a_mean / b_mean) * 100
        if mode == "absolute":
            return b_mean - a_mean
        if mode == "relative":
            return 1 - a_mean / b_mean

    def sampling_metrics(
        self, data: pd.DataFrame, random_state: int = None, test_size: float = 0.5
    ) -> Dict:
        """Calculates metrics of one sampling.

        Args:
            data:
                Input data
            random_state:
                Random seeds for searching
            test_size:
                Proportion of the test group

        Returns:
            Dict of
                1) metrics dataframe (stat tests) and
                2) dict of random state with test_control dataframe
        """
        scores = []
        t_result = {"random_state": random_state}

        split = self.split(data, random_state, test_size)

        a = data.loc[split["control_indexes"]]
        b = data.loc[split["test_indexes"]]

        data_from_sampling_dict = {random_state: self._postprep_data(data, split)}
        for tf in self.target_fields:
            ta = a[tf]
            tb = b[tf]

            t_result[f"{tf} a mean"] = ta.mean()
            t_result[f"{tf} b mean"] = tb.mean()
            t_result[f"{tf} ab delta"] = self.calc_ab_delta(
                t_result[f"{tf} a mean"], t_result[f"{tf} b mean"], "absolute"
            )
            t_result[f"{tf} ab delta %"] = self.calc_ab_delta(
                t_result[f"{tf} a mean"], t_result[f"{tf} b mean"], "percentile"
            )
            t_result[f"{tf} t-test p-value"] = ttest_ind(
                ta, tb, nan_policy="omit"
            ).pvalue
            t_result[f"{tf} ks-test p-value"] = ks_2samp(ta, tb).pvalue
            t_result[f"{tf} t-test passed"] = (
                t_result[f"{tf} t-test p-value"] < self.alpha
            )
            t_result[f"{tf} ks-test passed"] = (
                t_result[f"{tf} ks-test p-value"] < self.alpha
            )
            scores.append(
                (
                    t_result[f"{tf} t-test p-value"]
                    + 2 * t_result[f"{tf} ks-test p-value"]
                )
                / 3
            )

        t_result["control %"] = len(a) / len(data) * 100
        t_result["test %"] = len(b) / len(data) * 100
        t_result["control size"] = len(a)
        t_result["test size"] = len(b)
        t_result["t-test mean p-value"] = np.mean(
            [p_value for key, p_value in t_result.items() if "t-test p-value" in key]
        )
        t_result["ks-test mean p-value"] = np.mean(
            [p_value for key, p_value in t_result.items() if "ks-test p-value" in key]
        )
        t_result["t-test passed %"] = np.mean(
            [passed * 100 for key, passed in t_result.items() if "t-test passed" in key]
        )
        t_result["ks-test passed %"] = np.mean(
            [
                passed * 100
                for key, passed in t_result.items()
                if "ks-test passed" in key
            ]
        )
        t_result["mean_tests_score"] = np.mean(scores)
        return {"metrics": t_result, "data_from_experiment": data_from_sampling_dict}

    def calc_uniform_tests(
        self,
        data: pd.DataFrame,
        test_size: float = 0.5,
        iterations: int = 2000,
        file_name: Union[Path, str] = None,
        experiment_write_mode: str = "full",
        split_write_mode: str = "full",
        write_step: int = None,
        pbar: bool = True,
        **kwargs,
    ) -> Optional[Tuple[pd.DataFrame, Dict[Any, Dict]]]:
        """Performs multiple separation experiments for different random states.

        Args:
            data:
                Input data
            iterations:
                Number of iterations to search uniform sampling to searching
            test_size:
                Proportion of the test group
            file_name:
                Name of file to save results (if None - no results will be saved, func returns result)
            experiment_write_mode:
                Mode to write experiment results:
                    'full' - save all experiments
                    'all' - save experiments that passed all statistical tests
                    'any' - save experiments that passed any statistical test
            split_write_mode:
                Mode to write split results:
                    'full' - save all experiments
                    'all' - save experiments that passed all statistical tests
                    'any' - save experiments that passed any statistical test
            write_step:
                Step to write experiments to file
            pbar:
                Flag to show progress bar

        Returns:
                If no saving (no file_name, no write mode and no write_step) returns dataframe
                else None and saves file to csv
        """
        random_states = range(iterations)
        results = []
        data_from_sampling = {}

        if experiment_write_mode not in ("full", "all", "any"):
            warnings.warn(
                f"Write mode '{experiment_write_mode}' is not supported. Mode 'full' will be used"
            )
            experiment_write_mode = "full"
        if split_write_mode not in ("full", "all", "any"):
            warnings.warn(
                f"Write mode '{split_write_mode}' is not supported. Mode 'full' will be used"
            )
            split_write_mode = "full"

        for i, rs in tqdm(
            enumerate(random_states), total=len(random_states), disable=not pbar
        ):
            res = self.sampling_metrics(data, random_state=rs, test_size=test_size)

            # write to file
            passed = []
            for tf in self.target_fields:
                passed += [
                    not res["metrics"][f"{tf} t-test passed"],
                    not res["metrics"][f"{tf} ks-test passed"],
                ]

            if all(passed):
                if experiment_write_mode == "all":
                    results.append(res["metrics"])
                if split_write_mode == "all":
                    data_from_sampling.update(res["data_from_experiment"])
            if any(passed):
                if experiment_write_mode == "any":
                    results.append(res["metrics"])
                if split_write_mode == "any":
                    data_from_sampling.update(res["data_from_experiment"])
            if experiment_write_mode == "full":
                results.append(res["metrics"])
            if split_write_mode == "full":
                data_from_sampling.update(res["data_from_experiment"])

            if file_name and write_step:
                if i == write_step:
                    pd.DataFrame(results).to_csv(file_name, index=False)
                elif i % write_step == 0:
                    pd.DataFrame(results).to_csv(
                        file_name, index=False, header=False, mode="a"
                    )
                    results = []

        results = pd.DataFrame(results)
        if file_name and write_step:
            results.to_csv(file_name, index=False, header=False, mode="a")
        elif file_name:
            results.to_csv(file_name, index=False)
            return results, data_from_sampling
        else:
            return results, data_from_sampling

    def features_p_value_distribution(
        self, experiment_results: pd.DataFrame, figsize=None, bin_step=0.05
    ):
        """Process plots of features' p-value distribution.

        Args:
            experiment_results:
                Results of experiments
            figsize:
                Size of figure for plot
            bin_step:
                Step for bins in X axis
        """
        feature_num = len(self.target_fields)
        figsize = figsize or (15, 7 * feature_num)
        bin_step = bin_step or self.alpha
        bins = np.arange(0, 1 + bin_step, bin_step)
        figure, axs = plt.subplots(nrows=feature_num, ncols=2, figsize=figsize)
        for i in range(feature_num):
            sns.histplot(
                data=experiment_results,
                x=f"{self.target_fields[i]} t-test p-value",
                ax=axs[i, 0],
                bins=bins,
                stat="percent",
                shrink=0.8,
            )
            sns.histplot(
                data=experiment_results,
                x=f"{self.target_fields[i]} ks-test p-value",
                ax=axs[i, 1],
                bins=bins,
                stat="percent",
                shrink=0.8,
            )

            axs[i, 0].set_title(
                f"{self.target_fields[i]} t-test p-value\npassed score: {experiment_results[f'{self.target_fields[i]} t-test passed'].mean():.3f}"
            )
            axs[i, 1].set_title(
                f"{self.target_fields[i]} ks-test p-value\npassed score: {experiment_results[f'{self.target_fields[i]} ks-test passed'].mean():.3f}"
            )
        plt.show()

    def aa_score(self, experiment_results: pd.DataFrame) -> pd.DataFrame:
        """Estimates mean passed score for t-test and ks-test in AA-test.

        Args:
            experiment_results:
                Results of the experiment

        Returns:
            Pandas dataframe containing the results of the AA-test
        """
        result = pd.DataFrame(
            {
                f: {
                    "t-test passed score": experiment_results[
                        f"{f} t-test passed"
                    ].mean(),
                    "ks-test passed score": experiment_results[
                        f"{f} ks-test passed"
                    ].mean(),
                }
                for f in self.target_fields
            }
        ).T

        result["t-test aa passed"] = result["t-test passed score"].apply(
            lambda x: 0.8 * self.alpha <= x <= 1.2 * self.alpha
        )
        result["ks-test aa passed"] = result["ks-test passed score"].apply(
            lambda x: 0.8 * self.alpha <= x <= 1.2 * self.alpha
        )
        result.loc["mean"] = result.mean()

        return result

    def uniform_tests_interpretation(
        self, experiment_results: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
        """Process plotting of p-value distribution and results of AA-test.

        Args:
            experiment_results:
                Results of experiments
            **kwargs:
                Some extra keyword arguments:
                    * figsize: Size of figure for plot
                    * bin_step: Step for bins in X axis

        Returns:
            Pandas dataframe containing the results of the AA-test
        """
        self.features_p_value_distribution(
            experiment_results,
            figsize=kwargs.get("figsize"),
            bin_step=kwargs.get("bin_step"),
        )
        return self.aa_score(experiment_results)

    def num_feature_uniform_analysis(
        self,
        control_data: pd.Series,
        test_data: pd.Series,
        plot_set: Tuple = ("hist", "cumulative", "percentile"),
        **kwargs,
    ):
        """Show plots of distribution in groups with uniform tests.

        Args:
            control_data:
                Data from control group
            test_data:
                Data from test group
            plot_set:
                Type of plot
                Available types:
                    * hist
                    * cumulative
                    * percentile
            **kwargs:
                Some extra keyword arguments:
                    * figsize: Size of figure for plot
                    * bins: Number of bins in X axis
                    * alpha: Transparency of histograms
        """
        if not plot_set:
            return

        figsize = kwargs.get("figsize", (25, 20))
        figure, axs = plt.subplots(
            nrows=len(plot_set),
            ncols=1,
            figsize=figsize,
            facecolor="honeydew",
            edgecolor="black",
        )
        ax_count = 0

        bins = np.arange(
            min(control_data.min(), test_data.min()),
            max(control_data.max(), test_data.max()),
            (
                max(control_data.max(), test_data.max())
                - min(control_data.min(), test_data.min())
            )
            / kwargs.get("bins", 100),
        )

        if "hist" in plot_set:
            sns.histplot(
                data=control_data,
                ax=axs[ax_count],
                bins=bins,
                stat="percent",
                element="poly",
                alpha=kwargs.get("alpha", 0.3),
                color="blue",
            )
            sns.histplot(
                data=test_data,
                ax=axs[ax_count],
                bins=bins,
                stat="percent",
                element="poly",
                alpha=kwargs.get("alpha", 0.3),
                color="red",
            )
            axs[ax_count].grid(True)
            axs[ax_count].legend(["control", "test"])
            axs[ax_count].set_title("Histogram")
            ax_count += 1

        if "cumulative" in plot_set:
            sns.histplot(
                data=control_data,
                ax=axs[ax_count],
                bins=bins,
                stat="percent",
                element="poly",
                cumulative=True,
                alpha=kwargs.get("alpha", 0.3),
                color="blue",
            )
            sns.histplot(
                data=test_data,
                ax=axs[ax_count],
                bins=bins,
                stat="percent",
                element="poly",
                cumulative=True,
                alpha=kwargs.get("alpha", 0.3),
                color="red",
            )
            axs[ax_count].legend(["control", "test"])
            axs[ax_count].set_title("Cumulative destribution")
            ax_count += 1

        if "percentile" in plot_set:
            axs[ax_count].fill_between(
                range(101),
                [control_data.quantile(q) for q in np.arange(0, 1.01, 0.01)],
                color="blue",
                alpha=kwargs.get("alpha", 0.3),
            )
            axs[ax_count].fill_between(
                range(101),
                [test_data.quantile(q) for q in np.arange(0, 1.01, 0.01)],
                color="red",
                alpha=kwargs.get("alpha", 0.3),
            )
            axs[ax_count].legend(["control", "test"])
            axs[ax_count].set_xticks(np.arange(0, 101))
            axs[ax_count].set_xticklabels(np.arange(0, 101), rotation=45)
            axs[ax_count].set_title("Percentile destribution")

        fig_title = f"""{control_data.name}

            t-test p-value: {ttest_ind(control_data, test_data, nan_policy='omit').pvalue:.3f}
            ks-test p-value: {ks_2samp(control_data, test_data).pvalue:.3f}"""
        figure.suptitle(fig_title, fontsize=kwargs.get("title_size", 20))
        plt.show()

    def cat_feature_uniform_analysis(
        self, control_data: pd.Series, test_data: pd.Series, **kwargs
    ):
        """Show plots of distribution in groups.

        Args:
            control_data:
                Data from control group
            test_data:
                Data from test group
            **kwargs:
                Some extra keyword arguments:
                    * figsize: Size of figure for plot
                    * alpha: Transparency of histograms
        """
        s_control_data = control_data.astype("str")
        s_test_data = test_data.astype("str")

        figsize = kwargs.get("figsize", (15, 10))
        figure, ax = plt.subplots(
            nrows=1, ncols=1, figsize=figsize, facecolor="honeydew", edgecolor="black"
        )

        control_counts = s_control_data.value_counts(normalize=True) * 100
        test_counts = s_test_data.value_counts(normalize=True) * 100

        ax.fill_between(
            control_counts.index,
            control_counts.values,
            color="blue",
            alpha=kwargs.get("alpha", 0.3),
            label="control",
        )
        ax.fill_between(
            test_counts.index,
            test_counts[
                [i for i in test_counts.index if i in control_counts.index]
            ].values,
            color="red",
            alpha=kwargs.get("alpha", 0.3),
            label="test",
        )

        ax.legend()
        ax.tick_params(axis="x", rotation=90)
        figure.suptitle(f"{control_data.name}", fontsize=kwargs.get("title_size", 20))
        plt.show()

    def experiment_result_transform(self, experiment: pd.Series):
        """
        Transform experiments results into readable view.

        Args:
            experiment:
                Results of experiments

        Returns:
            DataFrame with results of the experiment and statistics from best split
        """
        targets_dict = {}
        for tf in self.target_fields:
            targets_dict[tf] = {}
            for i in experiment.index:
                if i.startswith(f"{tf} "):
                    targets_dict[tf][i[len(tf) + 1 :]] = experiment[i]
        return pd.DataFrame(targets_dict).T, experiment.iloc[-9:]

    def split_analysis(self, splited_data: pd.DataFrame, **kwargs):
        """Conducts a full splitting analysis.

        Args:
            splited_data:
                Data that has already been split
            **kwargs:
                Some extra keyword arguments for plots in visualization
        """
        ssp = split_splited_data(splited_data)
        for nf in self.target_fields:
            self.num_feature_uniform_analysis(
                ssp["control"][nf], ssp["test"][nf], **kwargs
            )
        for cf in self.group_cols:
            self.cat_feature_uniform_analysis(
                ssp["control"][cf], ssp["test"][cf], **kwargs
            )

    def get_resume(self, aa_score: pd.DataFrame, best_experiment_stat: pd.DataFrame):
        """Format results into clear format for understanding.

        Args:
            aa_score:
                Results of aa-test
            best_experiment_stat:
                Results of the best experiment

        Returns:
            DataFrame with OK and not OK depending on the results of statistical tests
        """
        result = {"aa test passed": {}, "split is uniform": {}}
        for field in self.target_fields:
            result["aa test passed"][field] = (
                aa_score.loc[field, "t-test aa passed"]
                or aa_score.loc[field, "ks-test aa passed"]
            )
            result["split is uniform"][field] = (
                best_experiment_stat.loc[field, "t-test passed"]
                or best_experiment_stat.loc[field, "ks-test passed"]
            )
        result = pd.DataFrame(result)
        result["split is uniform"] = (
            result["split is uniform"]
            .astype("bool")
            .replace({False: "OK", True: "not OK"})
        )
        result["aa test passed"] = (
            result["aa test passed"]
            .astype("bool")
            .replace({False: "not OK", True: "OK"})
        )
        return result

    def process(
        self,
        data: pd.DataFrame,
        optimize_groups: bool = False,
        iterations: int = 2000,
        show_plots: bool=True,
        test_size: float=0.5,
        pbar: bool=True,
        **kwargs,
    ):
        """Main function for AATest estimation.

        Provides:
            * Columns labeling
            * Results calculations
            * Plotting results

        Args:
            test_size:
                Proportion of the test group
            data:
                Input dataset
            optimize_groups:
                Is in necessary to optimize groups
            iterations:
                Number of iterations for AA-test
            show_plots:
                Is in necessary to show plots
            pbar:
                Show progress-bar
            **kwargs:
                Some extra keyword arguments

        Returns:
            best_results:
                Results of the experiment with metrics for all fields
            best_split:
                Result of separation
        """
        labeling = self.columns_labeling(data)
        best_results, best_split = None, None

        if not self.target_fields:
            self.target_fields = labeling["target_fields"]

        if optimize_groups:
            max_score = -1

            group_variants = [[]]
            for i in range(1, len(labeling["group_col"])):
                i_combinations = combinations(labeling["group_col"], i)
                group_variants.extend(iter(i_combinations))

            for gs in tqdm(group_variants, desc="Group optimization", disable=not pbar):
                self.group_cols = list(gs)
                experiment_results, data_splits = self.calc_uniform_tests(
                    data,
                    pbar=False,
                    iterations=iterations,
                    test_size=test_size,
                    **kwargs,
                )
                if len(experiment_results):
                    aa_scores = self.aa_score(experiment_results)
                    group_score = max(
                        aa_scores.loc["mean", "t-test aa passed"],
                        aa_scores.loc["mean", "ks-test aa passed"],
                    )
                    if group_score > max_score:
                        best_results, best_split = experiment_results, data_splits
                        max_score = group_score

        else:
            best_results, best_split = self.calc_uniform_tests(
                data,
                experiment_write_mode="full",
                split_write_mode="any",
                iterations=iterations,
                test_size=test_size,
                pbar=pbar,
                **kwargs,
            )

        if len(best_results) == 0:
            return best_results, best_split
        if len(best_results) > 0:
            if show_plots:
                aa_scores = self.uniform_tests_interpretation(best_results)
            else:
                aa_scores = self.aa_score(best_results)
            best_rs = best_results.loc[
                best_results["mean_tests_score"].idxmax(), "random_state"
            ]
            final_split = best_split[best_rs]
            if show_plots:
                self.split_analysis(final_split, **kwargs)

            best_experiment_stat, best_split_stat = self.experiment_result_transform(
                best_results[best_results["random_state"] == best_rs].iloc[0]
            )
            resume = self.get_resume(aa_scores, best_experiment_stat)
        else:
            aa_scores = None
            final_split = None
            best_experiment_stat = None
            best_split_stat = None
            resume = None

        return {
            "experiments": best_results,
            "aa_score": aa_scores,
            "split": final_split,
            "best_experiment_stat": best_experiment_stat,
            "split_stat": best_split_stat,
            "resume": resume,
        }
