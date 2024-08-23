"""Module for AA tests"""

import math
import warnings
from decimal import Decimal
from itertools import combinations
from pathlib import Path
from typing import Iterable, Union, Optional, Dict, Any, Tuple, List, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind, ks_2samp, norm, chi2_contingency
from sklearn.utils import shuffle
from statsmodels.stats.power import TTestIndPower

try:
    from tqdm.auto import tqdm
except:
    try:
        from tqdm import tqdm
    except:
        raise Exception("Can't import tqdm")


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
        merge_groups(control_group, test_group):
            Merges test and control groups in one DataFrame and creates column "group".
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
        calc_mde_unbalanced_group(data, target_field, group_flag_col, power):
            Finds minimum detectable effect (MDE) and effect size for unbalanced groups.
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

    def _postprep_data(self, data, spit_indexes: Dict = None) -> pd.DataFrame:
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
        data = self.merge_groups(control, test)

        return data

    @staticmethod
    def calc_ab_delta(a_mean: float, b_mean: float, mode: str = "percentile") -> float:
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
            axs[ax_count].set_title("Cumulative distribution")
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
            axs[ax_count].set_title("Percentile distribution")

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

    def __get_test_and_control_series(
        self,
        test_group: pd.Series = None,
        control_group: pd.Series = None,
        data: pd.DataFrame = None,
        group_field: str = "group",
        target_field: str = None,
    ):
        if test_group is None and control_group is None:
            if data is None or target_field is None:
                raise ValueError(
                    "test_group and control_group cannot be None if data and target_field are None"
                )
            splited_data = self.split_splited_data(data, group_field=group_field)
            test_group = splited_data["test"][target_field]
            control_group = splited_data["control"][target_field]
        return {"test_group": test_group, "control_group": control_group}

    def __mde_unbalanced_non_binomial(
        self,
        control_group_size: int,
        all_data_size: int,
        standard_deviation: float = 2.0,
        power: float = 0.8,
    ) -> Tuple:
        """Calculates minimum detectable effect (MDE) and significance
        of the effect size for unbalanced non-binomial groups.

                       Args:
                           control_group_size:
                               Size of the control group
                           all_data_size:
                               Size of the all data sample
                           standard_deviation:
                               Standard_deviation
                           power:
                               Level of power


                       Returns:
                           Tuple with MDE and significance of the effect size
        """
        proportion = round((all_data_size - control_group_size) / control_group_size, 2)
        z_alpha = norm.ppf(1 - self.alpha / 2)
        z_power = norm.ppf(power)
        mde = (standard_deviation * (z_alpha - z_power)) * math.sqrt(
            (1 + proportion) / (control_group_size * proportion)
        )
        effect_size = mde / standard_deviation  # effect_size

        return (
            float(Decimal(float(mde)).quantize(Decimal("1.00"))),
            float(Decimal(float(effect_size)).quantize(Decimal("1.00"))),
        )

    def __mde_unbalanced_binomial(
        self,
        control_group_size: int,
        all_data_size: int,
        fact_conversion: float,
        power: float = 0.8,
    ) -> Tuple:
        """Calculates minimum detectable effect (MDE) and significance
        of the effect size (Cohen's d) for unbalanced binomial groups.

               Args:
                   control_group_size:
                       Size of the control group
                   all_data_size:
                       Size of the all data sample
                   fact_conversion:
                       Conversion in the control group
                   power:
                       Level of power


               Returns:
                   Tuple with MDE and Cohen's d
        """
        proportion = round((all_data_size - control_group_size) / control_group_size, 2)
        z_alpha = norm.ppf(1 - self.alpha / 2)
        z_power = norm.ppf(power)
        cohen_d = (z_alpha - z_power) * math.sqrt(
            (1 + proportion) / (control_group_size * proportion)
        )  # effect_size
        expect_conversion = (
            math.sin(math.asin(math.sqrt(fact_conversion)) + cohen_d / 2) ** 2
        )  # expect_conversion
        mde = abs(expect_conversion - fact_conversion)

        return (
            float(Decimal(float(mde)).quantize(Decimal("1.00"))),
            float(Decimal(float(cohen_d)).quantize(Decimal("1.00"))),
        )

    def calc_mde(
        self,
        data: pd.DataFrame,
        target_field: str,
        group_field: str,
        power: float = 0.8,
    ) -> Tuple:
        """Finds minimum detectable effect (MDE) and effect size for unbalanced groups.

        Args:
            data:
                Input data
            target_field:
                Name of the target feature
            group_field:
                Name of the column with the group flag
            power:
                Level of power


        Returns:
            Tuple with MDE and effect size
        """
        fact_conversion = float(
            Decimal(float(data[target_field].mean())).quantize(Decimal("1.00"))
        )  # fact_conversion
        control_group = data[
            (data[group_field] == data[group_field].unique()[0])
        ]  # control_group
        control_group_size = len(control_group)  # control_group_size
        all_data_size = len(data[group_field])  # all_data_size (???)

        if data[target_field].nunique() == 2:
            res = self.__mde_unbalanced_binomial(
                control_group_size=control_group_size,
                all_data_size=all_data_size,
                fact_conversion=fact_conversion,
                power=power,
            )
        else:
            res = self.__mde_unbalanced_non_binomial(
                control_group_size=control_group_size,
                all_data_size=all_data_size,
                standard_deviation=data[target_field].std(),
                power=power,
            )

        return res

    def calc_sample_size(
        self,
        test_group: pd.Series = None,
        control_group: pd.Series = None,
        data: pd.DataFrame = None,
        target_field: str = None,
        group_field: str = "group",
        mde: float = None,
        significance: float = 0.05,
        power: float = 0.8,
    ) -> float:
        """Calculates sample size of dataframe depends on mde and power.

        Args:
            test_group:
                The test group as a pandas Series
            control_group:
                The control group as a pandas Series
            data:
                The input data as a pandas DataFrame used if test_group and control_group are None
            target_field:
                The target field used if given data is a DataFrame
            group_field:
                The group field used if given data is a DataFrame
            mde:
                Minimum detectable effect. If None, it will be calculated. If it is a tuple, it will be used as relative mde
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
                (z_alpha + z_beta) ** 2
                * (p1 * (1 - p1) + p2 * (1 - p2))
                / (p1 - p2) ** 2
            )
        else:
            groups = self.__get_test_and_control_series(
                test_group=test_group,
                control_group=control_group,
                data=data,
                target_field=target_field,
                group_field=group_field,
            )
            control_group = groups["control_group"]
            test_group = groups["test_group"]
            mde = mde or self.calc_mde(
                data=data,
                target_field=target_field,
                group_field=group_field,
                power=power,
            )

            control_std = control_group.std()
            test_std = test_group.std()

            test_proportion = len(test_group) / (len(test_group) + len(control_group))
            control_proportion = 1 - test_proportion

            d = ((norm.ppf(1 - significance / 2) + norm.ppf(power)) / mde) ** 2
            s = test_std**2 / test_proportion + control_std**2 / control_proportion
            return d * s

    def calc_imbalanced_sample_size(
        self,
        target_data: pd.Series,
        expected_mean: float,
        proportion: float = 0.5,
        power: float = 0.8,
    ) -> Tuple:
        """Calculates imbalanced sample size for control and test group.

        Args:
            target_data:
                The control group as a pandas Series
            expected_mean:
                Expected conversion of test group
            proportion:
                Proportion of control group
            power:
                Power of criterion

        Returns:
            Tuple with size for control and test group
        """
        target_mean = target_data.mean()

        if target_mean == expected_mean:
            raise ValueError("Current conversion and expected conversion are equal!")

        proportion = (1 - proportion) / proportion
        z_alpha = norm.ppf(1 - self.alpha / 2)
        z_power = norm.ppf(power)
        if target_data.nunique() == 2:
            h_cohen = 2 * np.arcsin(target_mean**0.5) - 2 * np.arcsin(
                expected_mean**0.5
            )
            control_size = (
                (1 + proportion) / proportion * ((z_alpha + z_power) / h_cohen) ** 2
            )
        else:
            mde = abs(expected_mean - target_mean)
            control_size = (
                (1 + proportion)
                / proportion
                * (target_data.std() ** 2)
                * (z_alpha + z_power) ** 2
                / mde**2
            )
        test_size = proportion * control_size

        return np.int32(np.ceil(control_size)), np.int32(np.ceil(test_size))

    @staticmethod
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

    def calc_chi2(self, df: pd.DataFrame, treated_column: str):
        """Chi2 criterio calculation.

        Args:
            df:
                Input data
            treated_column:
                Column name with group markers (treated/untreated)

        Returns:
            Dictionary with pvalues for all target fields.
        """
        df = df.sort_values(by=treated_column)
        groups = df[treated_column].unique().tolist()
        all_pvalues = {}
        for target_field in self.target_fields:
            group_a, group_b = (
                df[df[treated_column] == groups[0]][target_field],
                df[df[treated_column] == groups[1]][target_field],
            )
            proportions = group_a.shape[0] / (group_a.shape[0] + group_b.shape[0])
            group_a = group_a.value_counts().rename(target_field).sort_index()
            group_b = group_b.value_counts().rename(target_field).sort_index()
            index = (
                pd.Series(group_a.index.tolist() + group_b.index.tolist())
                .unique()
                .tolist()
            )
            group_a, group_b = [
                group.reindex(index, fill_value=0) for group in [group_a, group_b]
            ]
            merged_data = pd.DataFrame(
                {
                    "index_x": group_a * (1 - proportions),
                    "index_y": group_b * proportions,
                }
            ).fillna(0)
            sub_group = merged_data.sum(axis=1).sort_values()
            _filter = sub_group <= 15
            if _filter.sum():
                other = {
                    "index_x": merged_data["index_x"][_filter].sum(),
                    "index_y": merged_data["index_y"][_filter].sum(),
                }
                merged_data.drop(sub_group[_filter].index, inplace=True)
                merged_data = pd.concat(
                    [merged_data, pd.DataFrame([other], index=["Other"])]
                )
            all_pvalues[target_field] = chi2_contingency(
                merged_data[["index_x", "index_y"]]
            ).pvalue
        return all_pvalues

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

        return {
            "best_experiment_stat": pd.DataFrame(targets_dict).T,
            "best_split_stat": experiment.iloc[-9:],
        }

    @staticmethod
    def split_splited_data(
        splitted_data: pd.DataFrame, group_field
    ) -> Dict[str, pd.DataFrame]:
        """Splits a pandas DataFrame into two separate dataframes based on a specified group field.

        Args:
            splitted_data:
                The input dataframe to be split

        Returns:
            A dictionary containing two dataframes, 'test' and 'control', where 'test' contains rows where the
            group field is 'test', and 'control' contains rows where the group field is 'control'.
        """
        return {
            "control": splitted_data[splitted_data[group_field] == "control"],
            "test": splitted_data[splitted_data[group_field] == "test"],
        }

    def split_analysis(self, splited_data: pd.DataFrame, group_field, **kwargs):
        """Conducts a full splitting analysis.

        Args:
            splited_data:
                Data that has already been split
            **kwargs:
                Some extra keyword arguments for plots in visualization
        """
        ssp = self.split_splited_data(splited_data, group_field=group_field)
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
        show_plots: bool = True,
        test_size: float = 0.5,
        pbar: bool = True,
        group_field: str = "group",
        write_mode: Optional[Literal["any", "all", "full"]] = "all",
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
            self.target_fields = labeling["target_field"]

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
                experiment_write_mode=write_mode,
                split_write_mode=write_mode,
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
                self.split_analysis(
                    splited_data=final_split, group_field=group_field, **kwargs
                )

            transformed_results = self.experiment_result_transform(
                best_results[best_results["random_state"] == best_rs].iloc[0]
            )
            best_experiment_stat = transformed_results["best_experiment_stat"]
            best_split_stat = transformed_results["best_split_stat"]
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

    def multi_group_split(self, index: list, **kwargs) -> pd.DataFrame:
        """
        Shuffles index column a specified number of times and divides it in a specified proportion
        """

        groups = kwargs.get("groups", {"test": 0.5, "control": 0.5})
        iterations = kwargs.get("iterations", 2_000)
        show_pbar = kwargs.get("show_pbar", True)

        assert np.sum(list(groups.values())) == 1

        split_results = pd.DataFrame()
        edges = np.cumsum(list(groups.values())[:-1])

        for _indx, experiment_index in tqdm(
            enumerate(range(1, iterations + 1)),
            total=iterations,
            disable=not show_pbar,
            desc="Generating random divisions",
        ):

            shuffled_index = shuffle(index, random_state=_indx)
            splitted_index = np.split(
                shuffled_index, [int(edge * len(index)) for edge in edges]
            )

            series = []
            for group_indx, group_name in enumerate(groups):
                series.append(
                    pd.Series(
                        data=[group_name] * len(splitted_index[group_indx]),
                        index=splitted_index[group_indx],
                        dtype="category",
                    )
                )
            split_results[experiment_index] = pd.DataFrame(
                pd.concat(series), dtype="category"
            )
        return split_results

    def _get_experimen_metrics(
        self,
        df: pd.DataFrame,
        split_result: pd.Series,
        target_fields: list,
        alpha: float = 0.05,
    ) -> dict:
        """
        Returns a dictionary containing experiment metrics.
        For each target_field the average is calculated
        For each combination of groups, the deviation of the averages is calculated
        """

        metrics = {}
        test_scores = []

        for field in target_fields:
            field_values: dict = {}
            [
                field_values.update(
                    {group: df.loc[split_result[lambda x: x == group].index][field]}
                )
                for group in split_result.cat.categories
            ]

            [
                metrics.update({f"{group} {field} mean": np.mean(field_values[group])})
                for group in split_result.cat.categories
            ]

            for a, b in combinations(split_result.cat.categories, 2):
                metrics.update(
                    {
                        f"{field} mean delta ({a} - {b})": np.mean(field_values[a])
                        - np.mean(field_values[b]),
                        f"{field} mean delta% ({a} - {b})/{b}": (
                            np.mean(field_values[a]) - np.mean(field_values[b])
                        )
                        * 100
                        / np.mean(field_values[b]),
                        f"{field} t-test p-value ({a},{b})": ttest_ind(
                            field_values[a], field_values[b], nan_policy="omit"
                        ).pvalue,
                        f"{field} ks-test p-value ({a},{b})": ks_2samp(
                            field_values[a], field_values[b]
                        ).pvalue,
                    }
                )

            ttest_ind_pvalue = np.mean(
                [value for key, value in metrics.items() if "t-test p-value" in key]
            )
            ks_2samp_pvalue = np.mean(
                [value for key, value in metrics.items() if "ks-test p-value" in key]
            )

            metrics.update(
                {
                    f"{field} mean t-test p-value": ttest_ind_pvalue,
                    f"{field} mean ks-test p-value": ks_2samp_pvalue,
                    f"{field} t-test passed": ttest_ind_pvalue > alpha,
                    f"{field} ks-test passed": ks_2samp_pvalue > alpha,
                }
            )
            test_scores.append((ttest_ind_pvalue + 2 * ks_2samp_pvalue) / 3)

        metrics.update(
            {
                "mean of means t-test p-value": np.mean(
                    [
                        value
                        for key, value in metrics.items()
                        if "mean t-test p-value" in key
                    ]
                ),
                "mean of means ks-test p-value": np.mean(
                    [
                        value
                        for key, value in metrics.items()
                        if "mean ks-test p-value" in key
                    ]
                ),
                "t-test passed %": np.mean(
                    [
                        passed * 100
                        for key, passed in metrics.items()
                        if "t-test passed" in key
                    ]
                ),
                "ks-test passed %": np.mean(
                    [
                        passed * 100
                        for key, passed in metrics.items()
                        if "ks-test passed" in key
                    ]
                ),
                "mean_test_score": np.mean(test_scores),
                "experiment_index": int(split_result.name),
            }
        )
        return metrics

    def _plot_distribution(
        self, df: pd.DataFrame, plots_group_name: str, fields: List[str]
    ):
        figsize = (15, 3 * len(fields))
        fig, axis = plt.subplots(
            nrows=math.ceil(len(fields) / 2), ncols=2, figsize=figsize
        )
        fig.suptitle(plots_group_name, fontsize=16)
        for field, ax in zip(fields, axis.flat):
            sns.histplot(
                data=df,
                x=field,
                ax=ax,
                bins=20,
                stat="percent",
                shrink=0.8,
            )

    def _plot_distributions(self, df: pd.DataFrame, target_fields: List[str]):
        tests = ["t-test", "ks-test"]

        mean_of_means_pvalue = [
            f"mean of means {test_name} p-value" for test_name in tests
        ]
        mean_of_means_title = (
            "Distribution of averages of average p-values ​​of each target_field"
        )
        self._plot_distribution(
            df=df, plots_group_name=mean_of_means_title, fields=mean_of_means_pvalue
        )

        each_field_mean_pvalue = []
        for test in tests:
            for field in target_fields:
                each_field_mean_pvalue.append(f"{field} mean {test} p-value")
        each_field_mean_title = "Distribution of average p-values ​​of each target_field"
        self._plot_distribution(
            df=df, plots_group_name=each_field_mean_title, fields=each_field_mean_pvalue
        )

        each_field_pvalue = []
        for test in tests:
            for field in target_fields:
                for column_name in df.columns:
                    if f"{field} {test} p-value" in column_name:
                        each_field_pvalue.append(column_name)

        each_field_title = (
            "Distribution of p-value for each group combination for each target_field"
        )
        self._plot_distribution(
            df=df, plots_group_name=each_field_title, fields=each_field_pvalue
        )

    def process_split(
        self, df: pd.DataFrame, target_fields: List[str], **kwargs
    ) -> dict:
        """
        The function divides the passed DataFrame into the specified number of groups in specified proportions

        Args:
            df:
                Input pd.DataFame
            target_fields:
                List or str with target columns. This fields should be numeric.
            iterations:
                Number of iterations for AA-test
            alpha:
                Level of significance
            groups:
                Group proportions
            pbar:
                Show progress-bar
            inPlace:
                adds column to source DataFrame (True, False)
            group_column_name:
                group column name to be created after split
            show_plots:
                Is in necessary to show plots

        Returns dict with following keys:
            best metric:
                best metric by scores
            best_split:
                Result of separation
            all metrics:
                DataFrame with all calculcated metrics
            all splits:
                DataFrame with all split results
            best split DataFrame:
                DataFrame  with groups column
            get_resume:
                returns function plotting p-value distribution
        Example:
        >>> import hypex as hp
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = {
        >>>     'keys': range(1,100_001),
        >>>     'values': np.random.uniform(10,100,100_000)
        >>>     }
        >>> df = pd.DataFrame(data)
        >>> experiments = hp.AATest()
        >>> results = experiments.process_split(df=df, groups={'test1': 0.2,'test2': 0.3,'control': 0.5}, target_fields=['values'], iterations=100)
        """
        iterations = kwargs.get("iterations", 2_000)
        alpha = kwargs.get("alpha", self.alpha)
        groups = kwargs.get("groups", {"test": 0.5, "control": 0.5})
        show_pbar = kwargs.get("show_pbar", True)
        inPlace = kwargs.get("inPlace", True)
        group_column_name = kwargs.get("group_column_name", "group")
        show_plots = kwargs.get("show_plots", True)

        split_results = self.multi_group_split(
            index=list(df.index),
            groups=groups,
            iterations=iterations,
            show_pbar=show_pbar,
        )

        split_metrics = []

        for _, experiment_index in tqdm(
            enumerate(range(1, iterations + 1)),
            total=iterations,
            disable=not show_pbar,
            desc="Metrics calculations",
        ):
            split_metrics.append(
                self._get_experimen_metrics(
                    df=df[target_fields],
                    split_result=split_results[experiment_index],
                    target_fields=target_fields,
                    alpha=alpha,
                )
            )

        df_metrics = pd.DataFrame(split_metrics)

        best_metric: pd.Series = df_metrics.loc[df_metrics["mean_test_score"].idxmax()]
        best_split: pd.Series = split_results[best_metric.experiment_index]

        result_df: pd.DataFrame = df if inPlace == True else df.copy()
        result_df[group_column_name] = best_split

        def _plot_distributions():
            tests = ["t-test", "ks-test"]

            mean_of_means_pvalue = [
                f"mean of means {test_name} p-value" for test_name in tests
            ]
            mean_of_means_title = (
                "Distribution of averages of average p-values ​​of each target_field"
            )
            self._plot_distribution(
                df=df_metrics,
                plots_group_name=mean_of_means_title,
                fields=mean_of_means_pvalue,
            )

            each_field_mean_pvalue = []
            for test in tests:
                for field in target_fields:
                    each_field_mean_pvalue.append(f"{field} mean {test} p-value")
            each_field_mean_title = (
                "Distribution of average p-values ​​of each target_field"
            )
            self._plot_distribution(
                df=df_metrics,
                plots_group_name=each_field_mean_title,
                fields=each_field_mean_pvalue,
            )

            each_field_pvalue = []
            for test in tests:
                for field in target_fields:
                    for column_name in df_metrics.columns:
                        if f"{field} {test} p-value" in column_name:
                            each_field_pvalue.append(column_name)

            each_field_title = "Distribution of p-value for each group combination for each target_field"
            self._plot_distribution(
                df=df_metrics,
                plots_group_name=each_field_title,
                fields=each_field_pvalue,
            )
            pass

        if show_plots:
            _plot_distributions()

        result = {
            "best metric": best_metric,
            "best split": best_split,
            "all metrics": df_metrics,
            "all splits": split_results,
            "best split DataFrame": result_df,
            "get_resume": _plot_distributions,
        }
        return result
