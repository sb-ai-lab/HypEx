import warnings
from itertools import combinations
from pathlib import Path
from typing import Iterable, Union, Optional, Dict, Any, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from scipy.stats import ttest_ind, ks_2samp, mannwhitneyu, norm
from sklearn.utils import shuffle
from statsmodels.stats.power import TTestIndPower
from tqdm.auto import tqdm


def merge_groups(
        control_group: pd.DataFrame,
        test_group: pd.DataFrame,
) -> pd.DataFrame:
    """Merges test and control groups in one DataFrame and creates column "group".

    Column "group" contains of "test" and "control" values.

    Args:
        control_group: Data of control group
        test_group: Data of target group
    Returns:
        merged_data: Contacted DataFrame
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
        test_group: The test group as a pandas Series
        control_group: The control group as a pandas Series
        reliability: The reliability of the test
        power: The power of the test

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
        s = test_std ** 2 / test_proportion + control_std ** 2 / control_proportion
        return d * s


def calc_power(
        effect_size: float,
        control_size: float,
        test_size: float,
        significance: float = 0.05,
) -> float:
    analysis = TTestIndPower()
    ratio = test_size / control_size
    return analysis.power(
        effect_size=effect_size,
        nobs1=test_size,
        ratio=ratio,
        alpha=significance,
    )


class AATest:
    def __init__(
            self,
            target_fields: Union[Iterable[str], str] = None,
            group_cols: Union[str, Iterable[str]] = None,
            info_cols: Union[str, Iterable[str]] = None,
            quant_field: str = None,
            mode: str = "simple",
            alpha: float = 0.05,
    ):
        self.target_fields = (
            [target_fields] if isinstance(target_fields, str) else target_fields
        )
        self.group_cols = ([group_cols] if isinstance(group_cols, str) else group_cols) or []
        self.info_cols = [info_cols] if isinstance(info_cols, str) else info_cols
        self.quant_field = quant_field
        self.mode = mode
        self.alpha = alpha

    def columns_labeling(self, data: pd.DataFrame) -> Dict[str, List[str]]:
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
            data: Input data
            random_state: Seed of random

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
            data: Raw input data
            random_state: Seed of random - one integer to fix split
            test_size: Group size ratio

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

        Adds info_cols and decode binary variables

        Args:
            data: Raw input data
            spit_indexes: Dict of indexes with separation on test and control group

        Returns:
            Separated initial data with column "group"
        """
        # prep data to show user (add info_cols and decode binary variables)
        test = data.loc[spit_indexes["test_indexes"]]
        control = data.loc[spit_indexes["control_indexes"]]
        data = merge_groups(test, control)

        return data

    @staticmethod
    def calc_ab_delta(a_mean: float, b_mean: float, mode: str = "percentile"):
        """Calculates target delta between A and B groups.

        Args:
            a_mean: Average of target in one group
            b_mean: Average of target in another group
            mode: Type of expected result:
                'percentile' - percentage exceeding the average in group A compared to group B
                'absolute' - absolute value of difference between B and A group
                'relative' - percent in format of number (absolute) exceeding the average in group A compared to group B

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
    ):
        """Calculates metrics of one sampling.

        Args:
            data: Raw input data
            random_state: Random seeds for searching
            preprocessed_data: Pre-preprocessed data

        Returns:
            Tuple of
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
            ta, tb = a[tf], b[tf]

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
        t_result["control size"], t_result["test size"] = len(a), len(b)
        for test in ('t_test', 'ks-test'):
            t_result[f"{test} mean p-value"] = np.mean(
                [p_value for key, p_value in t_result.items() if f"{test} p-value" in key]
            )
            t_result[f"{test} passed %"] = np.mean(
                [passed * 100 for key, passed in t_result.items() if f"{test} passed" in key]
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
        """Chooses random_state for finding homogeneous distribution.

        Args:
            data: Raw input data
            iterations:
                Number of iterations to search uniform sampling to searching
            file_name:
                Name of file to save results (if None - no results will be saved, func returns result)
            write_mode:
                Mode to write:
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
                if experiment_write_mode in ["all", "full"]:
                    results.append(res["metrics"])
                if split_write_mode in ["all", "full"]:
                    data_from_sampling.update(res["data_from_experiment"])
            if any(passed):
                if experiment_write_mode in ["any", "full"]:
                    results.append(res["metrics"])
                if split_write_mode in ["any", "full"]:
                    data_from_sampling.update(res["data_from_experiment"])

            if file_name and write_step:
                if i == write_step:
                    pd.DataFrame(results).to_csv(file_name, index=False)
                elif i % write_step == 0:
                    pd.DataFrame(results).to_csv(
                        file_name, index=False, header=False, mode="a"
                    )
                    results = []

        if file_name and write_step:
            pd.DataFrame(results).to_csv(file_name, index=False, header=False, mode="a")
        elif file_name:
            results = pd.DataFrame(results)
            results.to_csv(file_name, index=False)
            return results, data_from_sampling
        else:
            return pd.DataFrame(results), data_from_sampling

    def features_p_value_distribution(
            self, experiment_results: pd.DataFrame, figsize=None, bin_step=0.05
    ):
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

        for test in ('t-test', 'ks-test'):
            result[f"{test} aa passed"] = result[f"{test} passed score"].apply(
                lambda x: 0.8 * self.alpha <= x <= 1.2 * self.alpha
            )
        result.loc["mean"] = result.mean()

        return result

    def uniform_tests_interpretation(
            self, experiment_results: pd.DataFrame, **kwargs
    ) -> pd.DataFrame:
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
        s_control_data = control_data.astype("str")
        s_test_data = test_data.astype("str")

        figsize = kwargs.get("figsize", (15, 10))
        figure, ax = plt.subplots(
            nrows=1, ncols=1, figsize=figsize, facecolor="honeydew", edgecolor="black"
        )

        control_counts = s_control_data.value_counts(normalize=True) * 100
        test_counts = s_test_data.value_counts(normalize=True) * 100

        ax.fill_between(
            s_control_data.index,
            s_control_data.values,
            color="blue",
            alpha=0.3,
            label="control",
        )
        ax.fill_between(
            s_control_data.index,
            test_counts[[i for i in test_counts.index if i in control_counts.index]].values,
            color="red", alpha=0.3, label="test"
        )

        ax.legend()
        ax.tick_params(axis="x", rotation=90)
        figure.suptitle(f"{control_data.name}", fontsize=kwargs.get("title_size", 20))
        plt.show()

    def split_analysis(self, splited_data: pd.DataFrame, **kwargs):
        ssp = split_splited_data(splited_data)
        for nf in self.target_fields:
            self.num_feature_uniform_analysis(
                ssp["control"][nf], ssp["test"][nf], **kwargs
            )
        for cf in self.group_cols:
            self.cat_feature_uniform_analysis(
                ssp["control"][cf], ssp["test"][cf], **kwargs
            )

    def process(self, data: pd.DataFrame, optimize_groups: bool = False, iterations: int = 2000, show_plots=True,
                **kwargs):
        labeling = self.columns_labeling(data)
        best_results, best_split = None, None

        if not self.target_fields:
            self.target_fields = labeling["target_fields"]

        if optimize_groups:
            max_score = -1

            group_variants = [[]]
            for i in range(1, len(labeling["group_col"])):
                i_combinstions = combinations(labeling["group_col"], i)
                group_variants.extend(iter(i_combinstions))

            for gs in tqdm(group_variants, desc="Group optimization"):
                self.group_cols = list(gs)
                experiment_results, data_splits = self.calc_uniform_tests(
                    data, pbar=False, iterations=iterations, **kwargs
                )
                if len(experiment_results):
                    aa_scores = self.aa_score(experiment_results)
                    group_score = max(
                        aa_scores.loc["mean", "t-test aa passed"],
                        aa_scores.loc["mean", "t-test aa passed"],
                    )
                    if group_score > max_score:
                        best_results, best_split = experiment_results, data_splits
                        max_score = group_score

        else:
            best_results, best_split = self.calc_uniform_tests(
                data, experiment_write_mode="full", split_write_mode="any", iterations=iterations, **kwargs
            )

        if len(best_results) == 0:
            return best_results, best_split
        if show_plots:
            aa_scores = self.uniform_tests_interpretation(best_results)
        else:
            aa_scores = self.aa_score(best_results)
        best_rs = best_results.loc[
            best_results["mean_tests_score"].idxmax(), "random_state"
        ]
        if show_plots:
            self.split_analysis(best_split[best_rs], **kwargs)
        return best_results, best_split[best_rs]


class ABTest:
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
                Named tuple with pvalue, effect, ci_length, left_bound and right_bound
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
            test_data: Input data of test group
            control_data: Input data of control group
            target_field: Column name of target after pilot
            target_field_before: Column name of target before pilot

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
        """Calculates the p-value for a given data set.

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
            data: Input data as a pandas DataFrame
            target_field: Target field to be analyzed
            group_field: Field used to split the data into groups
            target_field_before: Target field without treatment to be analyzed

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
