from hypex.comparators.abstract import MatchingComparator
from hypex.dataset import Dataset


class PSI(MatchingComparator):
    """Calculates population stability index for different categories of data.

    For numeric data the class generates numeric buckets, except when numeric column
    includes only NULL. For categorical data:
    1. For n < 20, a bucket equals the proportion of each category,
    2. For n > 20, a bucket equals to a group of categories,
    3. For n > 100, it calculates unique_index based on Jaccard similarity,
        but in case of imbalance null-good data returns PSI

    Args:
        expected:
            The expected values
        actual:
            The actual values
        column_name:
            The column name for which to calculate the PSI
        plot:
            If true, generates a distribution plot. Defaults to False

    Returns:
        PSI for column
        The PSI for each bucket
        New categories (empty list for non-categorical data)
        Categories that are absent in actual column (empty list for non-categorical data)
    """

    def __init__(self, expected: Dataset, actual: Dataset, column_name: str):
        """Initializes the PSI class with given parameters.

        Args:
            expected:
                The expected values
            actual:
                The actual values
            column_name:
                The column name for which to calculate the PSI
        """
        self.expected = expected
        self.actual = actual
        self.expected_len = len(self.expected)
        self.actual_len = len(self.actual)
        self.column_name = column_name
        self.column_type = self.expected.roles[0].data_type
        self.expected_shape = len(self.expected.columns)
        self.expected_nulls = self.expected.isna().sum()
        self.actual_nulls = self.actual.isna().sum()
        self.axis = 1
        if isinstance(self.column_type, str):
            self.expected_uniqs = expected[column_name].unique()
            self.actual_uniqs = actual[column_name].unique()

    def jac(self) -> float:
        """Calculates the Jacquard similarity index.

        The Jacquard similarity index measures the intersection between two sequences
        versus the union of the two sequences

        Returns:
            The Jacquard similarity index

        """
        x = set(self.expected_uniqs)
        y = set(self.expected_uniqs)

        logger.info(
            f"Jacquard similarity is {len(x.intersection(y)) / len(x.union(y)): .6f}"
        )

        jac_sim_index = len(x.intersection(y)) / len(x.union(y))

        return jac_sim_index

    def sub_psi(self, e_perc: float, a_perc: float) -> float:
        """Calculates the sub PSI value.

        Args:
            e_perc:
                The expected percentage
            a_perc:
                The actual percentage

        Returns:
            The calculated sub PSI value.
        """
        if a_perc == 0:
            a_perc = 0.0001
        if e_perc == 0:
            e_perc = 0.0001

        sub_psi = (e_perc - a_perc) * np.log(e_perc / a_perc)

        logger.debug(f"sub_psi value is {sub_psi: .6f}")

        return sub_psi

    def psi_num(self):
        """Calculate the PSI for a single variable.

        Returns:
            PSI for column
            The PSI for each bucket
            New categories (empty list for non-categorical data)
            Categories that are absent in actual column (empty list for non-categorical data)

        """
        buckets = 10
        breakpoints = np.arange(0, buckets / 10, 0.1)

        # Заплатка, на случай, если в актуальной таблице появились значения отличные от null
        if (
            self.expected_nulls == self.expected_len
            and self.actual_nulls != self.actual_len
        ):
            breakpoints = np.array(list(sorted(set(Я(self.actual, breakpoints)))))
        else:
            breakpoints = np.array(
                list(sorted(set(np.nanquantile(self.expected, breakpoints))))
            )

        actual_nulls = self.actual_nulls / self.actual_len
        expected_nulls = self.expected_nulls / self.expected_len

        breakpoints = np.concatenate(([-np.inf], breakpoints, [np.inf]))

        expected_percents = np.histogram(self.expected, breakpoints)
        actual_percents = np.histogram(self.actual, breakpoints)
        # breakpoints[0] = -np.inf
        # breakpoints[-1] = np.inf
        expected_percents = [p / self.expected_len for p in expected_percents[0]]
        actual_percents = [p / self.actual_len for p in actual_percents[0]]

        if self.expected_nulls == 0 and actual_nulls == expected_nulls:
            expected_percents = expected_percents
            actual_percents = actual_percents
            nulls = False
        else:
            expected_percents.append(expected_nulls)
            actual_percents.append(actual_nulls)
            nulls = True

        points = [i for i in breakpoints]
        intervals = [
            f"({np.round(points[i], 5)};{np.round(points[i + 1], 5)})"
            for i in range(len(points) - 1)
        ]
        if nulls:
            intervals = np.append(intervals, "empty_values")

        if self.plot:
            self.plots(
                expected_percents, actual_percents, breakpoints, intervals
            )  # в функции нет аргумента nulls

        psi_dict = {}
        for i in range(0, len(expected_percents)):
            psi_val = self.sub_psi(expected_percents[i], actual_percents[i])
            psi_dict.update({intervals[i]: psi_val})

        psi_value = np.sum(list(psi_dict.values()))
        psi_dict = {
            k: v for k, v in sorted(psi_dict.items(), key=lambda x: x[1], reverse=True)
        }
        new_cats = []
        abs_cats = []

        return psi_value, psi_dict, new_cats, abs_cats

    def uniq_psi(self):
        """Calculates PSI for categorical unique counts grater than 100.

        Returns:
            PSI for column
            The PSI for each bucket
            New categories (empty list for non-categorical data)
            Categories that are absent in actual column (empty list for non-categorical data)

        """
        actual_nulls = self.actual_nulls / self.actual_len
        expected_nulls = self.expected_nulls / self.expected_len

        actual_not_nulls_arr = self.actual[~np.isnan(self.actual)]
        expected_not_nulls_arr = self.expected[~np.isnan(self.expected)]

        actual_not_nulls = len(actual_not_nulls_arr) / self.actual_len
        expected_not_nulls = len(expected_not_nulls_arr) / self.expected_len

        expected_percents = [expected_not_nulls, expected_nulls]
        actual_percents = [actual_not_nulls, actual_nulls]

        breakpoints = ["good_data", "nulls"]
        if self.plot:
            self.plots(
                expected_percents, actual_percents, breakpoints, breakpoints
            )  # в функции нет аргумента nulls

        psi_dict = {}
        for i in range(0, len(expected_percents)):
            psi_val = self.sub_psi(expected_percents[i], actual_percents[i])
            if breakpoints[i] == "None":
                psi_dict.update({"empty_value": psi_val})
            else:
                psi_dict.update({breakpoints[i]: psi_val})

        psi_value = np.sum(list(psi_dict.values()))
        jac_metric = self.jac()
        new_cats, abs_cats = [], []
        psi_dict = {
            k: v for k, v in sorted(psi_dict.items(), key=lambda x: x[1], reverse=True)
        }

        if psi_value >= 0.2:  # что такое 0.2? Может перенести его в константу?
            psi_value = psi_value
            psi_dict.update({"metric": "stability_index"})
        else:
            psi_value = 1 - jac_metric
            psi_dict.update({"metric": "unique_index"})

        logger.info(f"PSI for categorical unique >100 is {psi_value: .6f}")

        return psi_value, psi_dict, new_cats, abs_cats

    def psi_categ(self):
        """Calculates PSI for categorical data excluding unique counts grater than 100.

        Returns:
            PSI for column
            The PSI for each bucket
            New categories (empty list for non-categorical data)
            Categories that are absent in actual column (empty list for non-categorical data)

        """
        expected_uniq_count = len(self.expected_uniqs)
        actual_uniq_count = len(self.actual_uniqs)
        # правило для категориальных > 100
        if expected_uniq_count > 100 or actual_uniq_count > 100:
            psi_value, psi_dict, new_cats, abs_cats = self.uniq_psi()

            logger.info(f"PSI is {psi_value: .6f}")

            return psi_value, psi_dict, new_cats, abs_cats

        expected_dict = (
            pd.DataFrame(self.expected, columns=[self.column_name])
            .groupby(self.column_name)[self.column_name]
            .count()
            .sort_values(ascending=False)
            .to_dict()
        )
        actual_dict = (
            pd.DataFrame(self.actual, columns=[self.column_name])
            .groupby(self.column_name)[self.column_name]
            .count()
            .sort_values(ascending=False)
            .to_dict()
        )

        breakpoints = list(expected_dict.keys() | actual_dict.keys())

        new_cats = [k for k in actual_dict.keys() if k not in expected_dict.keys()]
        abs_cats = [k for k in expected_dict.keys() if k not in actual_dict.keys()]

        expected_dict_re = {}
        actual_dict_re = {}

        for b in breakpoints:
            if b in expected_dict and b not in actual_dict:
                expected_dict_re.update({b: expected_dict[b]})
                actual_dict_re.update({b: 0})
            elif b not in expected_dict and b in actual_dict:
                expected_dict_re.update({b: 0})
                actual_dict_re.update({b: actual_dict[b]})
            elif b in expected_dict and b in actual_dict:
                actual_dict_re.update({b: actual_dict[b]})
                expected_dict_re.update({b: expected_dict[b]})

        category_names = [c for c in expected_dict_re.keys()]
        groups = {}
        g_counts = len(category_names)
        group_num = 20
        if g_counts <= group_num:
            for g_n, val in enumerate(category_names):
                groups[val] = g_n
        else:
            group_size = np.floor(g_counts / group_num)
            current_pos = 0
            reminder = g_counts % group_num
            for g_n in range(group_num):
                if g_n < group_num - reminder:
                    group_values = category_names[
                        int(current_pos) : int(current_pos + group_size)
                    ]
                    current_pos += group_size
                else:
                    group_values = category_names[
                        int(current_pos) : int(current_pos + group_size + 1)
                    ]
                    current_pos += group_size + 1
                for val in group_values:
                    groups[val] = g_n
        group_sum_exp = 0
        group_sum_act = 0
        exp_dict = {}
        act_dict = {}
        group_re = -1
        cat_group_name = ""
        group_name_re = ""
        for k, v in groups.items():
            current_group = v
            if current_group == group_re:
                group_re = v
                exp_dict.pop(group_name_re, None)
                act_dict.pop(group_name_re, None)
                cat_group_name = cat_group_name + ", " + str(k)
                group_sum_exp += expected_dict_re[k]
                group_sum_act += actual_dict_re[k]
                exp_dict.update({cat_group_name: group_sum_exp})
                act_dict.update({cat_group_name: group_sum_act})
                group_name_re = cat_group_name
            else:
                group_name_re = str(k)
                group_re = v
                cat_group_name = str(k)
                group_sum_exp = expected_dict_re[k]
                group_sum_act = actual_dict_re[k]
                exp_dict.update({cat_group_name: group_sum_exp})
                act_dict.update({cat_group_name: group_sum_act})

        expected_percents = [e / self.expected_len for e in exp_dict.values()]
        actual_percents = [a / self.actual_len for a in act_dict.values()]

        breakpoints = [e for e in exp_dict.keys()]

        if self.plot:
            self.plots(
                expected_percents, actual_percents, breakpoints, breakpoints
            )  # в функции plots нет аргумента nulls

        psi_dict = {}
        for i in range(0, len(expected_percents)):
            psi_val = self.sub_psi(expected_percents[i], actual_percents[i])
            if breakpoints[i] == "None":
                psi_dict.update({"empty_value": psi_val})
            else:
                psi_dict.update({breakpoints[i]: psi_val})
        psi_value = np.sum(list(psi_dict.values()))
        psi_dict = {
            k: v for k, v in sorted(psi_dict.items(), key=lambda x: x[1], reverse=True)
        }

        return psi_value, psi_dict, new_cats, abs_cats

    def psi_result(self):
        """Calculates PSI.

        Returns:
            PSI for column
            The PSI for each bucket
            New categories (empty list for non-categorical data)
            Categories that are absent in actual column (empty list for non-categorical data)

        """
        if len(self.expected_shape) == 1:
            psi_values = np.empty(len(self.expected_shape))
        else:
            psi_values = np.empty(self.expected_shape[self.axis])

        for i in range(0, len(psi_values)):
            if (self.column_type == np.dtype("O")) or (
                self.expected_nulls == self.expected_len
                and self.actual_nulls == self.actual_len
            ):
                psi_values, psi_dict, new_cats, abs_cats = self.psi_categ()
            else:
                psi_values, psi_dict, new_cats, abs_cats = self.psi_num()

        if self.silent:
            logger.debug(f"PSI value: {psi_values: .3f}")
        else:
            logger.info(f"PSI value: {psi_values: .3f}")

        # если expected_shape пустой - будет ошибка
        return round(psi_values, 2), psi_dict, new_cats, abs_cats


def report(expected: Dataset, actual: Dataset) -> Dataset:
    """Generates a report using PSI (Population Stability Index)  between the expected and actual data.

    Args:
        expected:
            The expected dataset
        actual:
            The new dataset you want to compare to the expected one
        plot:
            If True, plots the PSI are created. Defaults to False
        silent:
            If silent, logger in info mode


    Returns:
        A dataframe with the PSI report. The report includes the columns names,
        metric names, check results, failed buckets, new categories and disappeared categories.
        Anomaly score represent the PSI, metrics names indicate with metric was used for PSI calculation,
        check results indicate whether the PSI is under the threshold (0.2),
        and failed buckets include up to 5 buckets with the highest PSI.

    """

    data_cols = expected.columns
    score_dict = {}
    new_cat_dict = {}
    datas = []

    for col in data_cols:
        psi_res = PSI(expected, actual, col)
        score, psi_dict, new_cats, abs_cats = psi_res.psi_result()

        if len(new_cats) > 0:
            new_cat_dict.update({col: new_cats})

        score_dict.update({col: score})
        check_result = "OK" if score < 0.2 else "NOK"  # может 0.2 вынести в константу?
        # psi_dict = {k:v for k,v in sorted(psi_dict.items(), key=lambda x: x[1], reverse=True)}
        failed_buckets = list(psi_dict.keys())[:5] if score > 0.2 else []
        if "metric" in psi_dict:
            new_cats = None
            abs_cats = None
            metric_name = psi_dict["metric"]
            if metric_name == "unique_index":
                failed_buckets = None
        else:
            metric_name = "stability_index"
        data_tmp = pd.DataFrame(
            {
                "column": col,
                "anomaly_score": score,
                "metric_name": metric_name,
                "check_result": check_result,
                "failed_bucket": f"{failed_buckets}",
                "new_category": f"{new_cats}",
                "disappeared_category": f"{abs_cats}",
            },
            index=[1],
        )
        datas.append(data_tmp)

    data = pd.concat(datas, ignore_index=True)

    return data
