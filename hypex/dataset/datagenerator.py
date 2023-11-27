import string
from typing import Union
import numpy as np
import pandas as pd
import random


def to_categories(arr: np.ndarray, num_vars: int, num_discrete_vars: int) -> np.ndarray:
    """
    Converts some columns of array of floats to integers

    Args:
        arr:
            Array of floats with data
        num_vars:
            Total number of features
        num_discrete_vars:
            Number of discrete values

    Returns:
        arr_with_dummy:
            Array of floats with integer values
    """
    quantiles = np.array([0.25, 0.5, 0.75])
    arr_with_dummy = arr.copy()
    for arr_index in range(num_vars - num_discrete_vars, num_vars):
        arr_bins = np.quantile(arr[:, arr_index], q=quantiles)
        arr_categorical = np.searchsorted(arr_bins, arr[:, arr_index], 'right')
        arr_with_dummy = np.concatenate((arr_with_dummy, arr_categorical[:, np.newaxis]), axis=1)
    for arr_index in range(num_vars - 1, num_vars - num_discrete_vars - 1, -1):
        arr_with_dummy = np.delete(arr_with_dummy, arr_index, axis=1)
    return arr_with_dummy


class DataGenerator:
    """
    Class for generation dataset with multiple columns.

    Examples:

        >>> # Base generation
        >>> sample_data = DataGenerator()
        >>>
        >>> print(sample_data.df)
        >>>
        >>> # Optional
        >>> num_records = 10000 # Number of records to generate
        >>> num_features = 3 # Number of columns correlating with outcome
        >>> num_info_cols = 3 # Number of columns with information about each record
        >>> is_treatment = False #
        >>> num_targets = 4 # Number of columns with target values
        >>> num_discrete_features = 0 # Number of discrete columns correlating with outcome
        >>> is_binary_target = False # If outcome should be binary
        >>> na_columns = ['feature_1', 'feature_2'] # List of columns with NA values
        >>> na_step = 15 # Number of period to make NaN (step of range)
        >>>
        >>> sample_data = DataGenerator(num_records=num_records,num_features=num_features,num_info_cols=num_info_cols,is_treatment=is_treatment,num_targets=num_targets,is_binary_outcome=is_binary_outcome,na_columns=na_columns,na_step=na_step)
        >>>
        >>> print(sample_data.df)
        >>>
        >>> # Without treatment
        >>> sample_data = DataGenerator(is_treatment=False)
        >>>
        >>> print(sample_data.df)
        >>>
        >>>
        >>> print(sample_data) # More details about attributes
    """

    def __init__(
            self,
            num_records: int = 5000,
            num_features: int = 4,
            num_info_cols: int = 2,
            num_discrete_features: int = 1,
            num_targets: int = 1,
            is_treatment: bool = True,
            is_binary_target: bool = False,
            na_columns: Union[str, list, None] = None,
            na_step: int = 0) -> None:
        """
        Initialize the DataGenerator object.

        Args:
            num_records:
                Number of records in created dataset. Defaults to 5000
            num_features:
                Number of columns correlating with outcome. Defaults to 4
            num_info_cols:
                Number of columns with information about each record. Defaults to 2
            is_treatment:
                Number of treatments (info about 'treatment' 0 or 1). Defaults to 1
            num_discrete_features:
                Number of discrete columns correlating with outcome. Defaults to 1
            num_targets:
                Number of columns with target values. Defaults to 1
            is_binary_target:
                If outcome should be binary. Defaults to False
            na_columns:
                Column or list of columns with NA values. Defaults to None
            na_step:
                Number of period to make NaN (step of range). Defaults to 0
        """
        self.num_records = num_records
        self.num_features = num_features
        self.num_info_cols = num_info_cols
        self.is_treatment = is_treatment
        self.num_targets = num_targets
        self.num_discrete_features = num_discrete_features
        self.is_binary_target = is_binary_target
        self.na_columns = na_columns
        self.na_step = na_step
        self.df: pd.DataFrame = pd.DataFrame()
        self.treatment_name: Union[str, list, None] = None
        self.target_names: list = []
        self.info_col_names: Union[list, None] = None
        self.features_names: list = []
        self.ate: float = 0.0
        self.sigma = 1.3
        self._generate_df()

    def _generate_df(self):
        """
        Generates the dataset and predict ate for it
        """
        m_with_dummy = self._generate_feature_cols()
        info = self._generate_info_cols()
        treatment = self._generate_treatment_cols()
        data = self._generate_outcome_and_full_data(m_with_dummy, treatment)
        self._set_column_names()
        self.df = self._set_df(data, info)
        self._set_nans()
        self._process_ate()

    def _process_ate(self):
        pass

    def _generate_feature_cols(self):
        """
        Generate columns with main features for causal process

        Returns:
            m_with_dummy:
                Array with features
        """
        means = np.random.uniform(-1, 1, self.num_features)
        cov_mat = np.identity(self.num_features)
        m = np.random.multivariate_normal(means, cov_mat, self.num_records)
        m_with_dummy = to_categories(m, self.num_features, self.num_discrete_features)
        return m_with_dummy

    def _generate_info_cols(self):
        """
        Generate columns with information features for causal process

        Returns:
            info:
                List with features
        """
        info = []
        for i in range(self.num_info_cols):
            if i % 2 == 0:
                coefficient = 3 if i <= 2 else i
                array = list(range(1, self.num_records * coefficient, coefficient))
                random.shuffle(array)
                info.append(array)
            else:
                unique_letters = list(set(string.ascii_letters.upper()))
                letters = [random.choice(unique_letters) for _ in range(i * 2)]
                info.append([letters[j] * ((i // 2) + 1) for j in np.random.choice(a=np.arange(i * 2),
                                                                                   size=self.num_records)])
        return info

    def _generate_treatment_cols(self):
        """
        Generate columns with treatment values

        Returns:
            treatment:
                Array with treatment
        """
        if self.is_treatment and self.num_targets == 1:
            treatment = np.random.normal(0, self.sigma, (self.num_records, 1))
        else:
            treatment = np.random.normal(0, self.sigma, (self.num_records, self.num_targets))
        treatment = np.digitize(treatment, bins=np.array([0, 0.5, 1])).astype(bool)
        return treatment

    def _generate_y_for_target(self, m_with_dummy):
        """
        Generate data for target field
        Args:
            m_with_dummy: Array with main features

        Returns:
            y0, y1: Arrays with values for target field
        """
        m_vector = m_with_dummy.dot(np.ones(self.num_features))
        epsilon = np.random.multivariate_normal(mean=np.zeros(2),
                                                cov=np.identity(2),
                                                size=self.num_records)
        y0 = m_vector + epsilon[:, 0]
        y1 = self.sigma + m_with_dummy.dot(np.ones(self.num_features) * 2) + epsilon[:, 1]
        return y0, y1

    def _generate_outcome_and_full_data(self, m_with_dummy, treatment):
        """
        Generates outcome depending on treatment and create array with main numeric features
        Args:
            m_with_dummy: Array with main features
            treatment: Array with treatment values

        Returns:
            data: Array with main numeric features
        """
        y0, y1 = self._generate_y_for_target(m_with_dummy)
        if self.is_treatment and self.num_targets == 1:
            outcome = (1 - treatment.flatten()) * y0 + treatment.flatten() * y1
        else:
            outcome = np.multiply((1 - treatment), y0[:, np.newaxis]) + np.multiply(treatment, y1[:, np.newaxis])
            treatment = treatment[:, 0:1]

        outcome = np.digitize(outcome, bins=np.array([np.mean(outcome) - 2])) if self.is_binary_target else outcome

        if self.is_treatment:
            return np.column_stack((m_with_dummy, treatment, outcome))
        return np.column_stack((m_with_dummy, outcome))

    def _set_df(self, data: np.ndarray, info: np.ndarray):
        """
        Generate names for columns and combine all data to DataFrame
        Args:
            data:
                Array with main features
            info:
                Array with info features

        Returns:
            df:
                DataFrame with generated data
        """
        gender = [["male", "female"][i] for i in np.random.choice(a=np.array([0, 1]), size=self.num_records)]
        product = [['Deposit', 'Credit', 'Investment'][i] for i in np.random.choice(a=np.array([0, 1, 2]),
                                                                                    size=self.num_records)]
        self.treatment_name = ['treatment'] if self.is_treatment else []
        df = pd.DataFrame(data=data, columns=self.features_names + self.treatment_name + self.target_names)
        self.treatment_name = 'treatment' if self.is_treatment else []
        for i in range(self.num_info_cols):
            df.insert(i, self.info_col_names[i], pd.Series(info[i]))
        df.insert(self.num_info_cols, 'feature_1', pd.Series(gender))
        df.insert(self.num_info_cols + 1, 'feature_2', pd.Series(product))
        self.features_names = ['feature_1', 'feature_2'] + self.features_names
        return df

    def _set_column_names(self):
        """
        Set names for columns in DataFrame
        """
        self.features_names = ['feature_' + str(i) for i in range(3, self.num_features + 3)]
        self.treatment_name = ['treatment'] if self.is_treatment else []
        self.target_names = ['target_' + str(i) for i in range(1, self.num_targets + 1)] if self.num_targets > 1 else [
            'target']
        self.info_col_names = ['info'] if self.num_info_cols == 1 else ['info_' + str(i) for i in
                                                                        range(1, self.num_info_cols + 1)]

    def _set_nans(self):
        """
        Set NaN values to DataFrame according to their respective
        """
        if self.na_columns or self.na_step:
            self.na_step = [self.na_step] if self.na_step else [10]
            self.na_columns = self.na_columns if self.na_columns else self.features_names + self.info_col_names
            self.na_columns = [self.na_columns] if isinstance(self.na_columns, str) else self.na_columns
            if not set(self.na_columns).issubset(self.df.columns):
                raise KeyError(f'There is no columns {self.na_columns} in data. Only {list(self.df.columns)} provided.')
            self.na_step = self.na_step[:len(self.na_columns)] if len(self.na_step) > len(
                self.na_columns) else self.na_step + [self.na_step[-1]] * (len(self.na_columns) - len(self.na_step))
            nans_indexes = [list(range(i, len(self.df), period)) for i, period in enumerate(self.na_step)]
            for i in range(len(self.na_columns)):
                self.df.loc[nans_indexes[i], self.na_columns[i]] = np.nan

    def __repr__(self):
        return """
Attributes:
---------------------------------------------------------------
df - generated dataset
target_name - name for column with treatment
target_names - names for columns with outcome
info_col_names - names for columns with information about records
features_names - names for columns with main features for process
ate - predicted ate for sample data
--------------------------------------------
    """