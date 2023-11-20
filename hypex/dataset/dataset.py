import string
import numpy as np
import pandas as pd
import random


def to_categories(arr, num_vars, num_discrete_vars):
    quantiles = np.array([0.25, 0.5, 0.75])
    arr_with_dummy = arr.copy()
    for arr_index in range(num_vars - num_discrete_vars, num_vars):
        arr_bins = np.quantile(arr[:, arr_index], q=quantiles)
        arr_categorical = np.searchsorted(arr_bins, arr[:, arr_index], 'right')
        arr_with_dummy = np.concatenate((arr_with_dummy, arr_categorical[:, np.newaxis]), axis=1)
    for arr_index in range(num_vars - 1, num_vars - num_discrete_vars - 1, -1):
        arr_with_dummy = np.delete(arr_with_dummy, arr_index, axis=1)
    return arr_with_dummy


class Dataset:
    def __init__(
            self,
            num_records: int = 5000,
            num_main_causes_cols: int = 4,
            num_info_cols: int = 2,
            num_treatments: int = 1,
            num_discrete_main_causes_cols: int = 1,
            num_outcomes: int = 1,
            binary_outcome: bool = False,
            na_columns: str | list = None,
            na_step: int = None) -> None:
        self.num_records = num_records
        self.num_main_causes_cols = num_main_causes_cols
        self.num_info_cols = num_info_cols
        self.num_treatments = num_treatments
        self.num_outcomes = num_outcomes
        self.num_discrete_main_causes_cols = num_discrete_main_causes_cols
        self.binary_outcome = binary_outcome
        self.na_columns = na_columns
        self.na_step = na_step
        self.treatment_name: str | list = ''
        self.outcome_name: str | list = ''
        self.info_col_names: str | list = ''
        self.main_causes_names: str | list = ''
        self.ate: float = 0.0
        self.sigma = 1.3
        self.__generate_df__()

    def __generate_df__(self):
        m_with_dummy = self.generate_feature_cols()
        info = self.generate_info_cols()
        treatment = self.generate_treatment_cols()

        m_vector = m_with_dummy.dot(np.ones(self.num_main_causes_cols))
        epsilon = np.random.multivariate_normal(mean=np.zeros(2),
                                                cov=np.identity(2),
                                                size=self.num_records)
        y0 = m_vector + epsilon[:, 0]
        beta = np.ones(self.num_main_causes_cols) * 2
        y1 = self.sigma + m_with_dummy.dot(beta) + epsilon[:, 1]
        if (self.num_treatments == 1 or self.num_treatments == 0) and self.num_outcomes == 1:
            outcome = (1 - treatment.flatten()) * y0 + treatment.flatten() * y1
        else:
            outcome = np.multiply((1 - treatment), y0[:, np.newaxis]) + np.multiply(treatment, y1[:, np.newaxis])
            if self.num_outcomes > self.num_treatments:
                treatment = treatment[:, 0:self.num_treatments]
            else:
                outcome = outcome[:, 0:self.num_outcomes]

        if self.binary_outcome:
            outcome = np.digitize(outcome, bins=np.array([np.mean(outcome) - 2]))

        if self.num_treatments > 0:
            data = np.column_stack((m_with_dummy, treatment, outcome))
        else:
            data = np.column_stack((m_with_dummy, outcome))

        self.df = self.set_df(data, info)
        self.set_nans()
        # self.__process_ate__()

    # def __process_ate__(self):
    #   self.ate = CausalForestDML().fit(
    #       Y=self.df[self.outcome_name],
    #       T=self.df[self.treatment_name],
    #       X=self.df[self.main_causes_names]).marginal_ate(
    #           X=self.df[self.main_causes_names],
    #           T=self.df[self.treatment_name])

    def generate_feature_cols(self):
        means = np.random.uniform(-1, 1, self.num_main_causes_cols)
        cov_mat = np.identity(self.num_main_causes_cols)
        m = np.random.multivariate_normal(means, cov_mat, self.num_records)
        m_with_dummy = to_categories(m, self.num_main_causes_cols, self.num_discrete_main_causes_cols)
        return m_with_dummy

    def generate_info_cols(self):
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
                info.append(
                    [letters[a] * ((i // 2) + 1) for a in np.random.choice(a=np.arange(i * 2), size=self.num_records)])
        return info

    def generate_treatment_cols(self):
        if (self.num_treatments > 0 and self.num_outcomes == 1) or (self.num_treatments > self.num_outcomes):
            treatment = np.random.normal(0, 1.3, (self.num_records, self.num_treatments))
        else:
            treatment = np.random.normal(0, 1.3, (self.num_records, self.num_outcomes))
        treatment = np.digitize(treatment, bins=np.array([0, 0.5, 1])).astype(bool)
        return treatment

    def set_df(self, data, info):
        gender = [["male", "female"][i] for i in np.random.choice(a=np.array([0, 1]), size=self.num_records)]
        product = [['Deposit', 'Credit', 'Investment'][i] for i in np.random.choice(a=np.array([0, 1, 2]),
                                                                                    size=self.num_records)]
        self.main_causes_names = ['feature_col_' + str(i) for i in range(3, self.num_main_causes_cols + 3)]
        self.treatment_name = ['treatment_' + str(i) for i in range(1, self.num_treatments + 1)]
        self.outcome_name = ['outcome_' + str(i) for i in range(1, self.num_outcomes + 1)]
        self.info_col_names = ['info_col_' + str(i) for i in range(1, self.num_info_cols + 1)]
        df = pd.DataFrame(data=data, columns=self.main_causes_names + self.treatment_name + self.outcome_name)
        for i in range(self.num_info_cols):
            df.insert(i, self.info_col_names[i], pd.Series(info[i]))
        df.insert(self.num_info_cols, 'feature_col_1', pd.Series(gender))
        df.insert(self.num_info_cols + 1, 'feature_col_2', pd.Series(product))
        return df

    def set_nans(self):
        if self.na_columns or self.na_step:
            self.na_step = [self.na_step] if self.na_step else [10]
            self.na_columns = self.na_columns if self.na_columns else self.main_causes_names + self.info_col_names
            self.na_columns = [self.na_columns] if isinstance(self.na_columns, str) else self.na_columns
            self.na_step = self.na_step[:len(self.na_columns)] if len(self.na_step) > len(
                self.na_columns) else self.na_step + [self.na_step[-1]] * (len(self.na_columns) - len(self.na_step))
            nans_indexes = [list(range(i, len(self.df), period)) for i, period in enumerate(self.na_step)]
            for i in range(len(self.na_columns)):
                try:
                    self.df.loc[nans_indexes[i], self.na_columns[i]] = np.nan
                except KeyError:
                    print(f'There is no column {self.na_columns[i]} in data. No nans in this column will be added.')


    def __repr__(self):
        text = """
Attributes:
---------------------------------------------------------------
df - generated dataset
treatment_name - name for column with treatment
outcome_name - names for columns with outcome
info_col_names - names for columns with infornamion
main_causes_names - names for columns with features for process
ate - predicted ate for sample data
--------------------------------------------
    """
        return text


sample_data = Dataset(num_info_cols=0, num_treatments=3, num_outcomes=1, na_step=5)
print(sample_data.df)