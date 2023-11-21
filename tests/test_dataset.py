import unittest
from hypex.dataset.dataset import Dataset

class TestDataset(unittest.TestCase):
    def test_simple_data__generate_df(self):
        data = Dataset()
        self.assertEqual(len(data.df.columns), 10)
        self.assertEqual(len(data.main_causes_names), data.num_main_causes_cols + 2)
        self.assertEqual(len(data.outcome_name), data.num_outcomes)
        self.assertEqual(len(data.treatment_name), data.num_treatments)
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.outcome_1.dtype, 'float64')
        self.assertEqual(list(set(data.df.treatment_1)), [0.0, 1.0])
        print()

    def test_more_records__generate_df(self):
        data = Dataset(num_records=10000)
        self.assertEqual(len(data.df.columns), 10)
        self.assertEqual(len(data.main_causes_names), data.num_main_causes_cols + 2)
        self.assertEqual(len(data.outcome_name), data.num_outcomes)
        self.assertEqual(len(data.treatment_name), data.num_treatments)
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.outcome_1.dtype, 'float64')
        self.assertEqual(list(set(data.df.treatment_1)), [0.0, 1.0])
        print()

    def test_less_records__generate_df(self):
        data = Dataset(num_records=1000)
        self.assertEqual(len(data.df.columns), 10)
        self.assertEqual(len(data.main_causes_names), data.num_main_causes_cols + 2)
        self.assertEqual(len(data.outcome_name), data.num_outcomes)
        self.assertEqual(len(data.treatment_name), data.num_treatments)
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.outcome_1.dtype, 'float64')
        self.assertEqual(list(set(data.df.treatment_1)), [0.0, 1.0])
        print()

    def test_more_features__generate_df(self):
        data = Dataset(num_main_causes_cols=6)
        self.assertEqual(len(data.df.columns), 12)
        self.assertEqual(len(data.main_causes_names), data.num_main_causes_cols + 2)
        self.assertEqual(len(data.outcome_name), data.num_outcomes)
        self.assertEqual(len(data.treatment_name), data.num_treatments)
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.outcome_1.dtype, 'float64')
        self.assertEqual(list(set(data.df.treatment_1)), [0.0, 1.0])
        print()

    def test_less_features__generate_df(self):
        data = Dataset(num_main_causes_cols=2)
        self.assertEqual(len(data.df.columns), 8)
        self.assertEqual(len(data.main_causes_names), data.num_main_causes_cols + 2)
        self.assertEqual(len(data.outcome_name), data.num_outcomes)
        self.assertEqual(len(data.treatment_name), data.num_treatments)
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.outcome_1.dtype, 'float64')
        self.assertEqual(list(set(data.df.treatment_1)), [0.0, 1.0])
        print()

    def test_more_info__generate_df(self):
        data = Dataset(num_info_cols=6)
        self.assertEqual(len(data.df.columns), 14)
        self.assertEqual(len(data.main_causes_names), data.num_main_causes_cols + 2)
        self.assertEqual(len(data.outcome_name), data.num_outcomes)
        self.assertEqual(len(data.treatment_name), data.num_treatments)
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.outcome_1.dtype, 'float64')
        self.assertEqual(list(set(data.df.treatment_1)), [0.0, 1.0])
        print()

    def test_less_info__generate_df(self):
        data = Dataset(num_info_cols=0)
        self.assertEqual(len(data.df.columns), 8)
        self.assertEqual(len(data.main_causes_names), data.num_main_causes_cols + 2)
        self.assertEqual(len(data.outcome_name), data.num_outcomes)
        self.assertEqual(len(data.treatment_name), data.num_treatments)
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.outcome_1.dtype, 'float64')
        self.assertEqual(list(set(data.df.treatment_1)), [0.0, 1.0])
        print()

    def test_less_treatment__generate_df(self):
        data = Dataset(num_treatments=0)
        self.assertEqual(len(data.df.columns), 9)
        self.assertEqual(len(data.main_causes_names), data.num_main_causes_cols + 2)
        self.assertEqual(len(data.outcome_name), data.num_outcomes)
        self.assertEqual(len(data.treatment_name), data.num_treatments)
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.outcome_1.dtype, 'float64')
        print()

    def test_more_outcome__generate_df(self):
        data = Dataset(num_outcomes=2)
        self.assertEqual(len(data.df.columns), 11)
        self.assertEqual(len(data.main_causes_names), data.num_main_causes_cols + 2)
        self.assertEqual(len(data.outcome_name), data.num_outcomes)
        self.assertEqual(len(data.treatment_name), data.num_treatments)
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.outcome_1.dtype, 'float64')
        print()

    def test_more_outcome_less_treatment__generate_df(self):
        data = Dataset(num_outcomes=4, num_treatments=0)
        self.assertEqual(len(data.df.columns), 12)
        self.assertEqual(len(data.main_causes_names), data.num_main_causes_cols + 2)
        self.assertEqual(len(data.outcome_name), data.num_outcomes)
        self.assertEqual(len(data.treatment_name), data.num_treatments)
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.outcome_1.dtype, 'float64')
        print()

    def test_less_outcome_more_treatment__generate_df(self):
        data = Dataset(num_outcomes=2, num_treatments=4)
        self.assertEqual(len(data.df.columns), 14)
        self.assertEqual(len(data.main_causes_names), data.num_main_causes_cols + 2)
        self.assertEqual(len(data.outcome_name), data.num_outcomes)
        self.assertEqual(len(data.treatment_name), data.num_treatments)
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.outcome_1.dtype, 'float64')
        print()

    def test_binary_outcome__generate_df(self):
        data = Dataset(binary_outcome=True)
        self.assertEqual(len(data.df.columns), 10)
        self.assertEqual(len(data.main_causes_names), data.num_main_causes_cols + 2)
        self.assertEqual(len(data.outcome_name), data.num_outcomes)
        self.assertEqual(len(data.treatment_name), data.num_treatments)
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(list(set(data.df.outcome_1)), [0.0, 1.0])
        print()

    def test_more_binary_outcome__generate_df(self):
        data = Dataset(binary_outcome=True, num_outcomes=3)
        self.assertEqual(len(data.df.columns), 12)
        self.assertEqual(len(data.main_causes_names), data.num_main_causes_cols + 2)
        self.assertEqual(len(data.outcome_name), data.num_outcomes)
        self.assertEqual(len(data.treatment_name), data.num_treatments)
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(list(set(data.df.outcome_1)), [0.0, 1.0])
        self.assertEqual(list(set(data.df.outcome_2)), [0.0, 1.0])
        self.assertEqual(list(set(data.df.outcome_3)), [0.0, 1.0])
        print()

    def test_list_na_columns__generate_df(self):
        data = Dataset(na_columns=['feature_col_1', 'feature_col_2'])
        self.assertEqual(len(data.df.columns), 10)
        self.assertEqual(len(data.main_causes_names), data.num_main_causes_cols + 2)
        self.assertEqual(len(data.outcome_name), data.num_outcomes)
        self.assertEqual(len(data.treatment_name), data.num_treatments)
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.feature_col_1.isna().sum(), 500)
        self.assertEqual(data.df.feature_col_2.isna().sum(), 500)
        self.assertEqual(data.df.feature_col_3.isna().sum(), 0)
        print()

    def test_one_na_column__generate_df(self):
        data = Dataset(na_columns='feature_col_1')
        self.assertEqual(len(data.df.columns), 10)
        self.assertEqual(len(data.main_causes_names), data.num_main_causes_cols + 2)
        self.assertEqual(len(data.outcome_name), data.num_outcomes)
        self.assertEqual(len(data.treatment_name), data.num_treatments)
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.feature_col_1.isna().sum(), 500)
        self.assertEqual(data.df.feature_col_2.isna().sum(), 0)
        print()

    def test_na_step__generate_df(self):
        data = Dataset(na_step=5, na_columns='feature_col_1')
        self.assertEqual(len(data.df.columns), 10)
        self.assertEqual(len(data.main_causes_names), data.num_main_causes_cols + 2)
        self.assertEqual(len(data.outcome_name), data.num_outcomes)
        self.assertEqual(len(data.treatment_name), data.num_treatments)
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.feature_col_1.isna().sum(), 1000)
        self.assertEqual(data.df.feature_col_2.isna().sum(), 0)
        print()

    def test_not_df_column__generate_df(self):
        with self.assertRaises(KeyError):
            data = Dataset(na_columns='feature_col_10')
        print()


if __name__ == '__main__':
    unittest.main()