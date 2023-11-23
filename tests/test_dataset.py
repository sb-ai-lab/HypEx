import unittest
from hypex.dataset.datagenerator import DataGenerator

class TestDataset(unittest.TestCase):
    def test_simple_data__generate_df(self):
        data = DataGenerator()
        self.assertEqual(len(data.df.columns), 10)
        self.assertEqual(len(data.features_names), data.num_features + 2)
        self.assertEqual(len(data.target_names), data.num_targets)
        self.assertEqual(data.treatment_name, 'treatment')
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.target.dtype, 'float64')
        self.assertEqual(list(set(data.df.treatment)), [0.0, 1.0])
        print()

    def test_more_records__generate_df(self):
        data = DataGenerator(num_records=10000)
        self.assertEqual(len(data.df.columns), 10)
        self.assertEqual(len(data.features_names), data.num_features + 2)
        self.assertEqual(len(data.target_names), data.num_targets)
        self.assertEqual(data.treatment_name, 'treatment')
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.target.dtype, 'float64')
        self.assertEqual(list(set(data.df.treatment)), [0.0, 1.0])
        print()

    def test_less_records__generate_df(self):
        data = DataGenerator(num_records=1000)
        self.assertEqual(len(data.df.columns), 10)
        self.assertEqual(len(data.features_names), data.num_features + 2)
        self.assertEqual(len(data.target_names), data.num_targets)
        self.assertEqual(data.treatment_name, 'treatment')
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.target.dtype, 'float64')
        self.assertEqual(list(set(data.df.treatment)), [0.0, 1.0])
        print()

    def test_more_features__generate_df(self):
        data = DataGenerator(num_features=6)
        self.assertEqual(len(data.df.columns), 12)
        self.assertEqual(len(data.features_names), data.num_features + 2)
        self.assertEqual(len(data.target_names), data.num_targets)
        self.assertEqual(data.treatment_name, 'treatment')
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.target.dtype, 'float64')
        self.assertEqual(list(set(data.df.treatment)), [0.0, 1.0])
        print()

    def test_less_features__generate_df(self):
        data = DataGenerator(num_features=2)
        self.assertEqual(len(data.df.columns), 8)
        self.assertEqual(len(data.features_names), data.num_features + 2)
        self.assertEqual(len(data.target_names), data.num_targets)
        self.assertEqual(data.treatment_name, 'treatment')
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.target.dtype, 'float64')
        self.assertEqual(list(set(data.df.treatment)), [0.0, 1.0])
        print()

    def test_more_info__generate_df(self):
        data = DataGenerator(num_info_cols=6)
        self.assertEqual(len(data.df.columns), 14)
        self.assertEqual(len(data.features_names), data.num_features + 2)
        self.assertEqual(len(data.target_names), data.num_targets)
        self.assertEqual(data.treatment_name, 'treatment')
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.target.dtype, 'float64')
        self.assertEqual(list(set(data.df.treatment)), [0.0, 1.0])
        print()

    def test_less_info__generate_df(self):
        data = DataGenerator(num_info_cols=0)
        self.assertEqual(len(data.df.columns), 8)
        self.assertEqual(len(data.features_names), data.num_features + 2)
        self.assertEqual(len(data.target_names), data.num_targets)
        self.assertEqual(data.treatment_name, 'treatment')
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.target.dtype, 'float64')
        self.assertEqual(list(set(data.df.treatment)), [0.0, 1.0])
        print()

    def test_less_treatment__generate_df(self):
        data = DataGenerator(is_treatment=False)
        self.assertEqual(len(data.df.columns), 9)
        self.assertEqual(len(data.features_names), data.num_features + 2)
        self.assertEqual(len(data.target_names), data.num_targets)
        self.assertEqual(len(data.treatment_name), 0)
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.target.dtype, 'float64')
        print()

    def test_more_outcome__generate_df(self):
        data = DataGenerator(num_targets=2)
        self.assertEqual(len(data.df.columns), 11)
        self.assertEqual(len(data.features_names), data.num_features + 2)
        self.assertEqual(len(data.target_names), data.num_targets)
        self.assertEqual(data.treatment_name, 'treatment')
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.target_1.dtype, 'float64')
        print()

    def test_more_outcome_less_treatment__generate_df(self):
        data = DataGenerator(num_targets=4, is_treatment=False)
        self.assertEqual(len(data.df.columns), 12)
        self.assertEqual(len(data.features_names), data.num_features + 2)
        self.assertEqual(len(data.target_names), data.num_targets)
        self.assertEqual(len(data.treatment_name), 0)
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.target_1.dtype, 'float64')
        print()

    def test_binary_outcome__generate_df(self):
        data = DataGenerator(is_binary_target=True)
        self.assertEqual(len(data.df.columns), 10)
        self.assertEqual(len(data.features_names), data.num_features + 2)
        self.assertEqual(len(data.target_names), data.num_targets)
        self.assertEqual(data.treatment_name, 'treatment')
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(list(set(data.df.target)), [0.0, 1.0])
        print()

    def test_more_binary_outcome__generate_df(self):
        data = DataGenerator(num_targets=3, is_binary_target=True)
        self.assertEqual(len(data.df.columns), 12)
        self.assertEqual(len(data.features_names), data.num_features + 2)
        self.assertEqual(len(data.target_names), data.num_targets)
        self.assertEqual(data.treatment_name, 'treatment')
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(list(set(data.df.target_1)), [0.0, 1.0])
        self.assertEqual(list(set(data.df.target_2)), [0.0, 1.0])
        self.assertEqual(list(set(data.df.target_3)), [0.0, 1.0])
        print()

    def test_list_na_columns__generate_df(self):
        data = DataGenerator(na_columns=['feature_1', 'feature_2'])
        self.assertEqual(len(data.df.columns), 10)
        self.assertEqual(len(data.features_names), data.num_features + 2)
        self.assertEqual(len(data.target_names), data.num_targets)
        self.assertEqual(data.treatment_name, 'treatment')
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.feature_1.isna().sum(), 500)
        self.assertEqual(data.df.feature_2.isna().sum(), 500)
        self.assertEqual(data.df.feature_3.isna().sum(), 0)
        print()

    def test_one_na_column__generate_df(self):
        data = DataGenerator(na_columns='feature_1')
        self.assertEqual(len(data.df.columns), 10)
        self.assertEqual(len(data.features_names), data.num_features + 2)
        self.assertEqual(len(data.target_names), data.num_targets)
        self.assertEqual(data.treatment_name, 'treatment')
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.feature_1.isna().sum(), 500)
        self.assertEqual(data.df.feature_2.isna().sum(), 0)
        print()

    def test_na_step__generate_df(self):
        data = DataGenerator(na_columns='feature_1', na_step=5)
        self.assertEqual(len(data.df.columns), 10)
        self.assertEqual(len(data.features_names), data.num_features + 2)
        self.assertEqual(len(data.target_names), data.num_targets)
        self.assertEqual(data.treatment_name, 'treatment')
        self.assertEqual(len(data.info_col_names), data.num_info_cols)
        self.assertEqual(len(data.df), data.num_records)
        self.assertEqual(data.df.feature_1.isna().sum(), 1000)
        self.assertEqual(data.df.feature_2.isna().sum(), 0)
        print()

    def test_not_df_column__generate_df(self):
        with self.assertRaises(KeyError):
            data = DataGenerator(na_columns='feature_10')
        print()


if __name__ == '__main__':
    unittest.main()