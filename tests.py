import unittest
import pandas as pd
import pandas as pd
from hypex.dataset import Dataset, DatasetBase, InfoRole, DefaultRole
import json

class TestDataset(unittest.TestCase):

    def setUp(self):
        roles = {
            'col1': InfoRole(),
            'col2': InfoRole()
        }
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        self.dataset = Dataset(roles=roles, data=data)

    def test_initialization(self):
        self.assertEqual(len(self.dataset), 3)
        self.assertListEqual(list(self.dataset.columns), ['col1', 'col2'])
        self.assertIn('col1', self.dataset.roles)

    def test_add_column(self):
        new_data = [7, 8, 9]
        self.dataset.add_column(new_data, {'col3': InfoRole()})
        self.assertIn('col3', self.dataset.columns)
        self.assertListEqual(self.dataset.data['col3'].tolist(), new_data)

    def test_astype(self):
        self.dataset = self.dataset.astype({'col1': float})
        self.assertEqual(self.dataset.data['col1'].dtype, float)

    def test_append(self):
        roles = {'col1': InfoRole(), 'col2': InfoRole()}
        data = pd.DataFrame({
            'col1': [7, 8],
            'col2': [9, 10]
        })
        dataset_to_append = Dataset(roles=roles, data=data)
        new_dataset = self.dataset.append(dataset_to_append)
        self.assertEqual(len(new_dataset), 5)

    def test_merge(self):
        roles = {'col1': InfoRole(), 'col3': InfoRole()}
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col3': [7, 8, 9]
        })
        other_dataset = Dataset(roles=roles, data=data)
        merged = self.dataset.merge(other_dataset, on='col1')
        self.assertIn('col3', merged.columns)

    def test_fillna(self):
        self.dataset.data.loc[1, 'col1'] = None
        filled_dataset = self.dataset.fillna({'col1': 0})
        self.assertEqual(filled_dataset.data.loc[1, 'col1'], 0)

    def test_dropna(self):
        self.dataset.data.loc[1, 'col1'] = None
        dropped_dataset = self.dataset.dropna()
        self.assertEqual(len(dropped_dataset), 2)

    def test_operators(self):
        result = self.dataset + 1
        self.assertListEqual(result.data['col1'].tolist(), [2, 3, 4])

    def test_groupby(self):
        grouped = self.dataset.groupby(by='col1')
        self.assertEqual(len(grouped), 3)
        for key, group in grouped:
            self.assertEqual(len(group), 1)

class TestDatasetBase(unittest.TestCase):

    def setUp(self):
        roles = {
            'col1': InfoRole(),
            'col2': InfoRole()
        }
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        self.dataset_base = DatasetBase(roles=roles, data=data)

    def test_roles_property(self):
        self.assertIn('col1', self.dataset_base.roles)
        self.assertIsInstance(self.dataset_base.roles['col1'], InfoRole)

    def test_replace_roles(self):
        new_role = DefaultRole()
        self.dataset_base.replace_roles({'col1': new_role})
        self.assertIsInstance(self.dataset_base.roles['col1'], DefaultRole)

    def test_search_columns(self):
        columns = self.dataset_base.search_columns(InfoRole())
        self.assertListEqual(columns, ['col1', 'col2'])

    def test_to_dict(self):
        result = self.dataset_base.to_dict()
        self.assertIn('data', result)
        self.assertIn('roles', result)

    def test_to_json(self):
        result = self.dataset_base.to_json()
        self.assertIsInstance(result, str)
        parsed = json.loads(result)
        self.assertIn('data', parsed)
        self.assertIn('roles', parsed)

if __name__ == '__main__':
    unittest.main()
