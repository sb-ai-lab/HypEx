import unittest
import pandas as pd
from hypex.dataset import *
import json

class TestDataset(unittest.TestCase):

    def test_initialization(self):
        roles = {
            'col1': InfoRole(),
            'col2': InfoRole()
        }
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        dataset = Dataset(roles=roles, data=data)

        self.assertEqual(len(dataset), 3)
        self.assertListEqual(list(dataset.columns), ['col1', 'col2'])
        self.assertIn('col1', dataset.roles)

    def test_add_column(self):
        roles = {
            'col1': InfoRole(),
            'col2': InfoRole()
        }
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        dataset = Dataset(roles=roles, data=data)

        new_data = [7, 8, 9]
        dataset.add_column(new_data, {'col3': InfoRole()})

        self.assertIn('col3', dataset.columns)
        self.assertListEqual(dataset.data['col3'].tolist(), new_data)

    def test_astype(self):
        roles = {
            'col1': InfoRole(),
            'col2': InfoRole()
        }
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        dataset = Dataset(roles=roles, data=data)

        dataset = dataset.astype({'col1': float})
        self.assertEqual(dataset.data['col1'].dtype, float)

    def test_append(self):
        roles1 = {'col1': InfoRole(), 'col2': InfoRole()}
        data1 = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        dataset1 = Dataset(roles=roles1, data=data1)

        roles2 = {'col1': InfoRole(), 'col2': InfoRole()}
        data2 = pd.DataFrame({
            'col1': [7, 8],
            'col2': [9, 10]
        })
        dataset2 = Dataset(roles=roles2, data=data2)

        new_dataset = dataset1.append(dataset2)
        self.assertEqual(len(new_dataset), 5)

    def test_merge(self):
        roles1 = {'col1': InfoRole(), 'col2': InfoRole()}
        data1 = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        dataset1 = Dataset(roles=roles1, data=data1)

        roles2 = {'col1': InfoRole(), 'col3': InfoRole()}
        data2 = pd.DataFrame({
            'col1': [1, 2, 3],
            'col3': [7, 8, 9]
        })
        dataset2 = Dataset(roles=roles2, data=data2)

        merged = dataset1.merge(dataset2, on='col1')
        self.assertIn('col3', merged.columns)

    def test_fillna(self):
        roles = {'col1': InfoRole(), 'col2': InfoRole()}
        data = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': [4, 5, 6]
        })
        dataset = Dataset(roles=roles, data=data)

        filled_dataset = dataset.fillna({'col1': 0})
        self.assertEqual(filled_dataset.data.loc[1, 'col1'], 0)

    def test_dropna(self):
        roles = {'col1': InfoRole(), 'col2': InfoRole()}
        data = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': [4, 5, 6]
        })
        dataset = Dataset(roles=roles, data=data)

        dropped_dataset = dataset.dropna()
        self.assertEqual(len(dropped_dataset), 2)

    def test_operators(self):
        roles = {'col1': InfoRole(), 'col2': InfoRole()}
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        dataset = Dataset(roles=roles, data=data)

        result = dataset + 1
        self.assertListEqual(result.data['col1'].tolist(), [2, 3, 4])

    def test_groupby(self):
        roles = {'col1': InfoRole(), 'col2': InfoRole()}
        data = pd.DataFrame({
            'col1': [1, 2, 1],
            'col2': [4, 5, 6]
        })
        dataset = Dataset(roles=roles, data=data)

        grouped = dataset.groupby(by='col1')
        self.assertEqual(len(grouped), 2)
        for key, group in grouped:
            self.assertGreaterEqual(len(group), 1)

class TestDatasetBase(unittest.TestCase):

    def test_roles_property(self):
        roles = {
            'col1': InfoRole(),
            'col2': InfoRole()
        }
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        dataset_base = DatasetBase(roles=roles, data=data)

        self.assertIn('col1', dataset_base.roles)
        self.assertIsInstance(dataset_base.roles['col1'], InfoRole)

    def test_replace_roles(self):
        roles = {
            'col1': InfoRole(),
            'col2': InfoRole()
        }
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        dataset_base = DatasetBase(roles=roles, data=data)

        new_role = DefaultRole()
        dataset_base.replace_roles({'col1': new_role})
        self.assertIsInstance(dataset_base.roles['col1'], DefaultRole)

    def test_search_columns(self):
        roles = {
            'col1': InfoRole(),
            'col2': InfoRole()
        }
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        dataset_base = DatasetBase(roles=roles, data=data)

        columns = dataset_base.search_columns(InfoRole())
        self.assertListEqual(columns, ['col1', 'col2'])

    def test_to_dict(self):
        roles = {
            'col1': InfoRole(),
            'col2': InfoRole()
        }
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        dataset_base = DatasetBase(roles=roles, data=data)

        result = dataset_base.to_dict()
        self.assertIn('data', result)
        self.assertIn('roles', result)

    def test_to_json(self):
        roles = {
            'col1': InfoRole(),
            'col2': InfoRole()
        }
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        dataset_base = DatasetBase(roles=roles, data=data)

        result = dataset_base.to_json()
        self.assertIsInstance(result, str)
        parsed = json.loads(result)
        self.assertIn('data', parsed)
        self.assertIn('roles', parsed)

if __name__ == '__main__':
    unittest.main()
