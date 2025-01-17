import unittest
import copy
import pandas as pd
from hypex.dataset import *
from hypex.dataset.roles import *
from hypex.utils import BackendsEnum, RoleColumnError, ExperimentDataEnum

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.roles = {
            'col1': InfoRole(),
            'col2': InfoRole()
        }
        self.data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        self.dataset = Dataset(roles=self.roles, data=self.data)
        

    def test_initialization(self):
        self.assertEqual(len(self.dataset), 3)
        self.assertListEqual(list(self.dataset.columns), ['col1', 'col2'])
        self.assertIn('col1', self.dataset.roles)

        roles_with_mapping = {
            InfoRole(): ['col1', 'col2']
        }
        dataset_with_mapping = Dataset(roles=roles_with_mapping, data=self.data)
        self.assertListEqual(list(dataset_with_mapping.columns), ['col1', 'col2'])

        dataset_with_backend = Dataset(roles=roles_with_mapping, data=self.data, backend=BackendsEnum.pandas)
        self.assertListEqual(list(dataset_with_backend.columns), ['col1', 'col2'])

        dataset_unknown_backend = Dataset(roles=roles_with_mapping, data=self.data, backend='unknow')
        self.assertListEqual(list(dataset_unknown_backend.columns), ['col1', 'col2'])

        roles_partial = {
            InfoRole(): ['col1']
        }
        dataset_partial = Dataset(roles=roles_partial, data=self.data)
        self.assertListEqual(list(dataset_partial.columns), ['col1', 'col2'])
        self.assertEqual(str(dataset_partial), str(dataset_partial.data))
        self.assertEqual(dataset_partial._repr_html_(), dataset_partial.data._repr_html_())

        roles_invalid = {
            InfoRole(): ['col1', 'col3']
        }
        with self.assertRaises(RoleColumnError):
            Dataset(roles=roles_invalid, data=self.data)

    def test_add_column(self):
        new_data = [7, 8, 9]
        self.dataset.add_column(new_data, {'col3': InfoRole()})

        self.assertIn('col3', self.dataset.columns)
        self.assertListEqual(self.dataset.data['col3'].tolist(), new_data)

    def test_astype(self):
        dataset = self.dataset.astype({'col1': float})
        self.assertEqual(dataset.data['col1'].dtype, float)

    def test_append(self):
        dataset_new = copy.deepcopy(self.dataset)
        self.dataset = self.dataset.append(dataset_new)
        self.assertEqual(len(self.dataset), 6)

    def test_merge(self):
        data_new = pd.DataFrame({
            'col1': [1, 2, 3],
            'col3': [7, 8, 9]
        })
        roles_new = {'col1': InfoRole(), 'col3': InfoRole()}
        dataset_new = Dataset(roles=roles_new, data=data_new)

        merged = self.dataset.merge(dataset_new, on='col1')
        self.assertIn('col3', merged.columns)

    def test_fillna(self):
        data_with_na = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': [4, 5, 6]
        })
        dataset_with_na = Dataset(roles=self.roles, data=data_with_na)

        filled_dataset = dataset_with_na.fillna({'col1': 0})
        self.assertEqual(filled_dataset.data.loc[1, 'col1'], 0)

    def test_dropna(self):
        data_with_na = pd.DataFrame({
            'col1': [1, None, 3],
            'col2': [4, 5, 6]
        })
        dataset_with_na = Dataset(roles=self.roles, data=data_with_na)

        dropped_dataset = dataset_with_na.dropna()
        self.assertEqual(len(dropped_dataset), 2)


    def test_operators(self):
        result = self.dataset + 1
        self.assertListEqual(result.data['col1'].tolist(), [2, 3, 4])

    def test_groupby(self):
        data_grouped = pd.DataFrame({
            'col1': [1, 2, 1],
            'col2': [4, 5, 6]
        })
        dataset_grouped = Dataset(roles=self.roles, data=data_grouped)

        grouped = dataset_grouped.groupby(by='col1')
        self.assertEqual(len(grouped), 2)
        for key, group in grouped:
            self.assertGreaterEqual(len(group), 1)

    def test_roles_property(self):
        dataset_base = DatasetBase(roles=self.roles, data=self.data)

        self.assertIn('col1', dataset_base.roles)
        self.assertIsInstance(dataset_base.roles['col1'], InfoRole)

    def test_replace_roles(self):
        dataset_base = DatasetBase(roles=self.roles, data=self.data)

        new_role = DefaultRole()
        dataset_base.replace_roles({'col1': new_role})
        self.assertIsInstance(dataset_base.roles['col1'], DefaultRole)
        dataset_base.replace_roles({'col1': InfoRole()}, tmp_role=True)
        dataset_base.replace_roles({'col1': InfoRole()}, auto_roles_types=True)

    def test_search_columns(self):
        dataset_base = DatasetBase(roles=self.roles, data=self.data)

        columns = dataset_base.search_columns(InfoRole())
        self.assertListEqual(columns, ['col1', 'col2'])

    def test_to_dict(self):
        dataset_base = DatasetBase(roles=self.roles, data=self.data)

        result = dataset_base.to_dict()
        self.assertIn('data', result)
        self.assertIn('roles', result)

    
    def test_getitem_by_column(self):

        # Получаем подмножество по столбцу 'col1'
        subset = self.dataset['col1']

        # Проверяем, что данные вернулись для 'col1'
        self.assertIn('col1', subset.columns)
        self.assertNotIn('col2', subset.columns)

        # Проверяем, что роли для 'col1' сохранены
        self.assertIn('col1', subset.roles)
        self.assertIsInstance(subset.roles['col1'], InfoRole)

    def test_getitem_with_multiple_columns(self):
        roles = {
            'col1': InfoRole(),
            'col2': InfoRole(),
            'col3': InfoRole()
        }
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })
        dataset = Dataset(roles=roles, data=data)

        # Получаем подмножество по столбцам 'col1' и 'col3'
        subset = dataset[['col1', 'col3']]

        # Проверяем, что данные вернулись для 'col1' и 'col3'
        self.assertIn('col1', subset.columns)
        self.assertIn('col3', subset.columns)
        self.assertNotIn('col2', subset.columns)

        # Проверяем, что роли для 'col1' и 'col3' сохранены
        self.assertIn('col1', subset.roles)
        self.assertIn('col3', subset.roles)
        self.assertIsInstance(subset.roles['col1'], InfoRole)
        self.assertIsInstance(subset.roles['col3'], InfoRole)

    def test_getitem_with_non_existing_column(self):

        # Пытаемся получить несуществующий столбец
        with self.assertRaises(KeyError):
            self.dataset['col3']

    def test_getitem_empty_result(self):

        # Пытаемся получить пустое подмножество (например, с применением условия)
        subset = self.dataset[self.dataset['col1'] > 3]
        self.assertTrue(len(subset)==0)
    
    def test_rename_single_column(self):

        # Переименовываем только 'col1' в 'new_col1'
        renamed_dataset = self.dataset.rename({'col1': 'new_col1'})

        # Проверяем, что столбец 'col1' переименован в 'new_col1'
        self.assertIn('new_col1', renamed_dataset.columns)
        self.assertNotIn('col1', renamed_dataset.columns)

        # Проверяем, что роль для 'col1' переименована в 'new_col1'
        self.assertIn('new_col1', renamed_dataset.roles)
        self.assertIsInstance(renamed_dataset.roles['new_col1'], InfoRole)

    def test_rename_multiple_columns(self):
        roles = {
            'col1': InfoRole(),
            'col2': InfoRole(),
            'col3': InfoRole()
        }
        data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6],
            'col3': [7, 8, 9]
        })
        dataset = Dataset(roles=roles, data=data)

        # Переименовываем несколько столбцов
        renamed_dataset = dataset.rename({'col1': 'new_col1', 'col3': 'new_col3'})

        # Проверяем, что столбцы 'col1' и 'col3' переименованы
        self.assertIn('new_col1', renamed_dataset.columns)
        self.assertIn('new_col3', renamed_dataset.columns)
        self.assertNotIn('col1', renamed_dataset.columns)
        self.assertNotIn('col3', renamed_dataset.columns)

        # Проверяем, что роли для 'col1' и 'col3' переименованы
        self.assertIn('new_col1', renamed_dataset.roles)
        self.assertIn('new_col3', renamed_dataset.roles)
        self.assertIsInstance(renamed_dataset.roles['new_col1'], InfoRole)
        self.assertIsInstance(renamed_dataset.roles['new_col3'], InfoRole)

    def test_rename_no_change(self):

        # Переименовываем столбцы, но без изменений (передаем пустой словарь)
        renamed_dataset = self.dataset.rename({})

        # Проверяем, что данные остались без изменений
        self.assertEqual(list(self.dataset.columns), list(renamed_dataset.columns))
        self.assertEqual(list(self.dataset.roles.keys()), list(renamed_dataset.roles.keys()))

    def test_rename_with_non_existent_column(self):
        # Переименовываем несуществующий столбец
        renamed_dataset = self.dataset.rename({'non_existent': 'new_col'})

        # Проверяем, что столбцы без изменений
        self.assertIn('col1', renamed_dataset.columns)
        self.assertIn('col2', renamed_dataset.columns)
        self.assertNotIn('new_col', renamed_dataset.columns)

    def test_rename_roles(self):
        # Переименовываем столбцы и проверяем, что роли тоже обновляются
        renamed_dataset = self.dataset.rename({'col1': 'new_col1'})

        # Проверяем, что роль для 'col1' переименована в 'new_col1'
        self.assertIn('new_col1', renamed_dataset.roles)
        self.assertIsInstance(renamed_dataset.roles['new_col1'], InfoRole)
        self.assertNotIn('col1', renamed_dataset.roles)

    def test_set_value_additional_fields_single_column(self):
        # Настроим необходимые данные
        experiment_data = ExperimentData(self.dataset)

        # Применяем set_value с одним столбцом в additional_fields
        experiment_data.set_value(
            space=ExperimentDataEnum.additional_fields,
            executor_id='executor_1',
            value=self.dataset
        )

        # Проверяем, что столбец был добавлен в additional_fields
        self.assertIn('executor_1', experiment_data.additional_fields.columns)

    def test_set_value_additional_fields_multiple_columns(self):
        experiment_data = ExperimentData(self.dataset)

        # Применяем set_value с несколькими столбцами в additional_fields
        experiment_data.set_value(
            space=ExperimentDataEnum.additional_fields,
            executor_id={'col1': 'executor_1', 'col2': 'executor_2'},
            value=self.dataset
        )

        # Проверяем, что столбцы были переименованы и добавлены
        self.assertIn('executor_1', experiment_data.additional_fields.columns)
        self.assertIn('executor_2', experiment_data.additional_fields.columns)

    def test_set_value_analysis_tables(self):
        # Настроим необходимые данные
        experiment_data = ExperimentData(self.dataset)

        # Применяем set_value для analysis_tables
        experiment_data.set_value(
            space=ExperimentDataEnum.analysis_tables,
            executor_id='executor_1',
            value='analysis_data'
        )

        # Проверяем, что данные были добавлены в analysis_tables
        self.assertIn('executor_1', experiment_data.analysis_tables)
        self.assertEqual(experiment_data.analysis_tables['executor_1'], 'analysis_data')

    def test_set_value_variables_dict(self):
        # Настроим необходимые данные
        experiment_data = ExperimentData(self.dataset)

        # Применяем set_value с значением типа Dict
        experiment_data.set_value(
            space=ExperimentDataEnum.variables,
            executor_id='executor_3',
            value={'key1': 'value1', 'key2': 'value2'}
        )

        # Проверяем, что данные были добавлены в variables
        self.assertIn('executor_3', experiment_data.variables)
        self.assertEqual(experiment_data.variables['executor_3'], {'key1': 'value1', 'key2': 'value2'})

    def test_set_value_variables_existing_executor(self):
        # Настроим необходимые данные
        experiment_data = ExperimentData(self.dataset)
        experiment_data.variables = {'executor_1': {'key1': 'value1'}}

        # Применяем set_value для существующего executor_id
        experiment_data.set_value(
            space=ExperimentDataEnum.variables,
            executor_id='executor_1',
            value='new_value',
            key='key2'
        )

        # Проверяем, что значение в variables обновилось
        self.assertEqual(experiment_data.variables['executor_1']['key2'], 'new_value')

    def test_set_value_variables_new_executor(self):
        # Настроим необходимые данные
        experiment_data = ExperimentData(self.dataset)

        # Применяем set_value для нового executor_id
        experiment_data.set_value(
            space=ExperimentDataEnum.variables,
            executor_id='executor_2',
            value='new_value',
            key='key1'
        )

        # Проверяем, что новый executor_id был добавлен в variables
        self.assertIn('executor_2', experiment_data.variables)
        self.assertEqual(experiment_data.variables['executor_2']['key1'], 'new_value')

    def test_set_value_groups(self):
        # Настроим необходимые данные
        experiment_data = ExperimentData(self.dataset)

        # Применяем set_value для groups
        experiment_data.set_value(
            space=ExperimentDataEnum.groups,
            executor_id='executor_1',
            value='group_data',
            key='key1'
        )

        # Проверяем, что данные были добавлены в groups
        self.assertIn('executor_1', experiment_data.groups)
        self.assertEqual(experiment_data.groups['executor_1']['key1'], 'group_data')

    def test_set_value_groups_existing_executor(self):
        # Настроим необходимые данные
        experiment_data = ExperimentData(self.dataset)
        experiment_data.groups = {'executor_1': {'key1': 'old_value'}}

        # Применяем set_value для существующего executor_id
        experiment_data.set_value(
            space=ExperimentDataEnum.groups,
            executor_id='executor_1',
            value='new_group_data',
            key='key2'
        )

        # Проверяем, что данные в groups обновились
        self.assertEqual(experiment_data.groups['executor_1']['key2'], 'new_group_data')
    
    def test_initialization(self):
        # Создаем пустой Dataset
        dataset = Dataset.create_empty()
        experiment_data = ExperimentData(dataset)

        # Проверяем, что атрибуты инициализированы правильно
        self.assertIsInstance(experiment_data.additional_fields, Dataset)
        self.assertEqual(len(experiment_data.variables), 0)
        self.assertEqual(len(experiment_data.groups), 0)
        self.assertEqual(len(experiment_data.analysis_tables), 0)

    def test_create_empty(self):
        # Создаем пустой Dataset с использованием create_empty
        experiment_data = ExperimentData.create_empty()

        # Проверяем, что объект ExperimentData был создан
        self.assertIsInstance(experiment_data, ExperimentData)
        self.assertIsInstance(experiment_data.ds, Dataset)
        self.assertIsInstance(experiment_data.additional_fields, Dataset)

    def test_check_hash_additional_fields(self):
        # Настроим необходимые данные
        dataset = Dataset.create_empty()
        experiment_data = ExperimentData(dataset)

        # Проверим, что check_hash возвращает True, если executor_id существует в additional_fields
        executor_id = 'executor_1'
        experiment_data.additional_fields = experiment_data.additional_fields.add_column(
            Dataset(roles={executor_id: InfoRole()}, data=pd.DataFrame({executor_id: [1, 2, 3]}))
        )

        self.assertTrue(experiment_data.check_hash(executor_id, ExperimentDataEnum.additional_fields))

        # Проверим, что check_hash возвращает False для несуществующего executor_id
        self.assertFalse(experiment_data.check_hash('nonexistent_executor', ExperimentDataEnum.additional_fields))

    def test_check_hash_variables(self):
        # Настроим необходимые данные
        dataset = Dataset.create_empty()
        experiment_data = ExperimentData(dataset)

        # Добавляем executor_id в переменные
        experiment_data.variables['executor_1'] = {'key1': 1}

        # Проверим, что check_hash возвращает True для существующего executor_id
        self.assertTrue(experiment_data.check_hash('executor_1', ExperimentDataEnum.variables))

        # Проверим, что check_hash возвращает False для несуществующего executor_id
        self.assertFalse(experiment_data.check_hash('nonexistent_executor', ExperimentDataEnum.variables))

    def test_check_hash_analysis_tables(self):
        # Настроим необходимые данные
        dataset = Dataset.create_empty()
        experiment_data = ExperimentData(dataset)

        # Добавляем executor_id в analysis_tables
        experiment_data.analysis_tables['executor_1'] = dataset

        # Проверим, что check_hash возвращает True для существующего executor_id
        self.assertTrue(experiment_data.check_hash('executor_1', ExperimentDataEnum.analysis_tables))
        self.assertTrue(experiment_data.check_hash('executor_1', 'unknown'))

        # Проверим, что check_hash возвращает False для несуществующего executor_id
        self.assertFalse(experiment_data.check_hash('nonexistent_executor', ExperimentDataEnum.analysis_tables))

if __name__ == '__main__':
    unittest.main()
