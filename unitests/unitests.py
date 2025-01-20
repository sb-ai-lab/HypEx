import unittest
import copy
import pandas as pd
import numpy as np
from hypex.dataset import *
from hypex.dataset.roles import *
from hypex.utils import *
import json

class TestDataset(unittest.TestCase):

    def setUp(self):
        self.roles = {
            'col1': InfoRole(int),
            'col2': InfoRole(int)
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

    def test_getitem_by_column_name(self):
        subset = self.dataset['col1']
        self.assertTrue(isinstance(subset, Dataset))
        self.assertIn('col1', subset.columns)

    def test_getitem_by_multiple_column_names(self):
        subset = self.dataset[['col1', 'col2']]
        self.assertTrue(isinstance(subset, Dataset))
        self.assertEqual(set(subset.columns), {'col1', 'col2'})

    def test_setitem_existing_column(self):
        self.dataset['col1'] = [7, 8, 9]
        self.assertEqual(self.dataset.data['col1'].tolist(), [7, 8, 9])

    def test_invalid_type_other(self):
        with self.assertRaises(DataTypeError):
            result = self.dataset + {}

    def test_backend_type_error(self):
        # Создаем второй Dataset с другим типом бэкенда
        other_roles = {'col1': InfoRole(int), 'col2': InfoRole(int)}
        other_data = pd.DataFrame({
            'col1': [10, 20, 30],
            'col2': [40, 50, 60]
        })
        other_dataset = Dataset(roles=other_roles, data=other_data, backend='unknown')

        # Проверка ошибки при несовместимых бэкендах
        with self.assertRaises(BackendTypeError):
            # Пытаемся выполнить операцию с несовместимым бэкендом
            result = self.dataset + other_dataset  # Ожидаем ошибку BackendTypeError

    def test_setitem_new_column(self):
        new_data = [10, 11, 12]
        self.dataset['col3'] = new_data
        self.assertIn('col3', self.dataset.columns)
        self.assertEqual(self.dataset.data['col3'].tolist(), new_data)

    def test_setitem_dataset(self):
        new_data = copy.deepcopy(self.dataset['col1'])
        self.dataset['col3'] = new_data
        self.assertIn('col3', self.dataset.columns)
        self.assertEqual(self.dataset.data['col3'].tolist(), self.data['col1'].tolist())

    def test_setitem_invalid_type(self):
        with self.assertRaises(TypeError):
            self.dataset['col1'] = ['string', 'another', 'string']  # Incorrect type

    def test_setitem_with_iloc(self):
        new_data = [15, 16, 17]
        self.dataset.iloc[1, 0] = new_data[0]
        self.assertEqual(self.dataset.data.iloc[1, 0], new_data[0])

    def test_getitem_column_not_found(self):
        with self.assertRaises(KeyError):
            self.dataset['nonexistent_column']
        
    def test_setitem_illegal_index(self):
        with self.assertRaises(IndexError):
            self.dataset.iloc[1, 3] = 10  # Index out of bounds

    def test_getitem_empty_dataset(self):
        empty_data = pd.DataFrame(columns=['col1', 'col2'])
        empty_dataset = Dataset(roles=self.roles, data=empty_data)
        subset = empty_dataset['col1']
        self.assertTrue(isinstance(subset, Dataset))
        self.assertEqual(len(subset), 0)

    def test_add_column(self):

        with self.assertRaises(ValueError):
            self.dataset.add_column([1, 2, 3])

        # Test basic add_column
        new_data = [7, 8, 9]
        self.dataset.add_column(new_data, {'col3': InfoRole()})
        self.assertIn('col3', self.dataset.columns)
        self.assertListEqual(self.dataset.data['col3'].tolist(), new_data)

        self.dataset.add_column(Dataset(pd.DataFrame({'col4': new_data})), {'col4': InfoRole()})
        self.assertIn('col4', self.dataset.columns)
        self.assertListEqual(self.dataset.data['col3'].tolist(), new_data)

        # Test with numpy array
        import numpy as np
        self.dataset.add_column(np.array([13, 14, 15]), {'col6': InfoRole()})
        self.assertListEqual(self.dataset.data['col6'].tolist(), [13, 14, 15])

        # Test with mismatched length
        with self.assertRaises(ValueError):
            self.dataset.add_column([1, 2], {'col8': InfoRole()})

        # Test with invalid role type
        with self.assertRaises(TypeError):
            self.dataset.add_column([1, 2, 3], {'col9': 'not_a_role'})

        # Test with duplicate column name
        with self.assertRaises(ValueError):
            self.dataset.add_column([1, 2, 3], {'col1': InfoRole()})

    def test_agg(self):
        # Test single function
        result = self.dataset.agg('mean')
        self.assertIsInstance(result, Dataset)

        # Test multiple functions
        result = self.dataset.agg(['mean', 'sum'])
        self.assertIn('mean', result.index)
        self.assertIn('sum', result.index)

        # Test dict of functions
        result = self.dataset.agg({'col1': 'mean', 'col2': 'sum'})
        self.assertEqual(result.loc['mean', 'col1'], self.dataset.data['col1'].mean())
        self.assertEqual(result.loc['sum', 'col2'], self.dataset.data['col2'].sum())

        # Edge cases
        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        with self.assertRaises(ValueError):
            empty_dataset.agg('mean')

        # Test with invalid function
        with self.assertRaises(ValueError):
            self.dataset.agg('invalid_function')

        # Test with NaN values
        self.dataset.data.loc[0, 'col1'] = None
        result = self.dataset.agg('mean')
        self.assertTrue(pd.notna(result['col1']))

        # Test with all NaN column
        self.dataset.data['col1'] = None
        result = self.dataset.agg('mean')
        self.assertTrue(pd.isna(result['col1']))

    def test_append(self):
        # Test basic append
        other_data = pd.DataFrame({
            'col1': [7, 8],
            'col2': [9, 10]
        })
        other_dataset = Dataset(roles=self.roles, data=other_data)
        result = self.dataset.append(other_dataset)
        self.assertEqual(len(result), 5)

        # # Test append with ignore_index
        # result = self.dataset.append(other_dataset, ignore_index=True)
        # self.assertEqual(list(result.index), [0, 1, 2, 3, 4])

        # Test append with different columns
        other_data = pd.DataFrame({
            'col1': [7],
            'col3': [11]
        })
        other_roles = {'col1': InfoRole(), 'col3': InfoRole()}
        other_dataset = Dataset(roles=other_roles, data=other_data)
        result = self.dataset.append(other_dataset)
        self.assertIn('col3', result.columns)
        self.assertTrue(result.data['col2'].isna().any())

        # Edge cases
        # Test append empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = self.dataset.append(empty_dataset)
        self.assertEqual(len(result), len(self.dataset))

        # Test append to empty dataset
        result = empty_dataset.append(self.dataset)
        self.assertEqual(len(result), len(self.dataset))

        # Test append with completely different columns
        other_data = pd.DataFrame({'col3': [1], 'col4': [2]})
        other_roles = {'col3': InfoRole(), 'col4': InfoRole()}
        other_dataset = Dataset(roles=other_roles, data=other_data)
        result = self.dataset.append(other_dataset)
        self.assertTrue(result.data['col3'].isna().sum() == len(self.dataset))
        self.assertTrue(result.data['col1'].isna().sum() == 1)

        # Test append with invalid type
        with self.assertRaises(ConcatDataError):
            self.dataset.append([1, 2, 3])

    def test_apply(self):
        # Test with lambda function
        result = self.dataset.apply(lambda x: x * 2, self.roles)
        self.assertEqual(result.data['col1'].tolist(), [2, 4, 6])

        # Test with named function
        def multiply_by_three(x):
            return x * 3
        result = self.dataset.apply(multiply_by_three, self.roles)
        self.assertEqual(result.data['col1'].tolist(), [3, 6, 9])

        # Test with axis=1
        result = self.dataset.apply(lambda x: x['col1'] + x['col2'], self.roles, axis=1)
        self.assertEqual(result.data['col1'].tolist(), [5, 7, 9])

        # Edge cases
        # Test with function that returns None
        result = self.dataset.apply(lambda x: None, self.roles)
        self.assertTrue(result.data.isna().all().all())

        # Test with function that raises exception
        def failing_function(x):
            raise ValueError("Test error")
        with self.assertRaises(ConcatDataError):
            self.dataset.apply(failing_function, InfoRole())

        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.apply(lambda x: x * 2, InfoRole())
        self.assertTrue(result.is_empty())

        # Test with non-numeric data
        self.dataset.data['col1'] = ['a', 'b', 'c']
        with self.assertRaises(TypeError):
            self.dataset.apply(lambda x: x * 2, InfoRole())

    def test_astype(self):
        # Test single column conversion
        result = self.dataset.astype({'col1': str})
        self.assertTrue(result.data['col1'].dtype == 'object')

        # Test multiple column conversion
        result = self.dataset.astype({'col1': float, 'col2': str})
        self.assertTrue(result.data['col1'].dtype == 'float64')
        self.assertTrue(result.data['col2'].dtype == 'object')

        # Test with errors='ignore'
        invalid_dataset = Dataset(roles = {'col1': InfoRole()}, data = pd.DataFrame({'col1': ['a', 'b', 'c']}))
        result = invalid_dataset.astype({'col1': int}, errors='ignore')
        self.assertTrue(result.data['col1'].dtype == 'object')

        # Edge cases
        # Test with non-existent column
        with self.assertRaises(KeyError):
            self.dataset.astype({'non_existent': int})

        # Test with invalid dtype
        with self.assertRaises(TypeError):
            self.dataset.astype({'col1': 'invalid_dtype'})

        # Test with mixed types
        self.dataset.data['col1'] = [1, 'two', 3]
        with self.assertRaises(ValueError):
            self.dataset.astype({'col1': int})

        # Test with NaN values
        self.dataset.data['col1'] = [1, None, 3]
        result = self.dataset.astype({'col1': float})
        self.assertTrue(pd.isna(result.data['col1'][1]))

    def test_coefficient_of_variation(self):
        # Test basic functionality
        cv = self.dataset.coefficient_of_variation()
        expected_cv1 = self.dataset.std()['col1'] / self.dataset.mean()['col1']
        self.assertAlmostEqual(cv['col1'], expected_cv1)

        # Test with ddof parameter
        cv = self.dataset.coefficient_of_variation(ddof=0)
        std = self.dataset.std(ddof=0)
        mean = self.dataset.mean()
        expected_cv1 = std['col1'] / mean['col1']
        self.assertAlmostEqual(cv['col1'], expected_cv1)

        # Edge cases
        # Test with zero mean
        self.dataset.data['col1'] = [0, 0, 0]
        with self.assertRaises(ZeroDivisionError):
            self.dataset.coefficient_of_variation()

        # Test with negative values
        self.dataset.data['col1'] = [-1, -2, -3]
        cv = self.dataset.coefficient_of_variation()
        self.assertTrue(cv['col1'] < 0)

        # Test with NaN values
        self.dataset.data['col1'] = [1, None, 3]
        cv = self.dataset.coefficient_of_variation()
        self.assertTrue(pd.notna(cv['col1']))

        # Test with all NaN values
        self.dataset.data['col1'] = [None, None, None]
        cv = self.dataset.coefficient_of_variation()
        self.assertTrue(pd.isna(cv['col1']))

    def test_corr(self):
        # Test Pearson correlation
        corr = self.dataset.corr(method='pearson')
        self.assertIsInstance(corr, Dataset)
        self.assertEqual(corr.shape, (2, 2))

        # Test Spearman correlation
        corr = self.dataset.corr(method='spearman')
        self.assertIsInstance(corr, pd.DataFrame)
        self.assertEqual(corr.shape, (2, 2))

        # Test Kendall correlation
        corr = self.dataset.corr(method='kendall')
        self.assertIsInstance(corr, pd.DataFrame)
        self.assertEqual(corr.shape, (2, 2))

        # Edge cases
        # Test with constant column
        self.dataset.data['col1'] = [1, 1, 1]
        corr = self.dataset.corr()
        self.assertTrue(pd.isna(corr.loc['col1', 'col2']))

        # Test with NaN values
        self.dataset.data['col1'] = [1, None, 3]
        corr = self.dataset.corr()
        self.assertTrue(pd.notna(corr.loc['col1', 'col2']))

        # Test with all NaN values
        self.dataset.data['col1'] = [None, None, None]
        corr = self.dataset.corr()
        self.assertTrue(pd.isna(corr.loc['col1', 'col2']))

        # Test with invalid method
        with self.assertRaises(ValueError):
            self.dataset.corr(method='invalid_method')

    def test_count(self):
        # Test basic count
        counts = self.dataset.count()
        self.assertEqual(counts['col1'], 3)
        self.assertEqual(counts['col2'], 3)

        # Test count with NaN values
        self.dataset.data.loc[0, 'col1'] = None
        counts = self.dataset.count()
        self.assertEqual(counts['col1'], 2)
        self.assertEqual(counts['col2'], 3)

        # Edge cases
        # Test with all NaN values
        self.dataset.data['col1'] = None
        counts = self.dataset.count()
        self.assertEqual(counts['col1'], 0)

        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        counts = empty_dataset.count()
        self.assertEqual(counts['col1'], 0)

        # Test with mixed types
        self.dataset.data['col1'] = [1, 'two', None]
        counts = self.dataset.count()
        self.assertEqual(counts['col1'], 2)

    def test_cov(self):
        # Test basic covariance
        cov = self.dataset.cov()
        self.assertIsInstance(cov, Dataset)
        self.assertEqual(cov.shape, (2, 2))

        # Test with ddof parameter
        cov = self.dataset.cov(ddof=0)
        self.assertIsInstance(cov, pd.DataFrame)
        self.assertEqual(cov.shape, (2, 2))

        # Test with min_periods parameter
        cov = self.dataset.cov(min_periods=2)
        self.assertIsInstance(cov, pd.DataFrame)
        self.assertEqual(cov.shape, (2, 2))

        # Edge cases
        # Test with constant column
        self.dataset.data['col1'] = [1, 1, 1]
        cov = self.dataset.cov()
        self.assertEqual(cov.loc['col1', 'col1'], 0)

        # Test with NaN values
        self.dataset.data['col1'] = [1, None, 3]
        cov = self.dataset.cov()
        self.assertTrue(pd.notna(cov.loc['col1', 'col2']))

        # Test with all NaN values
        self.dataset.data['col1'] = [None, None, None]
        cov = self.dataset.cov()
        self.assertTrue(pd.isna(cov.loc['col1', 'col2']))

        # Test with insufficient observations
        cov = self.dataset.cov(min_periods=4)
        self.assertTrue(pd.isna(cov.loc['col1', 'col2']))

    def test_create_empty(self):
        # Test basic empty creation
        empty = Dataset.create_empty(roles=self.roles)
        self.assertTrue(empty.is_empty())
        self.assertEqual(list(empty.columns), ['col1', 'col2'])

        # Test with index
        empty = Dataset.create_empty(roles=self.roles, index=[0, 1, 2])
        self.assertTrue(empty.is_empty())
        self.assertEqual(len(empty.index), 3)

        # Edge cases
        # Test with empty roles
        with self.assertRaises(ValueError):
            Dataset.create_empty(roles={})

        # Test with invalid role type
        with self.assertRaises(TypeError):
            Dataset.create_empty(roles={'col1': 'not_a_role'})

        # Test with duplicate column names
        with self.assertRaises(ValueError):
            Dataset.create_empty(roles={'col1': InfoRole(), 'col1': InfoRole()})

        # Test with invalid index
        with self.assertRaises(TypeError):
            Dataset.create_empty(roles=self.roles, index='invalid')

    def test_dot(self):
        # Test with Dataset
        other_data = pd.DataFrame({'a': [1, 2, 3]})
        other_roles = {'a': InfoRole()}
        other = Dataset(data=other_data, roles=other_roles)
        result = self.dataset.dot(other)
        self.assertIsInstance(result, pd.Dataset)

        # Edge cases
        # Test with mismatched dimensions
        other_data = pd.Series([1, 2, 3, 4])
        other_roles = {'a': InfoRole()}
        other = Dataset(data=other_data, roles=other_roles)
        with self.assertRaises(ValueError):
            self.dataset.dot(other)

        # Test with NaN values
        self.dataset.data.loc[0, 'col1'] = None
        other_data = np.array([1, 2])
        other_roles = {'a': InfoRole()}
        other = Dataset(data=other_data, roles=other_roles)
        result = self.dataset.dot(other)
        self.assertTrue(pd.isna(result[0]))

        # Test with non-numeric data
        self.dataset.data['col1'] = ['a', 'b', 'c']
        other_data = np.array([1, 2])
        other_roles = {'a': InfoRole()}
        other = Dataset(data=other_data, roles=other_roles)
        with self.assertRaises(TypeError):
            self.dataset.dot(other)

        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        with self.assertRaises(ValueError):
            empty_dataset.dot(np.array([1, 2]))

    def test_drop(self):
        # Test drop single column
        result = self.dataset.drop(['col1'])
        self.assertNotIn('col1', result.columns)
        self.assertIn('col2', result.columns)

        # Test drop multiple columns
        result = self.dataset.drop(['col1', 'col2'])
        self.assertEqual(len(result.columns), 0)

        # Test drop with axis=0 (rows)
        result = self.dataset.drop([0, 1], axis=0)
        self.assertEqual(len(result), 1)

        # Edge cases
        # Test drop non-existent column
        with self.assertRaises(KeyError):
            self.dataset.drop(['non_existent'])

        # Test drop all columns
        result = self.dataset.drop(self.dataset.columns)
        self.assertEqual(len(result.columns), 0)

        # Test drop with empty list
        result = self.dataset.drop([])
        self.assertEqual(len(result.columns), len(self.dataset.columns))

        # Test drop with invalid axis
        with self.assertRaises(ValueError):
            self.dataset.drop(['col1'], axis=2)

    def test_dropna(self):
        # Test basic dropna
        self.dataset.data.loc[0, 'col1'] = None
        result = self.dataset.dropna()
        self.assertEqual(len(result), 2)

        # Test with how='all'
        self.dataset.data.loc[0] = [None, None]
        result = self.dataset.dropna(how='all')
        self.assertEqual(len(result), 2)

        # # Test with thresh parameter
        # result = self.dataset.dropna(thresh=1)
        # self.assertEqual(len(result), 3)

        # Edge cases
        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.dropna()
        self.assertEqual(len(result), 0)

        # Test with no NaN values
        self.dataset.data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        result = self.dataset.dropna()
        self.assertEqual(len(result), 3)

        # Test with invalid how parameter
        with self.assertRaises(ValueError):
            self.dataset.dropna(how='invalid')

        # Test with negative thresh
        with self.assertRaises(ValueError):
            self.dataset.dropna(thresh=-1)

    def test_fillna(self):
        # Test fill with single value
        self.dataset.data.loc[0, 'col1'] = None
        result = self.dataset.fillna(0)
        self.assertEqual(result.data.loc[0, 'col1'], 0)

        # Test fill with dict
        self.dataset.data.loc[0, ['col1', 'col2']] = None
        result = self.dataset.fillna({'col1': 1, 'col2': 2})
        self.assertEqual(result.data.loc[0, 'col1'], 1)
        self.assertEqual(result.data.loc[0, 'col2'], 2)

        # Test with method='ffill'
        result = self.dataset.fillna(method='ffill')
        self.assertFalse(result.data.isna().any().any())

        # Edge cases
        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.fillna(0)
        self.assertFalse(result.data.isna().any().any())

        # Test with no NaN values
        self.dataset.data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        result = self.dataset.fillna(0)
        self.assertTrue((result.data == self.dataset.data).all().all())

        # Test with invalid method
        with self.assertRaises(ValueError):
            self.dataset.fillna(method='invalid')

        # Test with incompatible value type
        with self.assertRaises(TypeError):
            self.dataset.fillna('string')  # When data is numeric

    def test_filter(self):
        # Test with items
        result = self.dataset.filter(items=['col1'])
        self.assertEqual(list(result.columns), ['col1'])

        # Test with regex
        result = self.dataset.filter(regex='1$')
        self.assertEqual(list(result.columns), ['col1'])

        # Test with like
        result = self.dataset.filter(like='col')
        self.assertEqual(len(result.columns), 2)

        # Edge cases
        # Test with non-existent items
        result = self.dataset.filter(items=['non_existent'])
        self.assertEqual(len(result.columns), 0)

        # Test with no matches in regex
        result = self.dataset.filter(regex='xyz')
        self.assertEqual(len(result.columns), 0)

        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.filter(like='col')
        self.assertEqual(len(result.columns), 0)

        # Test with multiple filter criteria
        with self.assertRaises(ValueError):
            self.dataset.filter(items=['col1'], regex='1$')

    def test_from_dict(self):
        # Test with dict of lists
        data_dict = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
        result = Dataset.from_dict(data_dict, self.roles)
        self.assertEqual(len(result), 3)

        # Test with dict of values
        data_dict = {'col1': 1, 'col2': 2}
        result = Dataset.from_dict(data_dict, self.roles)
        self.assertEqual(len(result), 1)

        # Test with orient='index'
        data_dict = {'row1': {'col1': 1, 'col2': 2}}
        result = Dataset.from_dict(data_dict, self.roles, orient='index')
        self.assertEqual(len(result), 1)

        # Edge cases
        # Test with empty dict
        with self.assertRaises(ValueError):
            Dataset.from_dict({}, self.roles)

        # Test with missing columns
        data_dict = {'col1': [1, 2, 3]}  # Missing col2
        with self.assertRaises(ValueError):
            Dataset.from_dict(data_dict, self.roles)

        # Test with extra columns
        data_dict = {'col1': [1], 'col2': [2], 'col3': [3]}
        with self.assertRaises(ValueError):
            Dataset.from_dict(data_dict, self.roles)

        # Test with invalid orient
        with self.assertRaises(ValueError):
            Dataset.from_dict({'col1': [1]}, self.roles, orient='invalid')

    def test_groupby(self):

        data_grouped = pd.DataFrame({
            'col1': [1, 2, 1],
            'col2': [4, 5, 6]
        })
        dataset_grouped = Dataset(roles=self.roles, data=data_grouped)

        grouped = dataset_grouped.groupby(by='col1')
        self.assertEqual(len(grouped), 2)

    def test_groupby_with_single_column_dataset(self):
        # Создаем новый Dataset для группировки
        by = Dataset(roles={'col1': InfoRole(int)}, data=pd.DataFrame({'col1': [1, 2, 1]}))
        
        result = self.dataset.groupby(by=by)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0][1], Dataset)
    
    def test_groupby_with_column_name(self):
        # Группировка по столбцу 'col1'
        result = self.dataset.groupby(by='col1')
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)  # должно быть 3 группы для значений 1, 2 и 3
        self.assertIsInstance(result[0][1], Dataset)

    def test_groupby_with_func(self):
        # Агрегация с функцией sum
        result = self.dataset.groupby(by='col1', func='sum')
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)  # Проверка числа групп
        self.assertIsInstance(result[0][1], Dataset)
        
    def test_groupby_with_fields_list(self):
        # Отбор столбцов после группировки
        result = self.dataset.groupby(by='col1', fields_list=['col2'])
        self.assertIsInstance(result, list)
        self.assertEqual(result[0][1].data.columns.tolist(), ['col2'])

    def test_groupby_with_invalid_by_type(self):
        # Проверка на неправильный тип для 'by'
        with self.assertRaises(KeyError):
            self.dataset.groupby(by=123)

    def test_groupby_with_empty_dataset(self):
        # Проверка на пустой Dataset
        empty_dataset = Dataset(roles=self.roles, data=pd.DataFrame(columns=['col1', 'col2']))
        result = empty_dataset.groupby(by='col1')
        self.assertEqual(result, [])

    def test_groupby_with_tmp_roles(self):
        # Проверка, что tmp_roles сохраняются после группировки
        result = self.dataset.groupby(by='col1')
        self.assertTrue(all(dataset[1].tmp_roles == self.dataset.tmp_roles for dataset in result))
    
    def test_groupby_with_fields_list_and_func(self):
        # Группировка с полями и функцией
        result = self.dataset.groupby(by='col1', fields_list=['col2'], func='sum')
        self.assertIsInstance(result, list)
        self.assertEqual([int(x[1]) for x in result], self.data['col2'].tolist())

    def test_idxmax(self):
        # Test basic idxmax
        result = self.dataset.idxmax()
        self.assertEqual(result['col1'], 2)  # Index of max value (3)

        # Test with skipna=False
        self.dataset.data.loc[2, 'col1'] = None
        result = self.dataset.idxmax(skipna=False)
        self.assertTrue(pd.isna(result['col1']))

        # Test with axis=1
        result = self.dataset.idxmax(axis=1)
        self.assertEqual(result[0], 'col2')

        # Edge cases
        # Test with all equal values
        self.dataset.data['col1'] = [1, 1, 1]
        result = self.dataset.idxmax()
        self.assertEqual(result['col1'], 0)  # Returns first occurrence

        # Test with all NaN values
        self.dataset.data['col1'] = None
        result = self.dataset.idxmax()
        self.assertTrue(pd.isna(result['col1']))

        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        with self.assertRaises(ValueError):
            empty_dataset.idxmax()

        # Test with non-numeric data
        self.dataset.data['col1'] = ['a', 'b', 'c']
        with self.assertRaises(TypeError):
            self.dataset.idxmax()

    def test_is_empty(self):
        # Test non-empty dataset
        self.assertFalse(self.dataset.is_empty())

        # Test empty dataset
        empty_dataset = Dataset.create_empty(roles=self.roles)
        self.assertTrue(empty_dataset.is_empty())

        # Test dataset with empty DataFrame
        empty_df = pd.DataFrame(columns=['col1', 'col2'])
        empty_dataset = Dataset(roles=self.roles, data=empty_df)
        self.assertTrue(empty_dataset.is_empty())

        # Edge cases
        # Test with NaN values
        self.dataset.data[:] = None
        self.assertFalse(self.dataset.is_empty())

        # Test with zero rows but with columns
        empty_df = pd.DataFrame(columns=['col1', 'col2'])
        empty_dataset = Dataset(roles=self.roles, data=empty_df)
        self.assertTrue(empty_dataset.is_empty())

        # Test with one empty column
        self.dataset.data = pd.DataFrame({'col1': [], 'col2': []})
        self.assertTrue(self.dataset.is_empty())

    def test_isna(self):
        # Test with no NaN values
        result = self.dataset.isna()
        self.assertFalse(result.data.any().any())

        # Test with NaN values
        self.dataset.data.loc[0, 'col1'] = None
        result = self.dataset.isna()
        self.assertTrue(result.data.loc[0, 'col1'])
        self.assertFalse(result.data.loc[0, 'col2'])

        # Edge cases
        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.isna()
        self.assertTrue(result.data.all().all())

        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.isna()
        self.assertEqual(len(result.data), 0)

        # Test with mixed types
        self.dataset.data['col1'] = [1, 'two', None]
        result = self.dataset.isna()
        self.assertTrue(result.data.loc[2, 'col1'])

    def test_isin(self):
        # Test with list
        result = self.dataset.isin([1, 4])
        self.assertTrue(result.data.loc[0, 'col1'])
        self.assertTrue(result.data.loc[0, 'col2'])

        # Test with dict
        result = self.dataset.isin({'col1': [1, 2], 'col2': [4]})
        self.assertTrue(result.data.loc[0, 'col1'])
        self.assertTrue(result.data.loc[0, 'col2'])

        # Test with Series
        result = self.dataset.isin(pd.Series([1, 4]))
        self.assertTrue(result.data.loc[0, 'col1'])
        self.assertTrue(result.data.loc[0, 'col2'])

        # Edge cases
        # Test with empty values
        result = self.dataset.isin([])
        self.assertFalse(result.data.any().any())

        # Test with None values
        result = self.dataset.isin([None])
        self.assertFalse(result.data.any().any())

        # Test with mixed types
        self.dataset.data['col1'] = [1, 'two', 3]
        result = self.dataset.isin([1, 'two'])
        self.assertTrue(result.data.loc[0, 'col1'])
        self.assertTrue(result.data.loc[1, 'col1'])

    def test_log(self):
        # Test basic log
        result = self.dataset.log()
        self.assertTrue(all(result.data['col1'] > 0))

        # Test with base parameter
        result = self.dataset.log(2)  # base 2
        self.assertTrue(all(result.data['col1'] > 0))

        # Test with negative values
        self.dataset.data.loc[0, 'col1'] = -1
        result = self.dataset.log()
        self.assertTrue(pd.isna(result.data.loc[0, 'col1']))

        # Edge cases
        # Test with zero values
        self.dataset.data.loc[0, 'col1'] = 0
        result = self.dataset.log()
        self.assertTrue(pd.isna(result.data.loc[0, 'col1']))

        # Test with NaN values
        self.dataset.data.loc[0, 'col1'] = None
        result = self.dataset.log()
        self.assertTrue(pd.isna(result.data.loc[0, 'col1']))

        # Test with very large values
        self.dataset.data.loc[0, 'col1'] = 1e308
        result = self.dataset.log()
        self.assertTrue(np.isfinite(result.data.loc[0, 'col1']))

    def test_map(self):
        # Test with function
        result = self.dataset.map(lambda x: x * 2)
        self.assertEqual(result.data['col1'].tolist(), [2, 4, 6])

        # Test with dict
        result = self.dataset.map({'col1': {1: 10, 2: 20, 3: 30}})
        self.assertEqual(result.data['col1'].tolist(), [10, 20, 30])

        # Test with Series
        mapping = pd.Series({1: 10, 2: 20, 3: 30})
        result = self.dataset.map({'col1': mapping})
        self.assertEqual(result.data['col1'].tolist(), [10, 20, 30])

        # Edge cases
        # Test with empty mapping
        result = self.dataset.map({})
        self.assertTrue(result.data.equals(self.dataset.data))

        # Test with missing keys in mapping
        result = self.dataset.map({'col1': {1: 10}})
        self.assertTrue(pd.isna(result.data.loc[1:, 'col1']).all())

        # Test with None values in mapping
        result = self.dataset.map({'col1': {1: None, 2: 20, 3: 30}})
        self.assertTrue(pd.isna(result.data.loc[0, 'col1']))

    def test_max(self):
        # Test basic max
        result = self.dataset.max()
        self.assertEqual(result['col1'], 3)

        # Test with skipna=False
        self.dataset.data.loc[2, 'col1'] = None
        result = self.dataset.max(skipna=False)
        self.assertTrue(pd.isna(result['col1']))

        # Test with axis=1
        result = self.dataset.max(axis=1)
        self.assertEqual(result[0], 4)

        # Edge cases
        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.max()
        self.assertTrue(pd.isna(result['col1']))

        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.max()
        self.assertTrue(pd.isna(result['col1']))

        # Test with mixed types
        self.dataset.data['col1'] = [1, 'two', 3]
        with self.assertRaises(TypeError):
            result = self.dataset.max()

    def test_min(self):
        # Test basic min
        result = self.dataset.min()
        self.assertEqual(result['col1'], 1)

        # Test with skipna=False
        self.dataset.data.loc[0, 'col1'] = None
        result = self.dataset.min(skipna=False)
        self.assertTrue(pd.isna(result['col1']))

        # Test with axis=1
        result = self.dataset.min(axis=1)
        self.assertEqual(result[1], 2)

        # Edge cases
        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.min()
        self.assertTrue(pd.isna(result['col1']))

        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.min()
        self.assertTrue(pd.isna(result['col1']))

        # Test with mixed types
        self.dataset.data['col1'] = [1, 'two', 3]
        with self.assertRaises(TypeError):
            result = self.dataset.min()

    def test_mode(self):
        # Test basic mode
        result = self.dataset.mode()
        self.assertIsInstance(result, Dataset)

        # Test with duplicate values
        self.dataset.data.loc[0, 'col1'] = 2
        result = self.dataset.mode()
        self.assertEqual(result.data['col1'][0], 2)

        # Test with numeric_only=True
        self.dataset.add_column(['A', 'A', 'B'], {'group': InfoRole()})
        result = self.dataset.mode(numeric_only=True)
        self.assertNotIn('group', result.columns)

        # Edge cases
        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.mode()
        self.assertTrue(result.data.empty)

        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.mode()
        self.assertTrue(result.data.empty)

        # Test with multiple modes
        self.dataset.data['col1'] = [1, 1, 2, 2, 3]
        result = self.dataset.mode()
        self.assertEqual(len(result.data), 2)

    def test_na_counts(self):
        # Test with no NaN values
        result = self.dataset.na_counts()
        self.assertEqual(result['col1'], 0)

        # Test with NaN values
        self.dataset.data.loc[0, 'col1'] = None
        self.dataset.data.loc[1, 'col1'] = None
        result = self.dataset.na_counts()
        self.assertEqual(result['col1'], 2)

        # Test with different types of missing values
        self.dataset.data.loc[2, 'col1'] = np.nan
        result = self.dataset.na_counts()
        self.assertEqual(result['col1'], 3)

        # Edge cases
        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.na_counts()
        self.assertEqual(result['col1'], 0)

        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.na_counts()
        self.assertEqual(result['col1'], len(self.dataset.data))

        # Test with mixed types including strings
        self.dataset.data['col1'] = [None, 'NA', np.nan, pd.NA]
        result = self.dataset.na_counts()
        self.assertEqual(result['col1'], 3)  # 'NA' string is not counted as NaN

    def test_nunique(self):
        # Test basic nunique
        result = self.dataset.nunique()
        self.assertEqual(result['col1'], 3)

        # Test with dropna=False
        self.dataset.data.loc[0, 'col1'] = None
        result = self.dataset.nunique(dropna=False)
        self.assertEqual(result['col1'], 4)  # None counts as unique value

        # Test with duplicate values
        self.dataset.data.loc[1, 'col1'] = 2
        result = self.dataset.nunique()
        self.assertEqual(result['col1'], 2)

        # Edge cases
        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.nunique()
        self.assertEqual(result['col1'], 0)

        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.nunique()
        self.assertEqual(result['col1'], 0)

        # Test with mixed types
        self.dataset.data['col1'] = [1, 'two', 3, 'two', None]
        result = self.dataset.nunique()
        self.assertEqual(result['col1'], 3)

    def test_quantile(self):
        # Test single quantile
        result = self.dataset.quantile(0.5)
        self.assertEqual(result['col1'], 2)

        # Test multiple quantiles
        result = self.dataset.quantile([0.25, 0.75])
        self.assertEqual(len(result), 2)

        # Test with interpolation method
        result = self.dataset.quantile(0.5, interpolation='nearest')
        self.assertEqual(result['col1'], 2)

        # Edge cases
        # Test with invalid quantile values
        with self.assertRaises(ValueError):
            result = self.dataset.quantile(1.5)

        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.quantile(0.5)
        self.assertTrue(pd.isna(result['col1']))

        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.quantile(0.5)
        self.assertTrue(pd.isna(result['col1']))

    def test_reindex(self):
        # Test columns reindex
        result = self.dataset.reindex(['col1'])
        self.assertEqual(list(result.columns), ['col1'])

        # Test index reindex
        result = self.dataset.reindex(index=[0, 1, 2, 3])
        self.assertEqual(len(result), 4)
        self.assertTrue(pd.isna(result.data.loc[3, 'col1']))

        # Test with fill_value
        result = self.dataset.reindex(index=[0, 1, 2, 3], fill_value=0)
        self.assertEqual(result.data.loc[3, 'col1'], 0)

        # Edge cases
        # Test with empty index
        result = self.dataset.reindex(index=[])
        self.assertEqual(len(result), 0)

        # Test with duplicate indices
        result = self.dataset.reindex(index=[0, 0, 1])
        self.assertEqual(len(result), 3)

        # Test with non-existent columns
        result = self.dataset.reindex(columns=['non_existent'])
        self.assertTrue(result.data.empty)

    def test_rename(self):
        # Test with dict
        result = self.dataset.rename({'col1': 'new_col1'})
        self.assertIn('new_col1', result.columns)

        # Test with function
        result = self.dataset.rename(lambda x: x.upper())
        self.assertIn('COL1', result.columns)

        # Test with axis=0 (index)
        result = self.dataset.rename({0: 'first'}, axis=0)
        self.assertEqual(result.data.index[0], 'first')

        # Edge cases
        # Test with empty mapping
        result = self.dataset.rename({})
        self.assertEqual(list(result.columns), list(self.dataset.columns))

        # Test with non-existent columns
        result = self.dataset.rename({'non_existent': 'new_name'})
        self.assertEqual(list(result.columns), list(self.dataset.columns))

        # Test with duplicate names
        with self.assertRaises(ValueError):
            result = self.dataset.rename({'col1': 'col2'})

    def test_replace(self):
        # Test replace single value
        result = self.dataset.replace(1, 100)
        self.assertEqual(result.data.loc[0, 'col1'], 100)

        # Test replace with dict
        result = self.dataset.replace({1: 100, 2: 200})
        self.assertEqual(result.data.loc[0, 'col1'], 100)
        self.assertEqual(result.data.loc[1, 'col1'], 200)

        # Test replace with regex
        result = self.dataset.replace(regex=r'^1$', value=100)
        self.assertEqual(result.data.loc[0, 'col1'], 100)

        # Edge cases
        # Test replace with empty dict
        result = self.dataset.replace({})
        self.assertTrue(result.data.equals(self.dataset.data))

        # Test replace with None values
        result = self.dataset.replace({1: None})
        self.assertTrue(pd.isna(result.data.loc[0, 'col1']))

        # Test replace with non-existent values
        result = self.dataset.replace({999: 1000})
        self.assertTrue(result.data.equals(self.dataset.data))

    def test_sample(self):
        # Test with n parameter
        result = self.dataset.sample(n=2, random_state=42)
        self.assertEqual(len(result), 2)

        # Test with frac parameter
        result = self.dataset.sample(frac=0.5, random_state=42)
        self.assertEqual(len(result), 1)

        # Test with replace=True
        result = self.dataset.sample(n=4, replace=True, random_state=42)
        self.assertEqual(len(result), 4)

        # Edge cases
        # Test with n=0
        result = self.dataset.sample(n=0)
        self.assertEqual(len(result), 0)

        # Test with frac=0
        result = self.dataset.sample(frac=0)
        self.assertEqual(len(result), 0)

        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.sample(n=1)
        self.assertEqual(len(result), 0)

    def test_select_dtypes(self):
        # Test include
        result = self.dataset.select_dtypes(include=['int64'])
        self.assertEqual(len(result.columns), 2)

        # Test exclude
        result = self.dataset.select_dtypes(exclude=['int64'])
        self.assertEqual(len(result.columns), 0)

        # Test with multiple types
        self.dataset.data['col1'] = self.dataset.data['col1'].astype(float)
        result = self.dataset.select_dtypes(include=['int64', 'float64'])
        self.assertEqual(len(result.columns), 2)

        # Edge cases
        # Test with empty include list
        result = self.dataset.select_dtypes(include=[])
        self.assertEqual(len(result.columns), 0)

        # Test with non-existent dtype
        result = self.dataset.select_dtypes(include=['non_existent_dtype'])
        self.assertEqual(len(result.columns), 0)

        # Test with mixed types
        self.dataset.data['col3'] = ['a', 'b', 'c']
        result = self.dataset.select_dtypes(include=['object'])
        self.assertEqual(len(result.columns), 1)

    def test_sort(self):
        # Test basic sort
        result = self.dataset.sort(by='col1', ascending=False)
        self.assertEqual(result.data['col1'].tolist(), [3, 2, 1])

        # Test with multiple columns
        result = self.dataset.sort(by=['col1', 'col2'], ascending=[False, True])
        self.assertEqual(result.data['col1'].tolist(), [3, 2, 1])

        # Test with na_position
        self.dataset.data.loc[0, 'col1'] = None
        result = self.dataset.sort(by='col1', na_position='first')
        self.assertTrue(pd.isna(result.data['col1'].iloc[0]))

        # Edge cases
        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.sort(by='col1')
        self.assertTrue(result.data.empty)

        # Test with non-existent column
        with self.assertRaises(KeyError):
            result = self.dataset.sort(by='non_existent')

        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.sort(by='col1')
        self.assertTrue(pd.isna(result.data['col1']).all())

    def test_std(self):
        # Test basic std
        result = self.dataset.std()
        self.assertIsInstance(result, pd.Series)

        # Test with ddof parameter
        result = self.dataset.std(ddof=0)
        self.assertIsInstance(result, pd.Series)

        # Test with skipna parameter
        self.dataset.data.loc[0, 'col1'] = None
        result = self.dataset.std(skipna=False)
        self.assertTrue(pd.isna(result['col1']))

        # Edge cases
        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.std()
        self.assertTrue(pd.isna(result['col1']))

        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.std()
        self.assertTrue(pd.isna(result['col1']))

        # Test with single value
        self.dataset.data['col1'] = [1, 1, 1]
        result = self.dataset.std()
        self.assertEqual(result['col1'], 0)

    def test_sum(self):
        # Test basic sum
        result = self.dataset.sum()
        self.assertEqual(result['col1'], 6)

        # Test with skipna parameter
        self.dataset.data.loc[0, 'col1'] = None
        result = self.dataset.sum(skipna=True)
        self.assertEqual(result['col1'], 5)

        # Test with axis=1
        result = self.dataset.sum(axis=1)
        self.assertTrue(pd.isna(result[0]))
        self.assertEqual(result[1], 7)

        # Edge cases
        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.sum()
        self.assertEqual(result['col1'], 0)

        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.sum()
        self.assertEqual(result['col1'], 0)

        # Test with mixed types
        self.dataset.data['col1'] = [1, 'two', 3]
        with self.assertRaises(TypeError):
            result = self.dataset.sum()

    def test_transpose(self):
        # Test basic transpose
        result = self.dataset.transpose()
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], 3)

        # Test with copy parameter
        result = self.dataset.transpose(copy=True)
        self.assertFalse(result.data is self.dataset.data)

        # Edge cases
        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.transpose()
        self.assertTrue(result.data.empty)

        # Test with single row
        single_row_dataset = self.dataset.head(1)
        result = single_row_dataset.transpose()
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], 1)

        # Test with single column
        single_col_dataset = self.dataset[['col1']]
        result = single_col_dataset.transpose()
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], 3)

    def test_unique(self):
        # Test basic unique
        result = self.dataset.unique()
        self.assertEqual(len(result['col1']), 3)

        # Test with duplicate values
        self.dataset.data.loc[0, 'col1'] = 2
        result = self.dataset.unique()
        self.assertEqual(len(result['col1']), 2)

        # Test with NaN values
        self.dataset.data.loc[0, 'col1'] = None
        result = self.dataset.unique()
        self.assertEqual(len(result['col1']), 3)

        # Edge cases
        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.unique()
        self.assertEqual(len(result['col1']), 0)

        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.unique()
        self.assertEqual(len(result['col1']), 1)

        # Test with mixed types
        self.dataset.data['col1'] = [1, 'two', 1, 'two', None]
        result = self.dataset.unique()
        self.assertEqual(len(result['col1']), 3)

    def test_value_counts(self):
        # Test basic value_counts
        result = self.dataset.value_counts()
        self.assertEqual(len(result['col1']), 3)

        # Test with normalize parameter
        result = self.dataset.value_counts(normalize=True)
        self.assertAlmostEqual(result['col1'].sum(), 1.0)

        # Test with dropna parameter
        self.dataset.data.loc[0, 'col1'] = None
        result = self.dataset.value_counts(dropna=False)
        self.assertEqual(len(result['col1']), 3)

        # Edge cases
        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.value_counts()
        self.assertTrue(result['col1'].empty)

        # Test with all same values
        self.dataset.data['col1'] = [1, 1, 1]
        result = self.dataset.value_counts()
        self.assertEqual(len(result['col1']), 1)
        self.assertEqual(result['col1'].iloc[0], 3)

        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.value_counts(dropna=False)
        self.assertEqual(result['col1'].iloc[0], len(self.dataset))

    def test_var(self):
        # Test basic var
        result = self.dataset.var()
        self.assertIsInstance(result, pd.Series)

        # Test with ddof parameter
        result = self.dataset.var(ddof=0)
        self.assertIsInstance(result, pd.Series)

        # Test with skipna parameter
        self.dataset.data.loc[0, 'col1'] = None
        result = self.dataset.var(skipna=False)
        self.assertTrue(pd.isna(result['col1']))

        # Edge cases
        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.var()
        self.assertTrue(pd.isna(result['col1']))

        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.var()
        self.assertTrue(pd.isna(result['col1']))

        # Test with single value
        self.dataset.data['col1'] = [1, 1, 1]
        result = self.dataset.var()
        self.assertEqual(result['col1'], 0)

    def test_properties(self):
        # Test columns property
        self.assertEqual(list(self.dataset.columns), ['col1', 'col2'])

        # Test data property
        self.assertIsInstance(self.dataset.data, pd.DataFrame)
        self.assertEqual(self.dataset.data.shape, (3, 2))

        # Test index property
        self.assertEqual(len(self.dataset.index), 3)
        self.assertTrue(all(isinstance(i, int) for i in self.dataset.index))

        # Test shape property
        self.assertEqual(self.dataset.shape, (3, 2))

        # Test size property
        self.assertEqual(self.dataset.size, 6)

        # Test dtypes property
        self.assertEqual(len(self.dataset.dtypes), 2)
        self.assertTrue(all(dtype == 'int64' for dtype in self.dataset.dtypes))

    # Тесты для бинарных операторов сравнения
    def test_eq_operator(self):
        other_dataset = Dataset(roles=self.roles, data=self.data, backend=BackendsEnum.pandas)
        result = self.dataset == other_dataset
        self.assertTrue(result)  # Ожидаем True, так как данные одинаковые

    def test_ne_operator(self):
        other_dataset = Dataset(roles=self.roles, data=self.data, backend=BackendsEnum.pandas)
        result = self.dataset != other_dataset
        self.assertFalse(result)  # Ожидаем False, так как данные одинаковые

    def test_le_operator(self):
        other_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4, 5, 6]
        })
        other_dataset = Dataset(roles=self.roles, data=other_data, backend=BackendsEnum.pandas)
        result = self.dataset <= other_dataset
        self.assertTrue(result)  # Ожидаем True, так как все элементы меньше или равны

    # Тесты для унарных операторов
    def test_pos_operator(self):
        result = +self.dataset
        self.assertIsInstance(result, Dataset)  # Ожидаем возвращение Dataset
        self.assertTrue((result.data >= 0).all().all())  # Ожидаем, что все элементы >= 0

    def test_neg_operator(self):
        result = -self.dataset
        self.assertIsInstance(result, Dataset)  # Ожидаем возвращение Dataset
        self.assertTrue((result.data < 0).all().all())  # Ожидаем, что все элементы < 0

    def test_abs_operator(self):
        result = abs(self.dataset)
        self.assertIsInstance(result, Dataset)  # Ожидаем возвращение Dataset
        self.assertTrue((result.data >= 0).all().all())  # Ожидаем, что все элементы >= 0

    def test_invert_operator(self):
        result = ~self.dataset
        self.assertIsInstance(result, Dataset)  # Ожидаем возвращение Dataset
        # Допустим, что инверсия работает с битами, проверим корректность данных
        self.assertTrue((result.data.isin([0, 1])).all().all())  # Ожидаем 0 или 1

    def test_round_operator(self):
        result = round(self.dataset, 2)
        self.assertIsInstance(result, Dataset)  # Ожидаем возвращение Dataset
        # Проверим, что округление работает корректно
        self.assertTrue((result.data == self.dataset.data.round(2)).all().all())

    def test_bool_operator(self):
        result = bool(self.dataset)
        self.assertTrue(result)  # Ожидаем, что не пустой Dataset вернет True

    # Тесты для бинарных математических операторов
    def test_add_operator(self):
        other_data = pd.DataFrame({
            'col1': [1, 1, 1],
            'col2': [1, 1, 1]
        })
        other_dataset = Dataset(roles=self.roles, data=other_data, backend=BackendsEnum.pandas)
        result = self.dataset + other_dataset
        self.assertIsInstance(result, Dataset)  # Ожидаем, что результат будет Dataset
        # Проверим корректность сложения
        expected_data = self.data + other_data
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_sub_operator(self):
        other_data = pd.DataFrame({
            'col1': [1, 1, 1],
            'col2': [1, 1, 1]
        })
        other_dataset = Dataset(roles=self.roles, data=other_data, backend=BackendsEnum.pandas)
        result = self.dataset - other_dataset
        self.assertIsInstance(result, Dataset)  # Ожидаем, что результат будет Dataset
        # Проверим корректность вычитания
        expected_data = self.data - other_data
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_mul_operator(self):
        other_data = pd.DataFrame({
            'col1': [1, 1, 1],
            'col2': [1, 1, 1]
        })
        other_dataset = Dataset(roles=self.roles, data=other_data, backend=BackendsEnum.pandas)
        result = self.dataset * other_dataset
        self.assertIsInstance(result, Dataset)  # Ожидаем, что результат будет Dataset
        # Проверим корректность умножения
        expected_data = self.data * other_data
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_floordiv_operator(self):
        other_data = pd.DataFrame({
            'col1': [1, 1, 1],
            'col2': [1, 1, 1]
        })
        other_dataset = Dataset(roles=self.roles, data=other_data, backend=BackendsEnum.pandas)
        result = self.dataset // other_dataset
        self.assertIsInstance(result, Dataset)  # Ожидаем, что результат будет Dataset
        # Проверим корректность целочисленного деления
        expected_data = self.data // other_data
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_truediv_operator(self):
        other_data = pd.DataFrame({
            'col1': [1, 1, 1],
            'col2': [1, 1, 1]
        })
        other_dataset = Dataset(roles=self.roles, data=other_data, backend=BackendsEnum.pandas)
        result = self.dataset / other_dataset
        self.assertIsInstance(result, Dataset)  # Ожидаем, что результат будет Dataset
        # Проверим корректность деления
        expected_data = self.data / other_data
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_rdiv_operator(self):
        other_data = pd.DataFrame({
            'col1': [1, 1, 1],
            'col2': [1, 1, 1]
        })
        other_dataset = Dataset(roles=self.roles, data=other_data, backend=BackendsEnum.pandas)
        result = other_dataset / self.dataset
        self.assertIsInstance(result, Dataset)  # Ожидаем, что результат будет Dataset
        # Проверим корректность "обратного деления"
        expected_data = other_data / self.data
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_mod_operator(self):
        other_data = pd.DataFrame({
            'col1': [2, 3, 4],
            'col2': [1, 2, 3]
        })
        other_dataset = Dataset(roles=self.roles, data=other_data, backend=BackendsEnum.pandas)
        result = self.dataset % other_dataset
        self.assertIsInstance(result, Dataset)  # Ожидаем, что результат будет Dataset
        # Проверим корректность остатка от деления
        expected_data = self.data % other_data
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_pow_operator(self):
        other_data = pd.DataFrame({
            'col1': [2, 3, 2],
            'col2': [1, 2, 3]
        })
        other_dataset = Dataset(roles=self.roles, data=other_data, backend=BackendsEnum.pandas)
        result = self.dataset ** other_dataset
        self.assertIsInstance(result, Dataset)  # Ожидаем, что результат будет Dataset
        # Проверим корректность возведения в степень
        expected_data = self.data ** other_data
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_and_operator(self):
        other_data = pd.DataFrame({
            'col1': [1, 1, 0],
            'col2': [1, 0, 0]
        })
        other_dataset = Dataset(roles=self.roles, data=other_data, backend=BackendsEnum.pandas)
        result = self.dataset & other_dataset
        self.assertIsInstance(result, Dataset)  # Ожидаем, что результат будет Dataset
        # Проверим корректность побитовой операции И
        expected_data = self.data & other_data
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_or_operator(self):
        other_data = pd.DataFrame({
            'col1': [1, 0, 0],
            'col2': [0, 1, 1]
        })
        other_dataset = Dataset(roles=self.roles, data=other_data, backend=BackendsEnum.pandas)
        result = self.dataset | other_dataset
        self.assertIsInstance(result, Dataset)  # Ожидаем, что результат будет Dataset
        # Проверим корректность побитовой операции ИЛИ
        expected_data = self.data | other_data
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_radd_operator(self):
        # Тест правого оператора сложения
        result = 5 + self.dataset
        self.assertIsInstance(result, Dataset)
        expected_data = 5 + self.data
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_rsub_operator(self):
        # Тест правого оператора вычитания
        result = 10 - self.dataset
        self.assertIsInstance(result, Dataset)
        expected_data = 10 - self.data
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_rmul_operator(self):
        # Тест правого оператора умножения
        result = 3 * self.dataset
        self.assertIsInstance(result, Dataset)
        expected_data = 3 * self.data
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_rfloordiv_operator(self):
        # Тест правого оператора целочисленного деления
        result = 7 // self.dataset
        self.assertIsInstance(result, Dataset)
        expected_data = 7 // self.data
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_rdiv_operator(self):
        # Тест правого оператора деления
        result = 8 / self.dataset
        self.assertIsInstance(result, Dataset)
        expected_data = 8 / self.data
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_rtruediv_operator(self):
        # Тест правого оператора деления с плавающей точкой
        result = 8.0 / self.dataset
        self.assertIsInstance(result, Dataset)
        expected_data = 8.0 / self.data
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_rmod_operator(self):
        # Тест правого оператора остатка от деления
        result = 9 % self.dataset
        self.assertIsInstance(result, Dataset)
        expected_data = 9 % self.data
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_rpow_operator(self):
        # Тест правого оператора возведения в степень
        result = 2 ** self.dataset
        self.assertIsInstance(result, Dataset)
        expected_data = 2 ** self.data
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_rdiv_operator(self):
        # Проверка правого деления
        result = 10 / self.dataset
        self.assertIsInstance(result, Dataset)
        expected_data = 10 / self.data
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_div_operator(self):
        # Проверка деления
        result = self.dataset / 2
        self.assertIsInstance(result, Dataset)
        expected_data = self.data / 2
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_lt_operator(self):
        # Проверка оператора < (меньше)
        result = self.dataset < 3
        self.assertIsInstance(result, Dataset)
        expected_data = self.data < 3
        pd.testing.assert_frame_equal(result.data, expected_data)

    def test_ge_operator(self):
        # Проверка оператора >= (больше или равно)
        result = self.dataset >= 3
        self.assertIsInstance(result, Dataset)
        expected_data = self.data >= 3
        pd.testing.assert_frame_equal(result.data, expected_data)

if __name__ == '__main__':
    unittest.main()