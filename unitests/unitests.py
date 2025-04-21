import copy
import unittest

import numpy as np
import pandas as pd

from hypex.dataset import Dataset, ExperimentData
from hypex.dataset.roles import DefaultRole, InfoRole, TargetRole
from hypex.utils import (
    BackendsEnum,
    ConcatDataError,
    DataTypeError,
    ExperimentDataEnum,
    RoleColumnError,
)


class TestDataset(unittest.TestCase):

    def setUp(self):
        # Initialize test data and roles
        self.roles = {"col1": InfoRole(int), "col2": InfoRole(int)}
        self.data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        self.dataset = Dataset(roles=self.roles, data=self.data)

    def ged_dataset_copy(self):
        return copy.deepcopy(self.dataset)

    def test_initialization(self):
        # Test basic dataset initialization
        self.assertEqual(len(self.dataset), 3)
        self.assertListEqual(list(self.dataset.columns), ["col1", "col2"])
        self.assertIn("col1", self.dataset.roles)

    def test_role_initialization(self):
        # Test initialization with role mapping
        roles_with_mapping = {InfoRole(): ["col1", "col2"]}
        dataset_with_mapping = Dataset(roles=roles_with_mapping, data=self.data)
        self.assertListEqual(list(dataset_with_mapping.columns), ["col1", "col2"])

    def test_backend_initialization(self):
        # Test initialization with backend
        roles_with_mapping = {InfoRole(): ["col1", "col2"]}
        dataset_with_backend = Dataset(
            roles=roles_with_mapping, data=self.data, backend=BackendsEnum.pandas
        )
        self.assertListEqual(list(dataset_with_backend.columns), ["col1", "col2"])
        self.assertEqual(dataset_with_backend.backend.name, "pandasdataset")

    def test_partial_role_initialization(self):
        # Test initialization with partial roles
        roles_partial = {InfoRole(): ["col1"]}
        dataset_partial = Dataset(roles=roles_partial, data=self.data)
        self.assertListEqual(list(dataset_partial.columns), ["col1", "col2"])
        self.assertEqual(str(dataset_partial), str(dataset_partial.data))
        self.assertEqual(
            dataset_partial._repr_html_(), dataset_partial.data._repr_html_()
        )

    def test_invalid_role_initialization(self):
        roles_invalid = {InfoRole(): ["col1", "col3"]}
        with self.assertRaises(RoleColumnError):
            Dataset(roles=roles_invalid, data=self.data)

    def test_merge(self):
        # Test merging datasets
        data_new = pd.DataFrame({"col1": [1, 2, 3], "col3": [7, 8, 9]})
        roles_new = {"col1": InfoRole(), "col3": TargetRole()}
        dataset_new = Dataset(roles=roles_new, data=data_new)

        merged = self.dataset.merge(dataset_new, on="col1")
        self.assertIn("col3", merged.columns)
        self.assertTrue(isinstance(merged.roles["col1"], InfoRole))
        self.assertTrue(isinstance(merged.roles["col2"], InfoRole))
        self.assertTrue(isinstance(merged.roles["col3"], TargetRole))

    def test_operators(self):
        # Test arithmetic operators
        result = self.dataset + 1
        self.assertListEqual(result.data["col1"].tolist(), [2, 3, 4])

    def test_roles_property(self):
        # Test roles property access
        dataset = Dataset(roles=self.roles, data=self.data)

        self.assertIn("col1", dataset.roles)
        self.assertIsInstance(dataset.roles["col1"], InfoRole)

    def test_replace_roles(self):
        # Test role replacement functionality
        dataset = Dataset(roles=self.roles, data=self.data)

        new_role = DefaultRole()
        dataset.replace_roles({"col1": new_role})
        self.assertIsInstance(dataset.roles["col1"], DefaultRole)

    def test_search_columns(self):
        # Test column search by role
        dataset_base = Dataset(roles=self.roles, data=self.data)

        columns = dataset_base.search_columns(InfoRole())
        self.assertListEqual(columns, ["col1", "col2"])

    def test_to_dict(self):
        # Test conversion to dictionary
        dataset_base = Dataset(roles=self.roles, data=self.data)

        result = dataset_base.to_dict()
        self.assertIn("data", result)
        self.assertIn("roles", result)

    def test_getitem_by_column(self):
        # Test getting subset by single column
        subset = self.dataset["col1"]

        # Check that data is returned for 'col1'
        self.assertIn("col1", subset.columns)
        self.assertNotIn("col2", subset.columns)

        # Check that roles for 'col1' are preserved
        self.assertIn("col1", subset.roles)
        self.assertIsInstance(subset.roles["col1"], InfoRole)

    def test_getitem_with_multiple_columns(self):
        # Test getting subset with multiple columns
        roles = {"col1": InfoRole(), "col2": InfoRole(), "col3": InfoRole()}
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})
        dataset = Dataset(roles=roles, data=data)

        # Get subset for columns 'col1' and 'col3'
        subset = dataset[["col1", "col3"]]

        # Check that data is returned for 'col1' and 'col3'
        self.assertIn("col1", subset.columns)
        self.assertIn("col3", subset.columns)
        self.assertNotIn("col2", subset.columns)

        # Check that roles for 'col1' and 'col3' are preserved
        self.assertIn("col1", subset.roles)
        self.assertIn("col3", subset.roles)
        self.assertIsInstance(subset.roles["col1"], InfoRole)
        self.assertIsInstance(subset.roles["col3"], InfoRole)

    def test_getitem_with_non_existing_column(self):
        # Test getting non-existent column
        with self.assertRaises(KeyError):
            self.dataset["col3"]

    def test_getitem_empty_result(self):
        # Test getting empty subset with condition
        subset = self.dataset[self.dataset["col1"] > 3]
        self.assertTrue(len(subset) == 0)

    def test_rename_single_column(self):
        # Test renaming single column
        renamed_dataset = self.dataset.rename({"col1": "new_col1"})

        # Check that column 'col1' is renamed to 'new_col1'
        self.assertIn("new_col1", renamed_dataset.columns)
        self.assertNotIn("col1", renamed_dataset.columns)

        # Check that role for 'col1' is renamed to 'new_col1'
        self.assertIn("new_col1", renamed_dataset.roles)
        self.assertIsInstance(renamed_dataset.roles["new_col1"], InfoRole)

    def test_rename_multiple_columns(self):
        # Test renaming multiple columns
        roles = {"col1": InfoRole(), "col2": InfoRole(), "col3": InfoRole()}
        data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6], "col3": [7, 8, 9]})
        dataset = Dataset(roles=roles, data=data)

        # Rename multiple columns
        renamed_dataset = dataset.rename({"col1": "new_col1", "col3": "new_col3"})

        # Check that columns 'col1' and 'col3' are renamed
        self.assertIn("new_col1", renamed_dataset.columns)
        self.assertIn("new_col3", renamed_dataset.columns)
        self.assertNotIn("col1", renamed_dataset.columns)
        self.assertNotIn("col3", renamed_dataset.columns)

        # Check that roles for 'col1' and 'col3' are renamed
        self.assertIn("new_col1", renamed_dataset.roles)
        self.assertIn("new_col3", renamed_dataset.roles)
        self.assertIsInstance(renamed_dataset.roles["new_col1"], InfoRole)
        self.assertIsInstance(renamed_dataset.roles["new_col3"], InfoRole)

    def test_rename_no_change(self):
        # Test renaming with no changes (empty dict)
        renamed_dataset = self.dataset.rename({})

        # Check that data remains unchanged
        self.assertEqual(list(self.dataset.columns), list(renamed_dataset.columns))
        self.assertEqual(
            list(self.dataset.roles.keys()), list(renamed_dataset.roles.keys())
        )

    def test_rename_with_non_existent_column(self):
        # Test renaming non-existent column
        renamed_dataset = self.dataset.rename({"non_existent": "new_col"})

        # Check that columns remain unchanged
        self.assertIn("col1", renamed_dataset.columns)
        self.assertIn("col2", renamed_dataset.columns)
        self.assertNotIn("new_col", renamed_dataset.columns)

    def test_rename_roles(self):
        # Test renaming columns and check role updates
        renamed_dataset = self.dataset.rename({"col1": "new_col1"})

        # Check that role for 'col1' is renamed to 'new_col1'
        self.assertIn("new_col1", renamed_dataset.roles)
        self.assertIsInstance(renamed_dataset.roles["new_col1"], InfoRole)
        self.assertNotIn("col1", renamed_dataset.roles)

    def test_set_value_additional_fields_single_column(self):
        # Test setting value for single column in additional_fields
        experiment_data = ExperimentData(self.dataset)

        # Apply set_value with single column in additional_fields
        experiment_data.set_value(
            space=ExperimentDataEnum.additional_fields,
            executor_id="executor_1",
            value=self.dataset,
        )

        # Check that column was added to additional_fields
        self.assertIn("executor_1", experiment_data.additional_fields.columns)

    def test_set_value_additional_fields_multiple_columns(self):
        # Test setting value for multiple columns in additional_fields
        experiment_data = ExperimentData(self.dataset)

        # Apply set_value with multiple columns in additional_fields
        experiment_data.set_value(
            space=ExperimentDataEnum.additional_fields,
            executor_id={"col1": "executor_1", "col2": "executor_2"},
            value=self.dataset,
        )

        # Check that columns were renamed and added
        self.assertIn("executor_1", experiment_data.additional_fields.columns)
        self.assertIn("executor_2", experiment_data.additional_fields.columns)

    def test_set_value_analysis_tables(self):
        # Test setting value for analysis_tables
        experiment_data = ExperimentData(self.dataset)

        # Apply set_value for analysis_tables
        experiment_data.set_value(
            space=ExperimentDataEnum.analysis_tables,
            executor_id="executor_1",
            value="analysis_data",
        )

        # Check that data was added to analysis_tables
        self.assertIn("executor_1", experiment_data.analysis_tables)
        self.assertEqual(experiment_data.analysis_tables["executor_1"], "analysis_data")

    def test_set_value_variables_dict(self):
        # Test setting value with Dict type
        experiment_data = ExperimentData(self.dataset)

        # Apply set_value with Dict value
        experiment_data.set_value(
            space=ExperimentDataEnum.variables,
            executor_id="executor_3",
            value={"key1": "value1", "key2": "value2"},
        )

        # Check that data was added to variables
        self.assertIn("executor_3", experiment_data.variables)
        self.assertEqual(
            experiment_data.variables["executor_3"],
            {"key1": "value1", "key2": "value2"},
        )

    def test_set_value_variables_existing_executor(self):
        # Test setting value for existing executor
        experiment_data = ExperimentData(self.dataset)
        experiment_data.variables = {"executor_1": {"key1": "value1"}}

        # Apply set_value for existing executor_id
        experiment_data.set_value(
            space=ExperimentDataEnum.variables,
            executor_id="executor_1",
            value="new_value",
            key="key2",
        )

        # Check that value in variables was updated
        self.assertEqual(experiment_data.variables["executor_1"]["key2"], "new_value")

    def test_set_value_variables_new_executor(self):
        # Test setting value for new executor
        experiment_data = ExperimentData(self.dataset)

        # Apply set_value for new executor_id
        experiment_data.set_value(
            space=ExperimentDataEnum.variables,
            executor_id="executor_2",
            value="new_value",
            key="key1",
        )

        # Check that new executor_id was added to variables
        self.assertIn("executor_2", experiment_data.variables)
        self.assertEqual(experiment_data.variables["executor_2"]["key1"], "new_value")

    def test_set_value_groups(self):
        # Test setting value for groups
        experiment_data = ExperimentData(self.dataset)

        # Apply set_value for groups
        experiment_data.set_value(
            space=ExperimentDataEnum.groups,
            executor_id="executor_1",
            value="group_data",
            key="key1",
        )

        # Check that data was added to groups
        self.assertIn("executor_1", experiment_data.groups)
        self.assertEqual(experiment_data.groups["executor_1"]["key1"], "group_data")

    def test_set_value_groups_existing_executor(self):
        # Test setting value for existing executor in groups
        experiment_data = ExperimentData(self.dataset)
        experiment_data.groups = {"executor_1": {"key1": "old_value"}}

        # Apply set_value for existing executor_id
        experiment_data.set_value(
            space=ExperimentDataEnum.groups,
            executor_id="executor_1",
            value="new_group_data",
            key="key2",
        )

        # Check that data in groups was updated
        self.assertEqual(experiment_data.groups["executor_1"]["key2"], "new_group_data")

    def test_create_empty(self):
        # Test creating empty ExperimentData
        experiment_data = ExperimentData.create_empty()

        # Check that ExperimentData object was created
        self.assertIsInstance(experiment_data, ExperimentData)
        self.assertIsInstance(experiment_data.ds, Dataset)
        self.assertIsInstance(experiment_data.additional_fields, Dataset)

    def test_check_hash_additional_fields(self):
        # Test hash checking in additional_fields
        dataset = Dataset.create_empty()
        experiment_data = ExperimentData(dataset)

        # Check that check_hash returns True if executor_id exists in additional_fields
        executor_id = "executor_1"
        experiment_data.additional_fields = (
            experiment_data.additional_fields.add_column(
                Dataset(
                    roles={executor_id: InfoRole()},
                    data=pd.DataFrame({executor_id: [1, 2, 3]}),
                )
            )
        )

        self.assertTrue(
            experiment_data.check_hash(
                executor_id, ExperimentDataEnum.additional_fields
            )
        )

        # Check that check_hash returns False for non-existent executor_id
        self.assertFalse(
            experiment_data.check_hash(
                "nonexistent_executor", ExperimentDataEnum.additional_fields
            )
        )

    def test_check_hash_variables(self):
        # Test hash checking in variables
        dataset = Dataset.create_empty()
        experiment_data = ExperimentData(dataset)

        # Add executor_id to variables
        experiment_data.variables["executor_1"] = {"key1": 1}

        # Check that check_hash returns True for existing executor_id
        self.assertTrue(
            experiment_data.check_hash("executor_1", ExperimentDataEnum.variables)
        )

        # Check that check_hash returns False for non-existent executor_id
        self.assertFalse(
            experiment_data.check_hash(
                "nonexistent_executor", ExperimentDataEnum.variables
            )
        )

    def test_check_hash_analysis_tables(self):
        # Test hash checking in analysis_tables
        dataset = Dataset.create_empty()
        experiment_data = ExperimentData(dataset)

        # Add executor_id to analysis_tables
        experiment_data.analysis_tables["executor_1"] = dataset

        # Check that check_hash returns True for existing executor_id
        self.assertTrue(
            experiment_data.check_hash("executor_1", ExperimentDataEnum.analysis_tables)
        )
        self.assertTrue(experiment_data.check_hash("executor_1", "unknown"))

        # Check that check_hash returns False for non-existent executor_id
        self.assertFalse(
            experiment_data.check_hash(
                "nonexistent_executor", ExperimentDataEnum.analysis_tables
            )
        )

    def test_getitem_by_column_name(self):
        # Test getting item by column name
        subset = self.dataset["col1"]
        self.assertTrue(isinstance(subset, Dataset))
        self.assertIn("col1", subset.columns)

    def test_getitem_by_multiple_column_names(self):
        # Test getting item by multiple column names
        subset = self.dataset[["col1", "col2"]]
        self.assertTrue(isinstance(subset, Dataset))
        self.assertEqual(set(subset.columns), {"col1", "col2"})

    def test_setitem_existing_column(self):
        # Test setting value for existing column
        self.dataset["col1"] = [7, 8, 9]
        self.assertEqual(self.dataset.data["col1"].tolist(), [7, 8, 9])

    def test_invalid_type_other(self):
        # Test invalid type handling
        with self.assertRaises(DataTypeError):
            self.dataset + {}

    def test_backend_type_error(self):
        # Test backend type error handling
        other_roles = {"col1": InfoRole(int), "col2": InfoRole(int)}
        other_data = pd.DataFrame({"col1": [10, 20, 30], "col2": [40, 50, 60]})
        with self.assertRaises(TypeError):
            Dataset(roles=other_roles, data=other_data, backend="unknown")

    def test_setitem_new_column(self):
        # Test setting value for new column
        new_data = [10, 11, 12]
        self.dataset["col3"] = new_data
        self.assertIn("col3", self.dataset.columns)
        self.assertEqual(self.dataset.data["col3"].tolist(), new_data)

    def test_setitem_dataset(self):
        # Test setting value with Dataset
        new_data = self.ged_dataset_copy()["col1"]
        self.dataset["col3"] = new_data
        self.assertIn("col3", self.dataset.columns)
        self.assertEqual(self.dataset.data["col3"].tolist(), self.data["col1"].tolist())

    def test_setitem_invalid_type(self):
        # Test setting invalid type
        with self.assertRaises(TypeError):
            self.dataset["col1"] = ["string", "another", "string"]

    def test_setitem_invalid_len(self):
        # Test setting invalid type
        with self.assertRaises(ValueError):
            self.dataset["col1"] = [10, 11]

    def test_setitem_with_iloc(self):
        # Test setting value with iloc
        new_data = 15
        self.dataset.iloc[1, 0] = new_data
        self.assertEqual(self.dataset.data.iloc[1, 0], new_data)

    def test_getitem_column_not_found(self):
        # Test getting non-existent column
        with self.assertRaises(KeyError):
            self.dataset["nonexistent_column"]

    def test_setitem_illegal_index(self):
        # Test setting value with illegal index
        with self.assertRaises(IndexError):
            self.dataset.iloc[1, 3] = 10

    def test_getitem_empty_dataset(self):
        # Test getting item from empty dataset
        empty_data = pd.DataFrame(columns=["col1", "col2"])
        empty_dataset = Dataset(roles=self.roles, data=empty_data)
        subset = empty_dataset["col1"]
        self.assertTrue(isinstance(subset, Dataset))
        self.assertEqual(len(subset), 0)

    def test_add_column(self):
        # Test adding column validation
        with self.assertRaises(ValueError):
            self.dataset.add_column([1, 2, 3])

        # Test basic add_column
        new_data = [7, 8, 9]
        self.dataset.add_column(new_data, {"col3": InfoRole()})
        self.assertIn("col3", self.dataset.columns)
        self.assertListEqual(self.dataset.data["col3"].tolist(), new_data)

        # Test adding column with Dataset
        self.dataset.add_column(
            Dataset({"col4": InfoRole()}, pd.DataFrame({"col4": new_data}))
        )
        self.assertIn("col4", self.dataset.columns)
        self.assertListEqual(self.dataset.data["col3"].tolist(), new_data)

        # Test adding column with numpy array
        self.dataset.add_column(np.array([13, 14, 15]), {"col6": InfoRole()})
        self.assertListEqual(self.dataset.data["col6"].tolist(), [13, 14, 15])

        # Test adding column with invalid role type
        with self.assertRaises(TypeError):
            self.dataset.add_column([1, 2, 3], {"col9": "not_a_role"})

        # Test adding column with duplicate name
        with self.assertRaises(ValueError):
            self.dataset.add_column([1, 2, 3], {"col1": InfoRole()})

    def test_agg(self):
        # Test aggregation with single function
        result = self.dataset.agg("mean")
        self.assertIsInstance(result, Dataset)

        # Test aggregation with multiple functions
        result = self.dataset.agg(["mean", "sum"])
        self.assertIn("mean", result.index)
        self.assertIn("sum", result.index)

        # Test aggregation with invalid function
        with self.assertRaises(AttributeError):
            self.dataset.agg("invalid_function")

        # Test aggregation with NaN values
        self.dataset.data.loc[0, "col1"] = None
        result = self.dataset.agg("mean")
        self.assertTrue(pd.notna(result["col1"]))

    def test_append(self):
        # Test basic append
        other_data = pd.DataFrame({"col1": [7, 8], "col2": [9, 10]})
        other_dataset = Dataset(roles=self.roles, data=other_data)
        result = self.dataset.append(other_dataset)
        self.assertEqual(len(result), len(self.dataset) + len(other_dataset))

        # Test append with different columns
        other_data = pd.DataFrame({"col1": [7], "col3": [11]})
        other_roles = {"col1": InfoRole(), "col3": InfoRole()}
        other_dataset = Dataset(roles=other_roles, data=other_data)
        result = self.dataset.append(other_dataset)
        self.assertIn("col3", result.columns)
        self.assertTrue(result.data["col2"].isna().any())

        # Edge cases
        # Test append empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = self.dataset.append(empty_dataset)
        self.assertEqual(len(result), len(self.dataset))

        # Test append to empty dataset
        result = empty_dataset.append(self.dataset)
        self.assertEqual(len(result), len(self.dataset))
        self.assertEqual(result, self.dataset)

        # Test append with completely different columns
        other_data = pd.DataFrame({"col3": [1], "col4": [2]})
        other_roles = {"col3": InfoRole(), "col4": InfoRole()}
        other_dataset = Dataset(roles=other_roles, data=other_data)
        result = self.dataset.append(other_dataset)
        self.assertTrue(result.data["col3"].isna().sum() == len(self.dataset))
        self.assertTrue(result.data["col1"].isna().sum() == 1)

        # Test append with invalid type
        with self.assertRaises(ConcatDataError):
            self.dataset.append([1, 2, 3])

    def test_apply(self):
        # Test with lambda function
        result = self.dataset.apply(lambda x: x * 2, self.roles)
        self.assertEqual(result.data["col1"].tolist(), [2, 4, 6])

        # Test with named function
        def multiply_by_three(x):
            return x * 3

        result = self.dataset.apply(multiply_by_three, self.roles)
        self.assertEqual(result.data["col1"].tolist(), [3, 6, 9])

        # Test with axis=1
        result = self.dataset.apply(
            lambda x: x["col1"] + x["col2"], {"res": InfoRole()}, axis=1
        )
        self.assertEqual(result.data["res"].tolist(), [5, 7, 9])

        # Edge cases
        # Test with function that returns None
        result = self.dataset.apply(lambda x: None, {"res": InfoRole()})
        self.assertTrue(result.data.isna().all().all())

        # Test with function that raises exception
        def failing_function(x):
            raise ValueError("Test error")

        with self.assertRaises(ValueError):
            self.dataset.apply(failing_function, {"res": InfoRole()})

        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.apply(lambda x: x * 2, {"res": InfoRole()})
        self.assertTrue(result.is_empty())

    def test_astype(self):
        # Test single column type conversion
        result = self.dataset.astype({"col1": str})
        self.assertTrue(result.data["col1"].dtype == "object")

        # Test multiple column type conversion
        result = self.dataset.astype({"col1": float, "col2": str})
        self.assertTrue(result.data["col1"].dtype == "float64")
        self.assertTrue(result.data["col2"].dtype == "object")

        # Test with errors='ignore'
        invalid_dataset = Dataset(
            roles={"col1": InfoRole()}, data=pd.DataFrame({"col1": ["a", "b", "c"]})
        )
        result = invalid_dataset.astype({"col1": int}, errors="ignore")
        self.assertTrue(result.data["col1"].dtype == "object")

        invalid_dataset = Dataset(
            roles={"col1": InfoRole()}, data=pd.DataFrame({"col1": ["a", "b", "c"]})
        )
        result = invalid_dataset.astype({"col1": str}, errors="ignore")
        self.assertTrue(result.data["col1"].dtype == "object")

        # Edge cases
        # Test with non-existent column
        with self.assertRaises(KeyError):
            self.dataset.astype({"non_existent": int})

        # Test with invalid dtype
        with self.assertRaises(TypeError):
            self.dataset.astype({"col1": "invalid_dtype"})

        # Test with mixed types
        self.dataset.data["col1"] = [1, "two", 3]
        with self.assertRaises(ValueError):
            self.dataset.astype({"col1": int})

        # Test with NaN values
        self.dataset.data["col1"] = [1, None, 3]
        result = self.dataset.astype({"col1": float})
        self.assertTrue(pd.isna(result.data["col1"][1]))

    def test_coefficient_of_variation(self):
        # Test with correct positive values
        self.dataset.data["col1"] = [1, 2, 3]
        cv = self.dataset.coefficient_of_variation()

        # Calculate the expected CV manually
        mean_col1 = np.mean([1, 2, 3])
        std_col1 = np.std(
            [1, 2, 3], ddof=0
        )  # Standard deviation with population correction
        expected_cv = std_col1 / mean_col1

        # Check that the coefficient of variation is close to the expected value
        self.assertAlmostEqual(cv["col1"], expected_cv, places=3)

        # Test with negative values
        self.dataset.data["col1"] = [-1, -2, -3]
        cv = self.dataset.coefficient_of_variation()
        self.assertTrue(cv["col1"] < 0)

        # Test with NaN values
        self.dataset.data["col1"] = [1, None, 3]
        cv = self.dataset.coefficient_of_variation()
        self.assertTrue(pd.notna(cv["col1"]))

    def test_corr(self):
        # Test Pearson correlation
        corr = self.dataset.corr(method="pearson")
        self.assertIsInstance(corr, Dataset)
        self.assertEqual(corr.shape, (2, 2))

        # Test with NaN values
        self.dataset.data["col1"] = [1, None, 3]
        corr = self.dataset.corr()
        self.assertTrue(pd.notna(corr.loc["col1", "col2"]))

        # Test with invalid method
        with self.assertRaises(ValueError):
            self.dataset.corr(method="invalid_method")

    def test_count(self):
        # Test basic count
        counts = self.dataset.count()
        self.assertEqual(counts["col1"], 3)
        self.assertEqual(counts["col2"], 3)

        # Test count with NaN values
        self.dataset.data.loc[0, "col1"] = None
        counts = self.dataset.count()
        self.assertEqual(counts["col1"], 2)
        self.assertEqual(counts["col2"], 3)

        # Edge cases
        # Test with all NaN values
        self.dataset.data["col1"] = None
        counts = self.dataset.count()
        self.assertEqual(counts["col1"], 0)

        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        counts = empty_dataset.count()
        self.assertTrue(counts.data.empty)
        self.assertTrue(all(counts.columns == empty_dataset.columns))
        self.assertTrue(all(counts.index == empty_dataset.index))

        # Test with mixed types
        self.dataset.data["col1"] = [1, "two", None]
        counts = self.dataset.count()
        self.assertEqual(counts["col1"], 2)

    def test_cov(self):
        # Test basic covariance
        cov = self.dataset.cov()
        self.assertIsInstance(cov, Dataset)
        self.assertEqual(cov.shape, (2, 2))

        expected_cov = self.dataset.data.cov()
        pd.testing.assert_frame_equal(cov.data, expected_cov)

        # Edge cases
        # Test with constant column
        self.dataset.data["col1"] = [1, 1, 1]
        cov = self.dataset.cov()
        self.assertEqual(
            cov.loc["col1", "col1"], 0
        )  # Covariance with constant column should be 0

        # Test with NaN values
        self.dataset.data["col1"] = [1, None, 3]
        cov = self.dataset.cov()

        # The expected behavior for NaN values: pandas should handle them by excluding those rows in the covariance calculation
        expected_cov = self.dataset.data.cov()
        pd.testing.assert_frame_equal(
            cov.data, expected_cov
        )  # Ensure covariance matrix is correct even with NaN

        # Ensure covariance involving NaN is correctly handled (NaN should not propagate if handled by pandas)
        self.assertTrue(pd.notna(cov.loc["col1", "col2"]))

    def test_create_empty_2(self):
        # Test basic empty creation
        empty = Dataset.create_empty(roles=self.roles)
        self.assertTrue(empty.is_empty())
        self.assertEqual(list(empty.columns), ["col1", "col2"])

        # Test with index
        empty = Dataset.create_empty(roles=self.roles, index=[0, 1, 2])
        self.assertFalse(empty.is_empty())
        self.assertEqual(len(empty.index), 3)

        # Edge cases
        # Test with empty roles
        empty = Dataset.create_empty(roles={})
        self.assertTrue(empty.is_empty())
        self.assertEqual(len(empty.index), 0)

        # Test with invalid role type
        with self.assertRaises(TypeError):
            Dataset.create_empty(roles={"col1": "not_a_role"})

        # TODO:check if we need this test
        # # Test with duplicate column names
        # with self.assertRaises(ValueError):
        #     Dataset.create_empty(roles={"col1": InfoRole(), "col1": InfoRole()})

        # Test with invalid index
        with self.assertRaises(TypeError):
            Dataset.create_empty(roles=self.roles, index="invalid")

    def test_dot(self):
        # Test with Dataset
        other_data = pd.DataFrame({"a": [1, 2], "b": [2, 3], "c": [3, 4]})
        other_roles = {"a": InfoRole(), "b": InfoRole(), "c": InfoRole()}
        other_data.index = ["col1", "col2"]
        other = Dataset(data=other_data, roles=other_roles)
        result = self.dataset.dot(other)
        self.assertIsInstance(result, Dataset)

        # Edge cases
        # Test with mismatched dimensions
        other_data = pd.DataFrame({"a": [1, 2, 3, 4]})
        other_roles = {"a": InfoRole()}
        other = Dataset(data=other_data, roles=other_roles)
        other.index = ["col1", "col2", "col3", "col4"]
        with self.assertRaises(ValueError):
            self.dataset.dot(other)

        # Test with NaN values
        self.dataset.data.loc[0, "col1"] = None
        other_data = pd.DataFrame(np.array([1, 2]))
        other_roles = {}
        other = Dataset(data=other_data, roles=other_roles)
        other.index = ["col1", "col2"]
        result = self.dataset.dot(other)
        self.assertTrue(np.isnan(result.iget_values(0, 0)))

        # Test with non-numeric data
        self.dataset.data["col1"] = ["a", "b", "c"]
        other_data = pd.DataFrame({"a": [1, 2]})
        other_roles = {"a": InfoRole()}
        other = Dataset(data=other_data, roles=other_roles)
        with self.assertRaises(ValueError):
            self.dataset.dot(other)

        # Test with empty dataset
        empty_dataset = Dataset.create_empty()
        with self.assertRaises(ValueError):
            empty_dataset.dot(self.dataset)

    def test_drop(self):
        # Test drop single column
        result = self.dataset.drop(["col1"])
        self.assertNotIn("col1", result.columns)
        self.assertIn("col2", result.columns)

        # Test drop multiple columns
        result = self.dataset.drop(["col1", "col2"])
        self.assertEqual(len(result.columns), 0)

        # Test drop with axis=0 (rows)
        result = self.dataset.drop([0, 1], axis=0)
        self.assertEqual(len(result), 1)

        # Edge cases
        # Test drop non-existent column
        with self.assertRaises(KeyError):
            self.dataset.drop(["non_existent"])

        # Test drop all columns
        result = self.dataset.drop(self.dataset.columns)
        self.assertEqual(len(result.columns), 0)

        # Test drop with empty list
        result = self.dataset.drop([])
        self.assertEqual(len(result.columns), len(self.dataset.columns))

        # Test drop with invalid axis
        with self.assertRaises(ValueError):
            self.dataset.drop(["col1"], axis=2)

    def test_dropna_rows(self):
        # Test dropping NA values
        data_with_na = pd.DataFrame({"col1": [1, None, 3], "col2": [4, 5, 6]})
        dataset_with_na = Dataset(roles=self.roles, data=data_with_na)

        dropped_dataset = dataset_with_na.dropna()
        self.assertEqual(len(dropped_dataset), 2)

    def test_dropna_columns(self):
        # Test dropping NA values
        data_with_na = pd.DataFrame({"col1": [1, None, 3], "col2": [4, 5, 6]})
        dataset_with_na = Dataset(roles=self.roles, data=data_with_na)

        dropped_dataset = dataset_with_na.dropna(axis="columns")
        self.assertEqual(dropped_dataset.columns, "col2")

    def test_dropna_all(self):

        # Test with how='all'
        self.dataset.data.loc[0] = [None, None]
        result = self.dataset.dropna(how="all")
        self.assertEqual(len(result), 2)

        # Edge cases
        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.dropna()
        self.assertEqual(len(result), 0)

        # Test with no NaN values
        self.dataset.data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        result = self.dataset.dropna()
        self.assertEqual(len(result), 3)

        # Test with invalid how parameter
        with self.assertRaises(ValueError):
            self.dataset.dropna(how="invalid")

    def test_fillna(self):
        # Test fill with single value
        self.dataset.data.loc[0, "col1"] = None
        result = self.dataset.fillna(0)
        self.assertEqual(result.data.loc[0, "col1"], 0)

        # Test fill with dict
        self.dataset.data.loc[0, ["col1", "col2"]] = None
        result = self.dataset.fillna({"col1": 1, "col2": 2})
        self.assertEqual(result.data.loc[0, "col1"], 1)
        self.assertEqual(result.data.loc[0, "col2"], 2)

        # Edge cases
        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.fillna(0)
        self.assertFalse(result.data.isna().any().any())

        # Test with no NaN values
        self.dataset.data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        result = self.dataset.fillna(0)
        self.assertTrue((result.data == self.dataset.data).all().all())

    def test_fillna_with_str(self):
        # Test filling NA values with a string
        data_with_na = pd.DataFrame({"col1": ["uwu", None, ":)"], "col2": [4, 5, 6]})
        roles = {"col1": InfoRole(str), "col2": InfoRole()}
        dataset_with_na = Dataset(roles=roles, data=data_with_na)

        # Fill NaN values in 'col1' with the string 'missing'
        filled_dataset = dataset_with_na.fillna({"col1": "missing"})
        self.assertEqual(filled_dataset.data.loc[1, "col1"], "missing")

    def test_filter(self):
        # Test with items
        result = self.dataset.filter(items=["col1"])
        self.assertEqual(list(result.columns), ["col1"])

        # Test with regex
        result = self.dataset.filter(regex="1$")
        self.assertEqual(list(result.columns), ["col1"])

        # Test with like
        result = self.dataset.filter(like="col")
        self.assertEqual(len(result.columns), 2)

        # Edge cases
        # Test with non-existent items
        result = self.dataset.filter(items=["non_existent"])
        self.assertEqual(len(result.columns), 0)

        # Test with no matches in regex
        result = self.dataset.filter(regex="xyz")
        self.assertEqual(len(result.columns), 0)

        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.filter(like="col")
        self.assertEqual(len(result.columns), 2)

        # Test with multiple filter criteria
        with self.assertRaises(TypeError):
            self.dataset.filter(items=["col1"], regex="1$")

    def test_from_dict(self):
        # Test with dict of lists
        data_dict = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
        result = Dataset.from_dict(data_dict, self.roles)
        self.assertEqual(len(result), 3)

        # Test with dict of values
        data_dict = {"col1": [1], "col2": [2]}
        result = Dataset.from_dict(data_dict, self.roles)
        self.assertEqual(len(result), 1)

        # Edge cases
        # Test with empty dict
        empty_dataset = Dataset.from_dict({}, self.roles)
        self.assertTrue(empty_dataset.is_empty())

        # Test with missing columns
        data_dict = {"col1": [1, 2, 3]}  # Missing col2
        dataset = Dataset.from_dict(data_dict, self.roles)
        self.assertEqual(dataset.shape, (3, 1))

        data_list = [
            {"col1": 1, "col2": 4},
            {"col1": 2, "col2": 5},
            {"col1": 3, "col2": 6},
        ]
        dataset_from_list = Dataset.from_dict(data_list, self.roles)
        self.assertEqual(dataset_from_list.shape, (3, 2))

    def test_groupby(self):

        data_grouped = pd.DataFrame({"col1": [1, 2, 1], "col2": [4, 5, 6]})
        dataset_grouped = Dataset(roles=self.roles, data=data_grouped)

        grouped = dataset_grouped.groupby(by="col1")
        self.assertEqual(len(grouped), 2)

    def test_groupby_with_single_column_dataset(self):
        by = Dataset(
            roles={"col1": InfoRole(int)}, data=pd.DataFrame({"col1": [1, 2, 1]})
        )

        result = self.dataset.groupby(by=by)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0][1], Dataset)

    def test_groupby_with_column_name(self):
        result = self.dataset.groupby(by="col1")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)  # There should be 3 groups for values 1, 2, and 3.
        self.assertIsInstance(result[0][1], Dataset)

    def test_groupby_with_func(self):
        # Aggregation with sum function
        result = self.dataset.groupby(by="col1", func="sum")
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)  # Проверка числа групп
        self.assertIsInstance(result[0][1], Dataset)

    def test_groupby_with_fields_list(self):
        # Column selection after grouping
        result = self.dataset.groupby(by="col1", fields_list=["col2"])
        self.assertIsInstance(result, list)
        self.assertEqual(result[0][1].data.columns.tolist(), ["col2"])

    def test_groupby_with_invalid_by_type(self):
        # Check for incorrect type for 'by'
        with self.assertRaises(KeyError):
            self.dataset.groupby(by=123)

    def test_groupby_with_empty_dataset(self):
        # Check for empty Dataset
        empty_dataset = Dataset(
            roles=self.roles, data=pd.DataFrame(columns=["col1", "col2"])
        )
        result = empty_dataset.groupby(by="col1")
        self.assertEqual(result, [])

    def test_groupby_with_tmp_roles(self):
        # Check that tmp_roles are saved after grouping
        result = self.dataset.groupby(by="col1")
        self.assertTrue(
            all(dataset[1].tmp_roles == self.dataset.tmp_roles for dataset in result)
        )

    def test_groupby_with_fields_list_and_func(self):
        # Grouping with fields and function
        result = self.dataset.groupby(by="col1", fields_list=["col2"], func="sum")
        self.assertIsInstance(result, list)
        self.assertEqual([int(x[1]) for x in result], self.data["col2"].tolist())

    def test_idxmax(self):
        # Test basic idxmax
        result = self.dataset.idxmax()
        self.assertEqual(result["col1"], 2)  # Index of max value (3)

        # Edge cases
        # Test with all equal values
        self.dataset.data["col1"] = [1, 1, 1]
        result = self.dataset.idxmax()
        self.assertEqual(result["col1"], 0)  # Returns first occurrence

    def test_is_empty(self):
        # Test non-empty dataset
        self.assertFalse(self.dataset.is_empty())

        # Test empty dataset
        empty_dataset = Dataset.create_empty(roles=self.roles)
        self.assertTrue(empty_dataset.is_empty())

        # Test dataset with empty DataFrame
        empty_df = pd.DataFrame(columns=["col1", "col2"])
        empty_dataset = Dataset(roles=self.roles, data=empty_df)
        self.assertTrue(empty_dataset.is_empty())

        # Edge cases
        # Test with NaN values
        self.dataset.data[:] = None
        self.assertFalse(self.dataset.is_empty())

        # Test with zero rows but with columns
        empty_df = pd.DataFrame(columns=["col1", "col2"])
        empty_dataset = Dataset(roles=self.roles, data=empty_df)
        self.assertTrue(empty_dataset.is_empty())

        # Test with one empty column
        self.dataset.data = pd.DataFrame({"col1": [], "col2": []})
        self.assertTrue(self.dataset.is_empty())

    def test_isna(self):
        # Test with no NaN values
        result = self.dataset.isna()
        self.assertFalse(result.data.any().any())

        # Test with NaN values
        self.dataset.data.loc[0, "col1"] = None
        result = self.dataset.isna()
        self.assertTrue(result.data.loc[0, "col1"])
        self.assertFalse(result.data.loc[0, "col2"])

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
        self.dataset.data["col1"] = [1, "two", None]
        result = self.dataset.isna()
        self.assertTrue(result.data.loc[2, "col1"])

    def test_isin(self):
        result = self._extracted_from_test_isin_3(4, 0, "col2")
        # Test with dict
        result = self.dataset.isin({"col1": [1, 2], "col2": [4]})
        self.assertTrue(result.data.loc[0, "col1"])
        self.assertTrue(result.data.loc[0, "col2"])

        # Edge cases
        # Test with empty values
        result = self.dataset.isin([])
        self.assertFalse(result.data.any().any())

        # Test with None values
        result = self.dataset.isin([None])
        self.assertFalse(result.data.any().any())

        # Test with mixed types
        self.dataset.data["col1"] = [1, "two", 3]
        result = self._extracted_from_test_isin_3("two", 1, "col1")

    # TODO Rename this here and in `test_isin`
    def _extracted_from_test_isin_3(self, arg0, arg1, arg2):
        # Test with list
        result = self.dataset.isin([1, arg0])
        self.assertTrue(result.data.loc[0, "col1"])
        self.assertTrue(result.data.loc[arg1, arg2])

        return result

    def test_log(self):
        # Test basic log
        result = self.dataset.log()
        self.assertTrue(all(result.data["col1"] >= 0))

        # Test with negative values
        self.dataset.data.loc[0, "col1"] = -1
        result = self.dataset.log()
        self.assertTrue(pd.isna(result.data.loc[0, "col1"]))

        # Test with NaN values
        self.dataset.data.loc[0, "col1"] = None
        result = self.dataset.log()
        self.assertTrue(pd.isna(result.data.loc[0, "col1"]))

        # Test with very large values
        self.dataset.data.loc[0, "col1"] = 1e308
        result = self.dataset.log()
        self.assertTrue(np.isfinite(result.data.loc[0, "col1"]))

    def test_map(self):
        # Test with function
        result = self.dataset.map(lambda x: x * 2)
        self.assertEqual(result.data["col1"].tolist(), [2, 4, 6])

    def test_max(self):
        # Test basic max
        result = self.dataset.max()
        self.assertEqual(result["col1"], 3)

        # Test with mixed types
        self.dataset.data["col1"] = [1, "two", 3]
        with self.assertRaises(TypeError):
            result = self.dataset.max()

    def test_min(self):
        # Test basic min
        result = self.dataset.min()
        self.assertEqual(result["col1"], 1)

        # Test with mixed types
        self.dataset.data["col1"] = [1, "two", 3]
        with self.assertRaises(TypeError):
            result = self.dataset.min()

    def test_mode(self):
        # Test basic mode
        result = self.dataset.mode()
        self.assertIsInstance(result, Dataset)

        # Test with duplicate values
        self.dataset.data.loc[0, "col1"] = 2
        result = self.dataset.mode()
        self.assertEqual(result.data["col1"][0], 2)

        # Test with numeric_only=True
        self.dataset.add_column(["A", "A", "B"], {"group": InfoRole()})
        result = self.dataset.mode(numeric_only=True)
        self.assertNotIn("group", result.columns)

        # Edge cases
        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.mode()
        self.assertTrue(result.data.empty)

        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.mode()
        self.assertTrue(result.data.empty)

    def test_na_counts(self):
        # Test with no NaN values
        result = self.dataset.na_counts()
        self.assertEqual(result["col1"], 0)

        # Test with NaN values
        self.dataset.data.loc[0, "col1"] = None
        self.dataset.data.loc[1, "col1"] = None
        result = self.dataset.na_counts()
        self.assertEqual(result["col1"], 2)

        # Test with different types of missing values
        self.dataset.data.loc[2, "col1"] = np.nan
        result = self.dataset.na_counts()
        self.assertEqual(result["col1"], 3)

        # Edge cases
        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.na_counts()
        self.assertEqual(result["col1"], len(self.dataset.data))

        # Test with mixed types including strings
        dataset = Dataset(
            roles=self.roles,
            data=pd.DataFrame(
                {"col1": [4, 5, 6, 1], "col2": [None, "NA", np.nan, pd.NA]}
            ),
        )
        result = dataset.na_counts()
        self.assertEqual(result["col1"], 3)  # 'NA' string is not counted as NaN

    def test_nunique(self):
        # Test basic nunique
        result = self.dataset.nunique()
        self.assertEqual(result["col1"], 3)

        # Test with duplicate values
        self.dataset.data.loc[0, "col1"] = 2
        result = self.dataset.nunique()
        self.assertEqual(result["col1"], 2)

        # Edge cases
        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.nunique()
        self.assertEqual(result["col1"], 0)

        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.nunique()
        self.assertEqual(result["col1"], 0)

        # Test with mixed types
        self.dataset.data["col1"] = [1, "two", 3]
        result = self.dataset.nunique()
        self.assertEqual(result["col1"], 3)

    def test_quantile(self):
        # Test single quantile
        result = self.dataset.quantile(0.5)
        self.assertEqual(result["col1"], 2)

        # Test multiple quantiles
        result = self.dataset.quantile([0.25, 0.75])
        self.assertEqual(len(result), 2)
        self.assertTrue(isinstance(result, Dataset))

        # Edge cases
        # Test with invalid quantile values
        with self.assertRaises(ValueError):
            result = self.dataset.quantile(1.5)

    def test_reindex(self):
        # Test columns reindex

        self.data = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": [4, 5, 6]}, index=["one", "two", "three"]
        )
        self.dataset = Dataset(roles=self.roles, data=self.data)

        # Test index reindex
        result = self.dataset.reindex([0, 1, 2, 4])
        self.assertEqual(len(result), 4)

        # Test with fill_value
        result = self.dataset.reindex([0, 1, 2, 3], fill_value=0)
        self.assertEqual(result.data.loc[3, "col1"], 0)

        # Edge cases
        # Test with empty index
        result = self.dataset.reindex([])
        self.assertEqual(len(result), 0)

        # Test with duplicate indices
        result = self.dataset.reindex([0, 0, 1])
        self.assertEqual(len(result), 3)

    def test_rename(self):
        # Test with dict
        result = self.dataset.rename({"col1": "new_col1"})
        self.assertIn("new_col1", result.columns)

        # Edge cases
        # Test with empty mapping
        result = self.dataset.rename({})
        self.assertEqual(list(result.columns), list(self.dataset.columns))

        # Test with non-existent columns
        result = self.dataset.rename({"non_existent": "new_name"})
        self.assertEqual(list(result.columns), list(self.dataset.columns))

        # Test with duplicate names
        with self.assertRaises(ValueError):
            result = self.dataset.rename({"col1": "col2"})

    def test_replace(self):
        # Test replace single value
        result = self.dataset.replace(1, 100)
        self.assertEqual(result.data.loc[0, "col1"], 100)

        # Test replace with dict
        result = self.dataset.replace({1: 100, 2: 200})
        self.assertEqual(result.data.loc[0, "col1"], 100)
        self.assertEqual(result.data.loc[1, "col1"], 200)

        # Edge cases
        # Test replace with empty dict
        result = self.dataset.replace({})
        self.assertTrue(result.data.equals(self.dataset.data))

        # Test replace with None values
        result = self.dataset.replace({1: None})
        self.assertTrue(pd.isna(result.data.loc[0, "col1"]))

        # Test replace with non-existent values
        result = self.dataset.replace({999: 1000})
        self.assertTrue(result.data.equals(self.dataset.data))

    def test_replace_wrong_type(self):
        # Test replace single value
        with self.assertRaises(ValueError):
            self.dataset.replace(1, "uwu")

    def test_sample(self):
        # Test with n parameter
        result = self.dataset.sample(n=2, random_state=42)
        self.assertEqual(len(result), 2)

        # Test with frac parameter
        result = self.dataset.sample(frac=0.5, random_state=42)
        self.assertEqual(len(result), 2)

        # Edge cases
        # Test with n=0
        result = self.dataset.sample(n=0)
        self.assertEqual(len(result), 0)

        # Test with frac=0
        result = self.dataset.sample(frac=0)
        self.assertEqual(len(result), 0)

    def test_select_dtypes(self):
        # Test include
        result = self.dataset.select_dtypes(include=["int64"])
        self.assertEqual(len(result.columns), 2)

        # Test exclude
        result = self.dataset.select_dtypes(exclude=["int64"])
        self.assertEqual(len(result.columns), 0)

        # Test with multiple types
        self.dataset.data["col1"] = self.dataset.data["col1"].astype(float)
        result = self.dataset.select_dtypes(include=["int64", "float64"])
        self.assertEqual(len(result.columns), 2)

        # Edge cases
        # Test with empty include list
        with self.assertRaises(ValueError):
            result = self.dataset.select_dtypes(include=[])

        # Test with non-existent dtype
        with self.assertRaises(TypeError):
            result = self.dataset.select_dtypes(include=["non_existent_dtype"])

        # Test with mixed types
        self.dataset.data["col3"] = ["a", "b", "c"]
        result = self.dataset.select_dtypes(include=["object"])
        self.assertEqual(len(result.columns), 1)

    def test_sort(self):
        # Test basic sort
        result = self.dataset.sort(by="col1", ascending=False)
        self.assertEqual(result.data["col1"].tolist(), [3, 2, 1])

        # Test with multiple columns
        result = self.dataset.sort(by=["col1", "col2"], ascending=[False, True])
        self.assertEqual(result.data["col1"].tolist(), [3, 2, 1])

        # Test with na_position
        self.dataset.data.loc[0, "col1"] = None
        result = self.dataset.sort(by="col1", na_position="first")
        self.assertTrue(pd.isna(result.data["col1"].iloc[0]))

        # Test with non-existent column
        with self.assertRaises(KeyError):
            result = self.dataset.sort(by="non_existent")

        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.sort(by="col1")
        self.assertTrue(pd.isna(result.data["col1"]).all())

    def test_std(self):
        # Test basic std
        result = self.dataset.std()
        self.assertEqual(result.iget_values(0), [1.0, 1.0])

        # Test with ddof parameter
        self.assertEqual(
            self.dataset.std(ddof=0).iget_values(0),
            self.dataset.data.std(ddof=0).to_list(),
        )

        # Test with skipna parameter
        self.dataset.data.loc[0, "col1"] = None
        result = self.dataset.std(skipna=False)
        self.assertTrue(np.isnan(result.iget_values(0, 0)))

        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.std()
        self.assertTrue(np.isnan(result.iget_values(0, 0)))

        # Test with single value
        self.dataset.data["col1"] = [1, 1, 1]
        result = self.dataset.std()
        self.assertEqual(result["col1"], 0)

    def test_sum(self):
        # Test basic sum
        result = self.dataset.sum()
        self.assertEqual(result["col1"], 6)

        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.sum()
        self.assertEqual(result["col1"], 0)

        # Test with mixed types
        self.dataset.data["col1"] = [1, "two", 3]
        with self.assertRaises(TypeError):
            result = self.dataset.sum()

    def test_transpose(self):
        # Test basic transpose
        result = self.dataset.transpose()
        self.assertEqual(result.shape[0], 2)
        self.assertEqual(result.shape[1], 3)

        # Edge cases
        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.transpose()
        self.assertTrue(result.data.empty)

        # Test with single column
        single_col_dataset = self.dataset[["col1"]]
        result = single_col_dataset.transpose()
        self.assertEqual(result.shape[0], 1)
        self.assertEqual(result.shape[1], 3)

    def test_unique(self):
        # Test basic unique
        result = self.dataset.unique()
        self.assertEqual(len(result["col1"]), 3)

        # Test with duplicate values
        self.dataset.data.loc[0, "col1"] = 2
        result = self.dataset.unique()
        self.assertEqual(len(result["col1"]), 2)

        # Test with NaN values
        self.dataset.data.loc[0, "col1"] = None
        result = self.dataset.unique()
        self.assertTrue(np.isnan(result["col1"][0]))

        # Edge cases
        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.unique()
        self.assertEqual(len(result["col1"]), 0)

        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.unique()
        self.assertEqual(len(result["col1"]), 1)

    def test_value_counts(self):
        # Test basic value_counts
        result = self.dataset.value_counts()
        self.assertEqual(len(result["col1"]), 3)

        # Test with normalize parameter
        self.assertEqual(
            self.dataset.value_counts(normalize=True).iget_values(0, 2),
            self.dataset.data.value_counts(normalize=True).iloc[0],
        )

        # Test with dropna parameter
        self.dataset.data.loc[0, "col1"] = None
        result = self.dataset.value_counts(dropna=False)
        self.assertEqual(len(result["col1"]), 3)

        # Edge cases
        # Test with empty dataset
        empty_dataset = Dataset.create_empty(self.roles)
        result = empty_dataset.value_counts()
        self.assertTrue(result["col1"].is_empty())

        # Test with all same values
        self.dataset.data["col1"] = [1, 1, 1]
        self.dataset.data["col2"] = [1, 1, 1]
        result = self.dataset.value_counts()
        self.assertEqual(len(result["col1"]), 1)
        self.assertEqual(result["col1"].iloc[0], 3)

        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.value_counts(dropna=False)
        self.assertEqual(result["col1"].iloc[0], len(self.dataset))

    def test_var(self):
        # Test basic var
        result = {
            k: v[0] for k, v in self.dataset.var().to_dict()["data"]["data"].items()
        }
        self.assertEqual(result, self.dataset.data.var().to_dict())

        # Test with ddof parameter
        result = {
            k: v[0]
            for k, v in self.dataset.var(ddof=0).to_dict()["data"]["data"].items()
        }
        self.assertEqual(result, self.dataset.data.var(ddof=0).to_dict())

        # Test with skipna parameter
        dataset = self.ged_dataset_copy()
        dataset.data.loc[0, "col1"] = None
        result = dataset.var(skipna=False)
        self.assertTrue(np.isnan(result.iget_values(0, 0)))

        # Test with all NaN values
        self.dataset.data[:] = None
        result = self.dataset.var()
        self.assertTrue(np.isnan(result.iget_values(0, 0)))

        # Test with single value
        self.dataset.data["col1"] = [1, 1, 1]
        result = self.dataset.var()
        self.assertEqual(result["col1"], 0)

    def test_properties(self):
        # Test columns property
        self.assertEqual(list(self.dataset.columns), ["col1", "col2"])

        # Test data property
        self.assertIsInstance(self.dataset.data, pd.DataFrame)
        self.assertEqual(self.dataset.data.shape, (3, 2))

        # Test index property
        self.assertEqual(len(self.dataset.index), 3)
        self.assertTrue(all(isinstance(i, int) for i in self.dataset.index))

        # Test shape property
        self.assertEqual(self.dataset.shape, (3, 2))

    # Tests for unary operators
    def test_pos_operator(self):
        result = +self.dataset
        self.assertIsInstance(result, Dataset)  # Expecting a Dataset return
        self.assertTrue(
            (result.data >= 0).all().all()
        )  # Expecting all elements >= 0

    def test_neg_operator(self):
        result = -self.dataset
        self.assertIsInstance(result, Dataset)  # Expecting a Dataset return
        self.assertTrue((result.data < 0).all().all())  # Expecting all elements < 0

    def test_abs_operator(self):
        result = abs(self.dataset)
        self.assertIsInstance(result, Dataset)  # Expecting a Dataset return
        self.assertTrue(
            (result.data >= 0).all().all()
        )  # Expecting all elements >= 0

    def test_bool_operator(self):
        result = bool(self.dataset)
        self.assertTrue(result)  # Expecting non-empty Dataset to return True


def test_operators(self):
    test_cases = [
        ("+", pd.DataFrame({"col1": [1, 1, 1], "col2": [1, 1, 1]}), lambda x, y: x + y),
        ("-", pd.DataFrame({"col1": [1, 1, 1], "col2": [1, 1, 1]}), lambda x, y: x - y),
        ("*", pd.DataFrame({"col1": [1, 1, 1], "col2": [1, 1, 1]}), lambda x, y: x * y),
        (
            "//",
            pd.DataFrame({"col1": [1, 1, 1], "col2": [1, 1, 1]}),
            lambda x, y: x // y,
        ),
        ("/", pd.DataFrame({"col1": [1, 1, 1], "col2": [1, 1, 1]}), lambda x, y: x / y),
        ("%", pd.DataFrame({"col1": [2, 3, 4], "col2": [1, 2, 3]}), lambda x, y: x % y),
        ("**", pd.DataFrame({"col1": [2, 3, 2], "col2": [1, 2, 3]}), lambda x, y: x**y),
        ("&", pd.DataFrame({"col1": [1, 1, 0], "col2": [1, 0, 0]}), lambda x, y: x & y),
        ("|", pd.DataFrame({"col1": [1, 0, 1], "col2": [0, 1, 1]}), lambda x, y: x | y),
        ("^", pd.DataFrame({"col1": [1, 0, 1], "col2": [0, 1, 1]}), lambda x, y: x ^ y),
        ("<", pd.DataFrame({"col1": [1, 2, 3], "col2": [3, 2, 1]}), lambda x, y: x < y),
        (
            "<=",
            pd.DataFrame({"col1": [1, 2, 3], "col2": [3, 2, 1]}),
            lambda x, y: x <= y,
        ),
        (">", pd.DataFrame({"col1": [1, 2, 3], "col2": [3, 2, 1]}), lambda x, y: x > y),
        (
            ">=",
            pd.DataFrame({"col1": [1, 2, 3], "col2": [3, 2, 1]}),
            lambda x, y: x >= y,
        ),
        (
            "==",
            pd.DataFrame({"col1": [1, 2, 3], "col2": [3, 2, 1]}),
            lambda x, y: x == y,
        ),
        (
            "!=",
            pd.DataFrame({"col1": [1, 2, 3], "col2": [3, 2, 1]}),
            lambda x, y: x != y,
        ),
        # Right-hand operator cases
        ("radd", 5, lambda x, y: x + y),
        ("rsub", 10, lambda x, y: x - y),
        ("rmul", 3, lambda x, y: x * y),
        ("rfloordiv", 7, lambda x, y: x // y),
        ("rdiv", 8, lambda x, y: x / y),
        ("rtruediv", 8.0, lambda x, y: x / y),
        ("rmod", 9, lambda x, y: x % y),
        ("rpow", 2, lambda x, y: x**y),
        ("rdiv2", 10, lambda x, y: x / y),
        ("div", 2, lambda x, y: x / y),
    ]

    for operator, other_data, expected_operator in test_cases:
        with self.subTest(operator=operator):
            # Create the other dataset
            other_dataset = Dataset(
                roles=self.roles, data=other_data, backend=BackendsEnum.pandas
            )
            # Perform the operation using the operator directly
            operator_functions = {
                "+": lambda self, other: self.dataset + other,
                "-": lambda self, other: self.dataset - other,
                "*": lambda self, other: self.dataset * other,
                "//": lambda self, other: self.dataset // other,
                "/": lambda self, other: self.dataset / other,
                "%": lambda self, other: self.dataset % other,
                "**": lambda self, other: self.dataset ** other,
                "&": lambda self, other: self.dataset & other,
                "|": lambda self, other: self.dataset | other,
                "^": lambda self, other: self.dataset ^ other,
                "<": lambda self, other: self.dataset < other,
                "<=": lambda self, other: self.dataset <= other,
                ">": lambda self, other: self.dataset > other,
                ">=": lambda self, other: self.dataset >= other,
                "==": lambda self, other: self.dataset == other,
                "!=": lambda self, other: self.dataset != other,
                "radd": lambda self, other: other + self.dataset,
                "rsub": lambda self, other: other - self.dataset,
                "rmul": lambda self, other: other * self.dataset,
                "rfloordiv": lambda self, other: other // self.dataset,
                "rdiv": lambda self, other: other / self.dataset,
                "rtruediv": lambda self, other: other / self.dataset,
                "rmod": lambda self, other: other % self.dataset,
                "rpow": lambda self, other: other ** self.dataset,
                "rdiv2": lambda self, other: other / self.dataset
            }

            operator = operator  # Assuming operator is defined somewhere in the code
            result = operator_functions.get(operator, lambda self, other: other)(self, other_dataset)

            # Check the result type
            self.assertIsInstance(
                result, Dataset, f"Expected result to be Dataset for {operator}"
            )

            # Check the operation result
            expected_data = (
                expected_operator(other_data, self.data)
                if operator.isalpha()
                else expected_operator(self.data, other_data)
            )
            pd.testing.assert_frame_equal(result.data, expected_data)

    def test_locker_getitem(self):
        # Using .loc (e.g., for the first row)
        self.data.index = ["a", "b", "c"]
        self.dataset = Dataset(roles=self.roles, data=self.data)
        t_data = self.dataset.loc["a"]
        self.assertTrue(isinstance(t_data, Dataset))
        self.assertEqual(t_data.loc["col1"], 1)
        self.assertEqual(t_data.loc["col2"], 4)

        with self.assertRaises(KeyError):
            self.dataset["q"]

        with self.assertRaises(RoleColumnError):
            self.dataset[1]

        # Testing roles
        self.assertIn("a", t_data.roles)

    def test_locker_setitem_valid(self):
        # Correct update
        self.data.index = ["a", "b", "c"]
        self.dataset = Dataset(roles=self.roles, data=self.data)
        self.dataset.loc["a", "col1"] = [10]
        self.assertEqual(self.dataset.data.loc["a", "col1"], 10)

        self.dataset.loc["q", "col1"] = [10]
        self.assertEqual(self.dataset.data.loc["q", "col1"], 10)

        self.dataset.loc[1, "col1"] = [10]
        self.assertEqual(self.dataset.data.loc[1, "col1"], 10)

    def test_locker_setitem_invalid_type(self):
        # Wrong data type
        self.data.index = ["a", "b", "c"]
        self.dataset = Dataset(roles=self.roles, data=self.data)
        with self.assertRaises(TypeError):
            self.dataset.loc["a", "col1"] = ["string"]

    def test_locker_setitem_type_mismatch(self):
        # Attempting to update a column with mismatched data type
        self.data.index = ["a", "b", "c"]
        self.dataset = Dataset(roles=self.roles, data=self.data)
        with self.assertRaises(TypeError):
            self.dataset.loc["a", "col2"] = [1.1]

    def test_ilocker_getitem(self):
        t_data = self.dataset.iloc[0]
        self.assertTrue(isinstance(t_data, Dataset))
        self.assertEqual(t_data.data.to_dict(), {0: {"col1": 1, "col2": 4}})

        with self.assertRaises(IndexError):
            self.dataset.iloc[7]

        with self.assertRaises(TypeError):
            self.dataset.iloc["a"]

        # Testing roles
        self.assertIn(0, t_data.roles)

    def test_ilocker_setitem_valid(self):
        # Correct update via iloc
        self.dataset.iloc[0, 0] = 10
        self.assertEqual(self.dataset.data.iloc[0, 0], 10)

        with self.assertRaises(IndexError):
            self.dataset.iloc["q", "col1"] = [10]

        with self.assertRaises(IndexError):
            self.dataset.iloc[7, "col1"] = [10]

    def test_ilocker_setitem_invalid_type(self):
        # Wrong data type via iloc
        with self.assertRaises(TypeError):
            self.dataset.iloc[0, 0] = "string"

    def test_ilocker_setitem_column_not_found(self):
        # Attempting to update a non-existent column via iloc
        with self.assertRaises(IndexError):
            self.dataset.iloc[0, 5] = 10

    def test_ilocker_setitem_type_mismatch(self):
        # Attempting to update a column with mismatched data type via iloc
        with self.assertRaises(TypeError):
            self.dataset.iloc[0, 1] = [1.1]

    def test_init(self):
        # Checking that the class initializes correctly
        self.experiment_data = ExperimentData(self.dataset)
        self.assertEqual(self.experiment_data._data.data["col1"].tolist(), [1, 2, 3])
        self.assertEqual(self.experiment_data._data.data["col2"].tolist(), [4, 5, 6])
        self.assertIsInstance(self.experiment_data.additional_fields, Dataset)


if __name__ == "__main__":
    unittest.main()
