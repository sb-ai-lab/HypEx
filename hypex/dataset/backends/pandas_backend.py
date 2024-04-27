from pathlib import Path
from typing import (
    Sequence,
    Union,
    Iterable,
    List,
    Dict,
    Tuple,
    Callable,
    Optional,
    Any,
)

import pandas as pd  # type: ignore

from hypex.dataset.abstract import DatasetBackend
from hypex.utils import FromDictType
from hypex.utils.decorators import inherit_docstring_from


class PandasDataset(DatasetBackend):
    """HypEx realization of pandas Dataset.
    Provides a Pandas DataFrame-based backend for handling dataset operations, encapsulating
    functionalities for data manipulation and analysis.

    Methods include file reading, data appending, transformation, aggregation, and column management,
    designed to abstract complex DataFrame operations into simpler high-level API calls.
    """

    @staticmethod
    def _read_file(filename: Union[str, Path]) -> pd.DataFrame:
        """Reads data from a file and returns a DataFrame.

        Supports CSV and Excel formats. Raises an error for unsupported file types.

        Args:
            filename (Union[str, Path]): The path to the file to be read.

        Returns:
            pd.DataFrame: The data read from the file.

        Raises:
            ValueError: If the file extension is neither '.csv' nor '.xlsx'.
        """
        file_extension = Path(filename).suffix
        if file_extension == ".csv":
            return pd.read_csv(filename)
        elif file_extension == ".xlsx":
            return pd.read_excel(filename)
        else:
            raise ValueError(f"Unsupported file extension {file_extension}")

    def __init__(self, data: Union[pd.DataFrame, Dict, str, pd.Series] = None):
        """Initializes the PandasDataset with data provided in various forms.

        Args:
            data (Union[pd.DataFrame, Dict, str, pd.Series], optional): Initial data. Can be a DataFrame,
                a dictionary with data and index keys, a path to a file, or a Pandas Series. Defaults to None.

        If a string is provided, it is assumed to be a file path which will be read into the DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = pd.DataFrame(data)
        elif isinstance(data, Dict):
            self.data = pd.DataFrame(data=data.get("data"), index=data.get("index", None))
        elif isinstance(data, str):
            self.data = self._read_file(data)
        else:
            # TODO: Maybe here better create empty dataset?
            self.data = None

    def __getitem__(self, item: Union[slice, int, str, List[str]]) -> pd.DataFrame:
        # TODO: Maybe it return not the pd.DataFrame, please check
        # TODO: Also for item maybe should create special typing if it used not only here
        """Allows indexed access to the dataset using both positional and label indexing.

        This method supports slicing, integer-location based indexing (iloc) for positional access,
        and label-location based indexing (loc) for accessing by DataFrame labels.

        Args:
            item (Union[slice, int, str, List[str]]): The indexer which can be a slice for range selection,
                                                      an integer for specific positional access, a string for
                                                      accessing a column by label, or a list of strings for
                                                      accessing multiple columns.

        Returns:
            pd.DataFrame: A DataFrame containing the accessed rows or columns.

        Raises:
            KeyError: If the indexer is not one of the expected types or no matching column/row is found.
        """
        if isinstance(item, (slice, int)):
            return self.data.iloc[item]
        if isinstance(item, (str, list)):
            return self.data.loc[:, item]
        raise KeyError("No such column or row")

    @inherit_docstring_from(pd.DataFrame.__len__)
    def __len__(self) -> int:
        return self.data.__len__()

    @inherit_docstring_from(pd.DataFrame.__repr__)
    def __repr__(self):
        return self.data.__repr__()

    def _create_empty(self, index: Optional[Iterable] = None,
                      columns: Optional[Iterable[str]] = None) -> 'PandasDataset':
        """
        Resets the internal DataFrame to an empty DataFrame, optionally setting specific indices and columns.

        This method is useful for clearing existing data and reinitializing the dataset with a new structure,
        defined by the index and columns parameters. It returns the dataset instance itself, allowing for method chaining.

        Args:
            index (Optional[Iterable], optional): An iterable of index labels to include in the new DataFrame.
                                                  Defaults to None, which initializes an empty index.
            columns (Optional[Iterable[str]], optional): An iterable of column names to include in the new DataFrame.
                                                         Defaults to None, which initializes without any columns.

        Returns:
            PandasDataset: Returns the dataset instance itself with its data attribute reset to the new empty DataFrame.
        """
        self.data = pd.DataFrame(index=index, columns=columns)
        return self

    def _get_column_index(
            self, column_name: Union[Sequence[str], str]
    ) -> Union[int, Sequence[int]]:
        # TODO: Couldn't understand shoud it work only for one column or for several
        # If several, why we return the '0' element? So it should returned int, not Sequence[int]
        # Will write my variant below, please check
        return (
            self.data.columns.get_loc(column_name)
            if isinstance(column_name, str)
            else self.data.columns.get_indexer(column_name)
        )[0]

    def _get_column_index(self, column_name: Union[str, Sequence[str]]) -> int:
        """
        Retrieves the positional index of a given column name or the first index of column names in the DataFrame.

        This method supports both single string column names and sequences of column names, returning the index of the
        first column name in the sequence. It simplifies the process of column indexing in pandas, especially when
        only the first column's index is needed from a potentially larger set.

        Args:
            column_name (Union[str, Sequence[str]]): A single column name or a sequence of column names.

        Returns:
            int: The index of the specified column if a single name is provided, or the index of the first column
                 in a sequence of names.

        Raises:
            KeyError: If the specified column name or any name in the sequence does not exist in the DataFrame's columns
        """
        if isinstance(column_name, str):
            return self.data.columns.get_loc(column_name)
        else:
            indices = self.data.columns.get_indexer(column_name)
            if -1 in indices:
                raise KeyError(f"One or more column names are not found in the DataFrame. {column_name}")
            return indices[0]

    def _get_column_type(self, column_name: str) -> str:
        """Retrieves the data type of specified column in the DataFrame as a string.

        This method provides a simple way to access the data type of column, which can be useful for
        data validation, processing, or logging purposes. The data type is returned as a string, which
        is derived from the pandas dtype of the column.

        Args:
            column_name (str): The name of the column for which the data type is requested.

        Returns:
            str: The data type of the column, formatted as a string.
        """
        return str(self.data.dtypes[column_name])

    def _update_column_type(self, column_name: str, type_name: str) -> 'PandasDataset':
        """Updates the data type of specified column in the DataFrame.

        This method changes the data type of column to the specified new type. This can be useful for data preparation
        and ensuring compatibility of columns for further operations such as merges, analysis, and exports.

        Args:
            column_name (str): The name of the column whose data type needs to be updated.
            type_name (str): The new data type to which the column should be converted. This should be a valid
                             numpy or pandas data type string (e.g., 'float', 'int', 'str').

        Returns:
            PandasDataset: Returns the dataset instance itself, allowing for method chaining.
        """
        self.data[column_name] = self.data[column_name].astype(type_name)
        return self

    def add_column(self, data: Sequence, name: str, index: Optional[Sequence] = None) -> 'PandasDataset':
        """
        Adds a new column to the DataFrame with optional custom indexing.

        This method allows adding data as a new column to the DataFrame. If an index is provided, it ensures
        the data is aligned with the specified index. If no index is specified, the data is added directly.

        Args:
            data (Sequence): Data to be added as a column.
            name (str): Name of the new column.
            index (Optional[Sequence], optional): Sequence of index labels corresponding to the data values.
                                                  If provided, data will be aligned to these labels, filling with None
                                                  for unmatched indices.

        Returns:
            PandasDataset: Returns the dataset instance itself, allowing for method chaining.
        """
        if index is not None:
            temp_series = pd.Series(data, index=index)
            self.data.loc[:, name] = temp_series.reindex(self.data.index)
        else:
            self.data.loc[:, name] = pd.Series(data, index=self.data.index)

        return self

    def append(self, other: 'PandasDataset', index: bool = False) -> pd.DataFrame:
        """
        Appends another PandasDataset's DataFrame to this dataset's DataFrame and returns the combined DataFrame.

        This method concatenates the current dataset's DataFrame with another, optionally resetting the index
        to ensure a continuous range from 0 to n-1 in the resulting DataFrame.

        Args:
            other (PandasDataset): Another dataset whose DataFrame is to be appended to this dataset's DataFrame.
            index (bool, optional): If True, the resulting DataFrame's index will be reset to a continuous range.
                                    Defaults to False, in which case the original indices are preserved.

        Returns:
            pd.DataFrame: A new DataFrame containing the data from this dataset combined with the other dataset.
        """
        new_data = pd.concat([self.data, other.data])
        if index:
            new_data = new_data.reset_index(drop=True)
        return new_data

    @staticmethod
    def concat(datasets: List['PandasDataset'], index: bool = False) -> 'PandasDataset':
        """Additional details specific to the PandasDataset class implementation.

        - This method extends the pandas.concat functionality to work directly with a list of PandasDataset objects.
        - The `index` parameter controls whether the concatenated DataFrame should have a reset index.

        Args:
            datasets (List['PandasDataset']): A list of PandasDataset instances to concatenate.
            index (bool, optional): If True, the resulting DataFrame's index will be reset. Defaults to False.

        Returns:
            PandasDataset: A new PandasDataset instance containing the concatenated data.
        """
        dataframes = [ds.data for ds in datasets]
        concatenated_df = pd.concat(dataframes)

        if index:
            concatenated_df.reset_index(drop=True, inplace=True)

        return PandasDataset(data=concatenated_df)

    @inherit_docstring_from(pd.DataFrame.index)
    @property
    def index(self):
        return self.data.index

    @inherit_docstring_from(pd.DataFrame.columns)
    @property
    def columns(self):
        return self.data.columns

    def from_dict(self, data: FromDictType, index: Optional[Iterable] = None) -> 'PandasDataset':
        """Populates the dataset with data from a dictionary, optionally setting a custom index.

        This method creates a new DataFrame from record-style data in a dictionary. If an index is provided,
        it sets this index to the newly created DataFrame, allowing for custom index alignment.

        Args:
            data (FromDictType): Data in dictionary format to be converted into a DataFrame.
                                 The dictionary should be in the form expected by `pd.DataFrame.from_records`,
                                 which is typically a list of dicts or a dict of sequences.
            index (Optional[Iterable], optional): An iterable of index labels to set for the new DataFrame.
                                                  If provided, this will override the default index generated
                                                  by pandas. Defaults to None.

        Returns:
            PandasDataset: Returns the instance itself,
            updated with the new data and index, allowing for method chaining.
        """
        self.data = pd.DataFrame.from_records(data)
        if index is not None:
            self.data.index = index
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataset's DataFrame into a dictionary format that includes both the data and the index.

        This method creates a dictionary representation of the DataFrame where 'data' contains the DataFrame's
        data converted to a dictionary with lists for each column, and 'index' contains the DataFrame's index as a list.

        Returns:
            Dict[str, Any]: A dictionary with two keys:
                            - 'data': a dictionary where each key is a column name and each value is
                             a list of column data.
                            - 'index': a list containing the index of the DataFrame.
        """
        data = {col: self.data[col].tolist() for col in self.data.columns}
        index = self.data.index.tolist()

        return {"data": data, "index": index}

    @inherit_docstring_from(pd.DataFrame.apply)
    def apply(self, func: Callable, **kwargs) -> pd.DataFrame:
        return self.data.apply(func, **kwargs)

    @inherit_docstring_from(pd.DataFrame.map)
    def map(self, func: Callable, **kwargs) -> pd.DataFrame:
        return self.data.map(func, **kwargs)

    def unique(self) -> pd.DataFrame:
        return self.data.unique()

    def isin(self, values: Iterable) -> Iterable[bool]:
        return self.data.isin(values)

    def groupby(self, by: Union[str, Iterable[str]], **kwargs) -> List[Tuple]:
        groups = self.data.groupby(by, **kwargs)
        return list(groups)

    def loc(self, items: Iterable) -> Iterable:
        data = self.data.loc[:, items]
        return pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data

    def iloc(self, items: Iterable) -> Iterable:
        data = self.data.iloc[items]
        return pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data

    def mean(self) -> pd.DataFrame:
        return self.data.agg(["mean"])

    def max(self) -> pd.DataFrame:
        return self.data.agg(["max"])

    def min(self) -> pd.DataFrame:
        return self.data.agg(["min"])

    def count(self) -> pd.DataFrame:
        return self.data.agg(["count"])

    def sum(self) -> pd.DataFrame:
        return self.data.agg(["sum"])

    def agg(self, func) -> pd.DataFrame:
        return self.data.agg(func)
