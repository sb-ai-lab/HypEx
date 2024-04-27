from abc import ABC, abstractmethod
from typing import Iterable


class DatasetBase(ABC):
    """
    Abstract base class for all datasets in the HypEx library. Defines a common API for dataset manipulations.
    """

    @abstractmethod
    def __len__(self):
        """Return the number of entries in the dataset."""
        pass

    @abstractmethod
    def __repr__(self):
        """Return a string representation of the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Retrieve an item by its index."""
        pass

    @abstractmethod
    def add_column(self, data, name):
        """Add a column to the dataset."""
        pass

    def from_dict(self, data):
        """Create a dataset from a dictionary."""
        pass

    def _create_empty(self, index=None):
        """Create an empty dataset with optional index."""
        pass

    @abstractmethod
    def apply(self, *args, **kwargs):
        """Apply a function across the dataset."""
        pass

    @abstractmethod
    def map(self, *args, **kwargs):
        """Map a function over the dataset entries."""
        pass

    @property
    @abstractmethod
    def index(self):
        """Get the index of the dataset."""
        pass

    @property
    @abstractmethod
    def columns(self):
        """Get the list of column names in the dataset."""
        pass

    @abstractmethod
    def isin(self, values: Iterable) -> Iterable[bool]:
        """Check whether each element in the dataset is in the specified values."""
        pass

    def groupby(self, by, **kwargs):
        """Group the dataset by some criteria."""
        pass

    def loc(self, values: Iterable) -> Iterable:
        """Access a group of rows and columns by labels."""
        pass

    def iloc(self, values: Iterable) -> Iterable:
        """Access a group of rows and columns by integer positions."""
        pass


class DatasetBackend(DatasetBase):
    @property
    def name(self):
        return str(self.__class__.__name__).lower().replace("dataset", "")
