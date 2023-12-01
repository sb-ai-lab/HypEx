from .base import BaseDataGenerator


class DataGenerator(BaseDataGenerator):
    """
    Class that will provide interface for data generation.
    It will make synthetic datasets for library testing.
    """
    def __init__(self):
        pass


class NormalDataGenerator(DataGenerator):
    """
    This class will make arrays with normal data distribution.
    """
    pass


class UniformDataGenerator(DataGenerator):
    """
    This class will make arrays with uniform data distribution.
    """
    pass


class CategoricalDataGenerator(DataGenerator):
    """
    This class will make arrays with categorical data. (words/numbers)
    """
    pass


class BinaryDataGenerator(DataGenerator):
    """
    This class will make arrays with binary data.
    """
    pass
