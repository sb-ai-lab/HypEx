from .base import BaseDataGenerator


class DataGenerator(BaseDataGenerator):
    """
    Class that will provide interface for data generation.
    It will make synthetic datasets for library testing.
    """

    def execute(self, **kwargs):
        pass

    def generate(self):
        pass

    def add(self, df):
        pass


class NormalDataGenerator(DataGenerator):
    """
    This class will make arrays with normal data distribution.
    """

    def execute(self, **kwargs):
        pass

    def generate(self):
        pass

    def add(self, df):
        pass


class UniformDataGenerator(DataGenerator):
    """
    This class will make arrays with uniform data distribution.
    """

    def execute(self, **kwargs):
        pass

    def generate(self):
        pass

    def add(self, df):
        pass


class CategoricalDataGenerator(DataGenerator):
    """
    This class will make arrays with categorical data. (words/numbers)
    """

    def execute(self, **kwargs):
        pass

    def generate(self):
        pass

    def add(self, df):
        pass


class BinaryDataGenerator(DataGenerator):
    """
    This class will make arrays with binary data.
    """

    def execute(self, **kwargs):
        pass

    def generate(self):
        pass

    def add(self, df):
        pass
