import numpy as np
from scipy.stats import mode  # type: ignore

from hypex.stats.base import AggStat


class Mean(AggStat):
    def calc(self, data):
        return data.data.mean()


class Median(AggStat):
    def calc(self, data):
        return np.median(data, **self.kwargs)


class Mode(AggStat):
    def calc(self, data):
        return mode(data, **self.kwargs)


class Std(AggStat):
    def calc(self, data):
        return np.std(data, **self.kwargs)


class Variance(AggStat):
    def calc(self, data):
        return np.var(data, **self.kwargs)


class Min(AggStat):
    def calc(self, data):
        return np.min(data, **self.kwargs)


class Max(AggStat):
    def calc(self, data):
        return np.max(data, **self.kwargs)


class Size(AggStat):
    def calc(self, data):
        return len(data)
