from abc import ABC

from scipy.stats import ttest_ind

from hypex.experiment.base import Executor

class StatHypothesisTesting(ABC, Executor):
    statistic: float
    p_value: float

    def __init__(self, target_field: str, group_field: str):
        self.target_field = target_field
        self.group_field = group_field

    def execute(self, data):
        pass
        
        
# class StatTTest(StatHypothesisTesting):
    
#     def execute(self, data):
        