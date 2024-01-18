from hypex.experiment.base import Executor

class MetricMean(Executor):
    def __init__(self, field: str, out_field: str):
        self.field = field
        self.out_field = out_field
    
    def execute(self, data):
        data[self.out_field] = data[self.field].mean()
        return data