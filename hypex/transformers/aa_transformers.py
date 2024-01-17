from hypex.splitter.aa_spliter import SplitterAA

class TransformerSetAAGroup:
    def __init__(self, splitter: SplitterAA):
        self.splitter = splitter

    def execute(self, data):
        control_data = data.loc[self.splitter.control_indexes]
        test_data = data.loc[self.splitter.test_indexes]

        control_data["group"] = "control"
        test_data["group"] = "test"

        return pd.concat([control_data, test_data])


