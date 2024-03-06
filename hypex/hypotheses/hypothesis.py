import json
from typing import Union, Dict

from jsonschema import validate

from hypex.dataset.dataset import Dataset
from hypex.dataset.roles import default_roles


class Hypothesis:
    def __init__(self, config: Union[Dict, str]):
        if isinstance(config, str):
            with open(config, "rb") as file:
                config = json.load(file)
        with open("hypothesis_scheme.json", "rb") as file:
            self.scheme = json.load(file)
        self.config = config
        self.dataset = config["dataset"]
        self.experiment = config["experiment"]
        self.report = config["report"]
        self.validate_config()
        self._parse_config()

    def validate_config(self):
        validate(self.config, self.scheme)
        if len(self.dataset["roles"]["role_names"]) != len(
            self.dataset["roles"]["columns"]
        ):
            raise ValueError(
                f"Invalid number of columns and role_names. Columns and role_names must have equal length.\n "
                f"role_names contains {len(self.dataset['roles']['role_names'])} values and columns contains {len(self.dataset['roles']['columns'])}"
            )

    def _parse_config(self):
        self.dataset = self._parse_dataset()
        self.experiment = self._parse_experiment()
        self.report = self._parse_report()

    def _parse_dataset(self):
        if "data" in self.dataset.keys():
            data = self.dataset["data"]
        else:
            data = self.dataset["path"]
        roles = {
            i: default_roles.get(j.lower())
            for i, j in zip(
                self.dataset["roles"]["columns"], self.dataset["roles"]["role_names"]
            )
        }
        return Dataset(data=data, roles=roles, backend=self.dataset["backend"])

    def _parse_experiment(self):
        pass

    def _parse_report(self):
        pass
