import json
from typing import Optional

from jsonschema import validate  # type: ignore

from hypex.dataset.dataset import Dataset
from hypex.dataset.roles import default_roles


class Hypothesis:
    def __init__(self, config: str):
        with open(config, "rb") as file:
            opened_config = json.load(file)
        with open("schemes/scheme.json", "rb") as file:
            self.scheme = json.load(file)
        self.config = opened_config
        self.dataset = self.config.get("dataset")
        self.experiment = self.config.get("experiment")
        self.report = self.config.get("report")
        self.validate_config()
        self._parse_config()

    def validate_config(self):
        validate(self.config, self.scheme)
        if (
            "data" in self.dataset.keys()
            and "path" not in self.dataset.keys()
            and not self.dataset["data"]["data"]
        ):
            raise ValueError("Data ot path to data must be added")
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
        data = (
            self.dataset["data"]
            if "data" in self.dataset.keys()
            else self.dataset["path"]
        )
        roles = {
            default_roles.get(i.lower()): j
            for i, j in zip(
                self.dataset["roles"]["role_names"], self.dataset["roles"]["columns"]
            )
        }
        return Dataset(data=data, roles=roles, backend=self.dataset["backend"])

    def _parse_experiment(self):
        print(self.experiment)

    def _parse_report(self):
        pass

    def to_json(self, file: Optional[str] = None):
        # return json.dumps(self.dataset.to_json(), self.experiment.to_json(), self.report.to_json())
        if file:
            with open(file, "w") as f:
                json.dump(
                    {"dataset": self.dataset.to_dict(), "experiment": {}, "report": {}},
                    f,
                    indent=4,
                )
        return json.dumps(
            {"dataset": self.dataset.to_dict(), "experiment": {}, "report": {}}
        )


if __name__ == "__main__":
    hypo = Hypothesis("test_config.json")
    print(hypo.dataset)
