# import json
# from typing import Optional, Union, Dict, Any
#
# from jsonschema import validate  # type: ignore
#
# from hypex.dataset import Dataset, default_roles, InfoRole
# from hypex.factory.base import Factory
#
#
# class Hypothesis:
#     def __init__(self, config: Union[str, Dict[str, Any]]):
#         if isinstance(config, str):
#             with open(config, "rb") as file:
#                 opened_config = json.load(file)
#         else:
#             opened_config = config
#         with open("hypex\\hypotheses\\schemes\\scheme.json", "rb") as file:
#             self.scheme = json.load(file)
#         self.config = opened_config
#         self.dataset = self.config.get("dataset")
#         self.experiment = self.config.get("experiment")
#         self.report = self.config.get("report")
#         self.validate_config()
#         self._parse_config()
#
#     def validate_config(self):
#         validate(self.config, self.scheme)
#         if (
#             "data" in self.dataset.keys()
#             and "path" not in self.dataset.keys()
#             and not self.dataset["data"]["data"]
#         ):
#             raise ValueError("Data or path to data must be added")
#         # if len(self.dataset["roles"]["role_names"]) != len(
#         #     self.dataset["roles"]["columns"]
#         # ):
#         #     raise ValueError(
#         #         f"Invalid number of columns and role_names. Columns and role_names must have equal length.\n "
#         #         f"role_names contains {len(self.dataset['roles']['role_names'])} values and columns contains {len(self.dataset['roles']['columns'])}"
#         #     )
#
#     def _parse_config(self):
#         self.dataset = self._parse_dataset()
#
#     def _parse_dataset(self):
#         data = (
#             self.dataset["data"]
#             if "data" in self.dataset.keys()
#             else self.dataset["path"]
#         )
#         roles = {}
#         for column in self.dataset["columns"]:
#             role = default_roles.get(column["role"].lower(), InfoRole())
#             role.data_type = column["dataType"] if column.get("dataType") else None
#             roles.update({column["name"]: role})
#         return Dataset(data=data, roles=roles, backend=self.dataset["backend"])
#
#     def to_json(self, file: Optional[str] = None):
#         # return json.dumps(self.dataset.to_json(), self.experiment.to_json(), self.report.to_json())
#         if file:
#             with open(file, "w") as f:
#                 json.dump(
#                     {"dataset": self.dataset.to_dict(), "experiment": {}, "report": {}},
#                     f,
#                     indent=4,
#                 )
#         return json.dumps(
#             {"dataset": self.dataset.to_dict(), "experiment": {}, "report": {}}
#         )
#
#     def execute(self):
#         experiment_data, self.experiment = Factory(self).execute()
#         return experiment_data, self.experiment
