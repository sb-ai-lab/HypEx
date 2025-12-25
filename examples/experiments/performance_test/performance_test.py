from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import sys
import time
import tracemalloc
import warnings
from collections import defaultdict
from typing import ClassVar

import jsonschema
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from alive_progress import alive_bar
from tqdm import tqdm

from hypex import AATest
from hypex.dataset import Dataset, TargetRole

warnings.filterwarnings("ignore")

sys.path.append("../../..")


class DataProfiler:
    default_data_params: ClassVar[dict] = {
        "n_columns": 10,
        "n_rows": 10000,
        "n2c_ratio": 0.7,
        "rs": 42,
        "num_range": (-100, 100),
        "n_categories": 10,
    }

    def __init__(self, fixed_data_params: dict | None = None):
        fixed_data_params = fixed_data_params or {}
        self.fixed_data_params = self.default_data_params.copy()
        self.fixed_data_params.update(fixed_data_params)

        # Remove any keys that aren't in default params
        for key in list(self.fixed_data_params.keys()):
            if key not in list(self.default_data_params.keys()):
                del self.fixed_data_params[key]

    @staticmethod
    def _generate_synthetic_data(
        n_columns: int,
        n_rows: int,
        n2c_ratio: float,
        rs: int | None,
        num_range: tuple,
        n_categories: int,
    ) -> pd.DataFrame:
        if rs is not None:
            np.random.seed(rs)

        n_numerical = int(n_columns * n2c_ratio)
        n_categorical = n_columns - n_numerical

        numerical_data = np.random.randint(
            num_range[0], num_range[1], size=(n_rows, n_numerical)
        )

        categories = [f"Category_{i + 1}" for i in range(n_categories)]
        categorical_data = np.random.choice(categories, size=(n_rows, n_categorical))

        return pd.DataFrame(
            np.hstack((numerical_data, categorical_data)),
            columns=[f"num_col_{i}" for i in range(n_numerical)]
            + [f"cat_col_{i}" for i in range(n_categorical)],
        )

    def create_dataset(
        self, params: dict
    ) -> tuple[Dataset, dict[str, int | tuple[int, int] | float]]:
        all_params = self.fixed_data_params.copy()
        all_params.update(params)

        data = self._generate_synthetic_data(**all_params)
        return (
            Dataset(roles={column: TargetRole() for column in data.columns}, data=data),
            all_params,
        )


class ExperimentProfiler:
    default_experiment_params: ClassVar[dict] = {"n_iterations": 10}

    def __init__(
        self,
        fixed_experiment_params: dict | None = None,
        experiment: type = AATest,
    ):
        fixed_experiment_params = fixed_experiment_params or {}
        self.fixed_experiment_params = self.default_experiment_params.copy()
        self.fixed_experiment_params.update(fixed_experiment_params)
        self.experiment = experiment

        # Remove any keys that aren't in default params
        for key in list(self.fixed_experiment_params.keys()):
            if key not in list(self.default_experiment_params.keys()):
                del self.fixed_experiment_params[key]

    def get_experiment(self, experiment_params):
        all_params = self.fixed_experiment_params.copy()
        all_params.update(experiment_params)
        return self.experiment(**all_params), all_params


class PerformanceTester:
    resume: ClassVar[defaultdict] = defaultdict(dict)

    def __init__(
        self,
        dataProfiler: DataProfiler,
        experimentProfiler: ExperimentProfiler,
        iterable_params: list | None = None,
        use_memory: bool = True,
        rewrite: bool = True,
    ):
        self.dataProfiler = dataProfiler
        self.experimentProfiler = experimentProfiler
        self.iterable_params = iterable_params or []
        self.use_memory = use_memory
        self.rewrite = rewrite

    def get_params(self):
        for params in self.iterable_params:
            all_params = params.copy()
            if "n_iterations" in list(params.keys()):
                experiment_params = {"n_iterations": params["n_iterations"]}
                params.pop("n_iterations", None)
            else:
                experiment_params = {}
            data_params = params
            yield all_params, self.dataProfiler.create_dataset(
                data_params
            ), self.experimentProfiler.get_experiment(experiment_params)

    def get_number_params(self):
        return len(self.iterable_params)

    def execute(self, file_name, analysis="onefactor"):
        if self.rewrite:
            with open(file_name, "w", newline="") as file:
                writer = csv.writer(file)
                row_items = [
                    "analysis",
                    *list(self.experimentProfiler.fixed_experiment_params.keys()),
                    *list(self.dataProfiler.fixed_data_params.keys()),
                    "time",
                    "M1",
                    "M2",
                ]
                writer.writerow(row_items)
        with alive_bar(
            self.get_number_params(),
            bar="squares",
            spinner="dots_waves2",
            title=f"Analysis : {analysis}",
        ) as bar:
            for params, data, experiment in tqdm(self.get_params()):
                combined_params = {**data[1], **experiment[1]}
                print(f"{combined_params}")

                manager = mp.Manager()
                return_dict1 = manager.dict()
                return_dict2 = manager.dict()

                process = mp.Process(
                    target=self.function_performance,
                    args=(experiment[0].execute, {"data": data[0]}, return_dict1),
                )
                process.start()

                monitor = mp.Process(
                    target=self._memory_monitor, args=(process.pid, return_dict2)
                )
                monitor.start()

                process.join()
                monitor.join()

                max_memory_mb = return_dict2["max_memory"] / 1024**2

                with open(file_name, "a", newline="") as file:
                    writer = csv.writer(file)
                    combined_params = {**experiment[1], **data[1]}
                    row_items = [
                        analysis,
                        *list(combined_params.values()),
                        *return_dict1["results"],
                        max_memory_mb,
                    ]
                    writer.writerow(row_items)
                bar()

    @staticmethod
    def _memory_monitor(pid, return_dict, interval=0.1):
        process = psutil.Process(pid)
        max_memory = 0

        while process.is_running():
            try:
                mem_info = process.memory_info().rss  # Current memory usage (RSS)
                max_memory = max(max_memory, mem_info)  # Update maximum
                time.sleep(interval)
            except psutil.NoSuchProcess:
                break  # If the process has finished

        return_dict["max_memory"] = max_memory  # Save the result

    def function_performance(self, func, param_dict, return_dict):
        param_dict = param_dict or {}
        exec_time = None
        memory_usage = None

        start_time = time.time()

        if self.use_memory:
            tracemalloc.start()

        func(**param_dict)

        if self.use_memory:
            _, memory_usage = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        end_time = time.time()
        exec_time = end_time - start_time

        return_dict["results"] = [
            exec_time,
            memory_usage / 10**6 if self.use_memory else None,
        ]


def performance_test_plot(
    params: dict,
    output_path: str,
    title="The results of the one-factor performance test of the AA Test",
):
    df = pd.read_csv(output_path)
    df = df[df.analysis == "onefactor"]
    df = df[["time", "M1", "M2"]]
    result = {"Var": [], "P": []}
    for key, values in params.items():
        for value in values:
            result["Var"].append(key)
            result["P"].append(value)
    df["Var"] = result["Var"]
    df["P"] = result["P"]
    plot_size = df["Var"].nunique()
    if plot_size == 1:
        fig, axs = plt.subplots(1, 3, figsize=(33, 5))
        axs = axs.reshape(1, -1)  # Делаем 2D массив
    else:
        fig, axs = plt.subplots(plot_size, 3, figsize=(plot_size * 11, 15))

    for counter, (var, table) in enumerate(df.groupby("Var")):
        table = table.sort_values(by="P")
        axs[counter, 0].plot(table["P"], table["time"])
        axs[counter, 0].set_title(f"{var} time")
        axs[counter, 0].grid(True)
        axs[counter, 0].set_ylabel("sec")
        axs[counter, 1].plot(table["P"], table["M1"])
        axs[counter, 1].set_title(f"{var} memory of execution")
        axs[counter, 1].grid(True)
        axs[counter, 1].set_ylabel("MB")
        axs[counter, 2].plot(table["P"], table["M2"])
        axs[counter, 2].set_title(f"{var} memory of process")
        axs[counter, 2].grid(True)
        axs[counter, 2].set_ylabel("MB")
    fig.suptitle(title)
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(f"{output_path[:output_path.rfind('.')]}.png")


def executor(config: dict, output_path: str):
    output_path = f"{output_path}.csv"

    if "fixed_params" not in config:
        config["fixed_params"] = {}

    experimentProfiler = ExperimentProfiler(
        fixed_experiment_params=config["fixed_params"], experiment=AATest
    )
    dataProfiler = DataProfiler(fixed_data_params=config["fixed_params"])
    test = PerformanceTester(
        experimentProfiler=experimentProfiler, dataProfiler=dataProfiler
    )

    if "onefactor_params" in config:
        iterable_params = []

        def _format(param):
            return param if isinstance(param, list) else [param]

        for param_name, params in config["onefactor_params"].items():
            params = _format(params)
            for param in params:
                iterable_params.append({param_name: param})
        test.iterable_params = iterable_params
        test.execute(output_path, analysis="onefactor")
        test.rewrite = False
        performance_test_plot(config["onefactor_params"], output_path)

    if "montecarlo_params" in config:
        mcparams = config["montecarlo_params"]
        df = {}
        for key in list(mcparams["bounds"].keys()):
            df[key] = np.round(
                np.random.uniform(
                    mcparams["bounds"][key]["min"],
                    mcparams["bounds"][key]["max"],
                    mcparams["num_points"],
                )
            ).astype(int)
        keys = list(df.keys())
        df = [
            {key: value.item() for key, value in zip(keys, values)}
            for values in zip(*df.values())
        ]
        test.iterable_params = df
        test.execute(output_path, analysis="montecarlo")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    file_path_schema = os.path.join(script_dir, "config.schema.json")
    file_path_config = os.path.join(script_dir, "config.json")

    with open(file_path_schema) as file1, open(file_path_config) as file2:
        schema = json.load(file1)
        config = json.load(file2)
    try:
        jsonschema.validate(instance=config, schema=schema)
    except jsonschema.exceptions.ValidationError as err:
        raise ValueError(f"JSON validation error: {err}") from err

    output_path = "aa_performance_test_result"
    executor(config=config, output_path=output_path)
