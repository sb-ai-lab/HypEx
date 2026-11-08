from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from hypex import AATest, ABTest, HomogeneityTest, Matching
from hypex.dataset import (
    Dataset,
    FeatureRole,
    InfoRole,
    PreTargetRole,
    StratificationRole,
    TargetRole,
    TreatmentRole,
)
from hypex.utils import BackendsEnum
from hypex.utils.enums import ABNTestMethodsEnum, ABTestTypesEnum

TRUE_EFFECT = 2.0


def to_pandas(dataset) -> pd.DataFrame:
    data = dataset.backend_data.data if hasattr(dataset, "backend_data") else dataset
    return data.to_pandas() if hasattr(data, "to_pandas") else data.copy()


def make_dataset(df, roles, backend, spark):
    return Dataset(
        data=df.copy(),
        roles=roles,
        backend=backend,
        session=spark if backend == BackendsEnum.spark else None,
    )


def row_for(frame, feature="y", group=1):
    selected = frame[
        (frame["feature"] == feature) & (frame["group"].astype(str) == str(group))
    ]
    assert len(selected) == 1, (feature, group, selected)
    return selected.iloc[0]


def continuous_data(n=1200, effect=TRUE_EFFECT, groups=2, seed=0, unequal=False):
    rng = np.random.default_rng(seed)
    treat = rng.integers(0, groups, n)
    x1 = rng.normal(size=n)
    x2 = 0.6 * x1 + rng.normal(scale=0.8, size=n)
    scale = np.where(treat == 0, 0.7, 1.8) if unequal else np.ones(n)
    y = 10 + 1.8 * x1 - 0.7 * x2 + effect * treat + rng.normal(size=n) * scale
    return pd.DataFrame(
        {"id": np.arange(n), "treat": treat, "x1": x1, "x2": x2, "y": y}
    )


def categorical_data(n=1800, groups=3, seed=1, effect=True):
    rng = np.random.default_rng(seed)
    treat = rng.integers(0, groups, n)
    category = np.empty(n, dtype=object)
    for group in range(groups):
        mask = treat == group
        p_yes = 0.25 + (0.25 * group / max(groups - 1, 1) if effect else 0)
        category[mask] = rng.choice(
            ["common", "rare"], mask.sum(), p=[1 - p_yes, p_yes]
        )
    category[(treat == 0) & (rng.random(n) < 0.08)] = "control_only"
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "treat": treat,
            "y_num": rng.normal(size=n) + effect * treat,
            "y_cat": category,
        }
    )


def matching_data(n=700, seed=12):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    segment = rng.choice(["a", "b", "c"], n, p=[0.5, 0.3, 0.2])
    segment_value = pd.Series(segment).map({"a": -0.5, "b": 0.2, "c": 0.8}).to_numpy()
    propensity = 1 / (1 + np.exp(-(1.5 * x1 + 0.6 * x2 + segment_value)))
    treat = (rng.random(n) < propensity).astype(int)
    y = 2.5 * x1 + x2 + segment_value + TRUE_EFFECT * treat + rng.normal(size=n)
    return pd.DataFrame(
        {
            "id": np.arange(n),
            "treat": treat,
            "x1": x1,
            "x2": x2,
            "segment": segment,
            "y": y,
        }
    )


def aa_null(backend, spark, iterations):
    rng = np.random.default_rng(101)
    n = 1000
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "y": rng.normal(5, 2, n),
            "y_second": rng.normal(size=n),
            "x": rng.normal(size=n),
            "category": rng.choice(["a", "b", "c"], n, p=[0.6, 0.3, 0.1]),
        }
    )
    roles = {
        "id": InfoRole(),
        "y": TargetRole(),
        "y_second": TargetRole(),
        "x": FeatureRole(),
        "category": TargetRole(str),
    }
    output = AATest(
        n_iterations=iterations,
        random_states=range(iterations),
        equal_variance=False,
    ).execute(make_dataset(df, roles, backend, spark))
    experiments = to_pandas(output.experiments)
    assert len(experiments) == iterations
    rejection_rates = {}
    for column in experiments.columns:
        if " pass" not in column or column.startswith("mean "):
            continue
        if not any(name in column for name in ("y ", "y_second ", "category ")):
            continue
        values = experiments[column].dropna().astype(bool)
        if len(values) == iterations:
            rejection_rates[column] = float(values.mean())
    assert rejection_rates
    threshold = max(0.15, 0.05 + 4 * np.sqrt(0.05 * 0.95 / iterations))
    assert max(rejection_rates.values()) <= threshold, rejection_rates
    categorical_p = experiments["category Chi2Test p-value test_1"].astype(float)
    uniformity_p = float(stats.kstest(categorical_p, "uniform").pvalue)
    assert uniformity_p > 0.001, {
        "categorical_chi2_mean_p": categorical_p.mean(),
        "uniformity_p": uniformity_p,
    }
    return {
        "iterations": iterations,
        "rejection_rates": rejection_rates,
        "categorical_chi2_mean_p": float(categorical_p.mean()),
        "categorical_chi2_uniformity_p": uniformity_p,
    }


def aa_parameters(backend, spark):
    rng = np.random.default_rng(102)
    n = 900
    df = pd.DataFrame(
        {
            "id": np.arange(n),
            "y": rng.normal(size=n),
            "x": rng.normal(size=n),
            "stratum": rng.choice(["a", "b"], n, p=[0.8, 0.2]),
        }
    )
    roles = {
        "id": InfoRole(),
        "y": TargetRole(),
        "x": FeatureRole(),
        "stratum": StratificationRole(),
    }
    output = AATest(
        n_iterations=12,
        random_states=range(12),
        stratification=True,
        sample_size=0.75,
        control_size=0.3,
        float32=True,
    ).execute(make_dataset(df, roles, backend, spark))
    assert len(to_pandas(output.experiments)) == 12
    assert not to_pandas(output.best_split).empty
    return {"stratified_iterations": 12, "sample_size": 0.75, "control_size": 0.3}


def aa_three_groups(backend, spark):
    df = continuous_data(n=900, effect=0, seed=103)
    roles = {
        "id": InfoRole(),
        "y": TargetRole(),
        "x1": FeatureRole(),
        "x2": FeatureRole(),
    }
    output = AATest(
        n_iterations=8,
        random_states=range(8),
        groups_sizes=[0.5, 0.3, 0.2],
    ).execute(make_dataset(df, roles, backend, spark))
    groups = set(to_pandas(output.summary)["group"])
    assert groups == {"test_1", "test_2"}, groups
    return {"groups": sorted(groups)}


def aa_reproducibility(backend, spark):
    df = continuous_data(n=400, effect=0, seed=104)
    roles = {
        "id": InfoRole(),
        "y": TargetRole(),
        "x1": FeatureRole(),
        "x2": FeatureRole(),
    }

    def run():
        return AATest(
            n_iterations=5,
            random_states=[7, 11, 13, 17, 19],
            equal_variance=True,
        ).execute(make_dataset(df, roles, backend, spark))

    first, second = run(), run()
    pd.testing.assert_frame_equal(
        to_pandas(first.experiments), to_pandas(second.experiments)
    )
    pd.testing.assert_frame_equal(to_pandas(first.summary), to_pandas(second.summary))
    return {"reproducible_iterations": 5}


def ab_welch_and_effect(backend, spark):
    df = continuous_data(n=1600, unequal=True, seed=201)
    roles = {"id": InfoRole(), "treat": TreatmentRole(), "y": TargetRole()}
    output = ABTest(equal_variance=False).execute(
        make_dataset(df, roles, backend, spark)
    )
    summary = to_pandas(output.summary)
    row = row_for(summary)
    control = df.loc[df.treat == 0, "y"]
    test = df.loc[df.treat == 1, "y"]
    expected = stats.ttest_ind(test, control, equal_var=False)
    np.testing.assert_approx_equal(
        float(row["difference"]), test.mean() - control.mean(), significant=6
    )
    np.testing.assert_approx_equal(
        float(row["TTest p-value"]), expected.pvalue, significant=5
    )
    assert abs(float(row["difference"]) - TRUE_EFFECT) < 0.5
    null_df = continuous_data(n=1000, effect=0, seed=202)
    null_output = ABTest(equal_variance=False).execute(
        make_dataset(null_df, roles, backend, spark)
    )
    null_p = float(row_for(to_pandas(null_output.summary))["TTest p-value"])
    assert null_p > 0.01, null_p
    return {
        "effect": float(row["difference"]),
        "p_value": expected.pvalue,
        "null_p": null_p,
    }


def ab_multigroup_order_and_bonferroni(backend, spark):
    rng = np.random.default_rng(203)
    n = 1800
    treat = rng.choice([0, 1, 2, 10], n)
    y = rng.normal(size=n) + pd.Series(treat).map({0: 0, 1: 2, 2: 4, 10: 20}).to_numpy()
    df = pd.DataFrame({"treat": treat, "y": y})
    output = ABTest(multitest_method=ABNTestMethodsEnum.bonferroni).execute(
        make_dataset(df, {"treat": TreatmentRole(), "y": TargetRole()}, backend, spark)
    )
    summary = to_pandas(output.summary)
    assert summary["group"].astype(str).tolist() == ["1", "2", "10"], summary["group"]
    for group in (1, 2, 10):
        row = row_for(summary, group=group)
        expected = y[treat == group].mean() - y[treat == 0].mean()
        np.testing.assert_allclose(float(row["difference"]), expected, atol=1e-6)
    corrected = to_pandas(output.multitest)
    old = corrected["old p-value"].astype(float).to_numpy()
    np.testing.assert_allclose(
        corrected["new p-value"].astype(float), np.minimum(old * 3, 1), atol=1e-10
    )
    return {"groups": summary["group"].astype(str).tolist(), "comparisons": len(old)}


def ab_categorical_and_quantile(backend, spark):
    df = categorical_data(seed=204)
    roles = {
        "id": InfoRole(),
        "treat": TreatmentRole(),
        "y_num": TargetRole(),
        "y_cat": TargetRole(str),
    }
    output = ABTest(
        additional_tests=[ABTestTypesEnum.t_test, ABTestTypesEnum.chi2_test],
        multitest_method=ABNTestMethodsEnum.holm,
    ).execute(make_dataset(df, roles, backend, spark))
    summary = to_pandas(output.summary)
    assert set(summary["feature"]) == {"y_num", "y_cat"}
    chi_row = row_for(summary, "y_cat", 1)
    control = df.loc[df.treat == 0, "y_cat"]
    test = df.loc[df.treat == 1, "y_cat"]
    levels = sorted(set(control) | set(test))
    contingency = np.array(
        [
            [(control == level).sum() for level in levels],
            [(test == level).sum() for level in levels],
        ]
    ).T
    reference_p = stats.chi2_contingency(contingency).pvalue
    np.testing.assert_allclose(
        float(chi_row["Chi2Test p-value"]), reference_p, rtol=0.05, atol=1e-12
    )
    corrected = to_pandas(output.multitest)
    assert len(corrected) == 4
    assert corrected["new p-value"].astype(float).between(0, 1).all()
    quantile_result = "not supported on Spark"
    if backend == BackendsEnum.pandas:
        quantile_roles = {
            column: role for column, role in roles.items() if column != "y_cat"
        }
        quantile = ABTest(multitest_method="quantile").execute(
            make_dataset(df[["id", "treat", "y_num"]], quantile_roles, backend, spark)
        )
        quantile_frame = to_pandas(quantile.multitest)
        assert quantile_frame["field"].tolist() == ["y_num"]
        assert quantile_frame["accepted hypothesis"].isin([0, 1, 2, 3]).all()
        quantile_result = int(quantile_frame["accepted hypothesis"].iloc[0])
    return {
        "chi2_p": float(chi_row["Chi2Test p-value"]),
        "reference_p": reference_p,
        "quantile_result": quantile_result,
    }


def ab_cuped(backend, spark):
    rng = np.random.default_rng(205)
    n = 2000
    baseline = rng.normal(scale=3, size=n)
    df = pd.DataFrame(
        {
            "treat": rng.integers(0, 2, n),
            "y_pre": baseline + rng.normal(scale=0.5, size=n),
        }
    )
    df["y"] = baseline + TRUE_EFFECT * df.treat + rng.normal(scale=0.5, size=n)
    roles = {"treat": TreatmentRole(), "y": TargetRole(), "y_pre": PreTargetRole()}
    plain = ABTest(equal_variance=False).execute(
        make_dataset(df, roles, backend, spark)
    )
    cuped = ABTest(equal_variance=False, cuped_features={"y": "y_pre"}).execute(
        make_dataset(df, roles, backend, spark)
    )
    plain_p = float(row_for(to_pandas(plain.summary))["TTest p-value"])
    cuped_row = row_for(to_pandas(cuped.summary), "y_cuped")
    cuped_p = float(cuped_row["TTest p-value"])
    assert cuped_p < plain_p
    assert abs(float(cuped_row["difference"]) - TRUE_EFFECT) < 0.25
    reduction = to_pandas(cuped.variance_reduction_report)
    assert not reduction.empty
    return {
        "plain_p": plain_p,
        "cuped_p": cuped_p,
        "effect": float(cuped_row["difference"]),
    }


def homogeneity_numeric(backend, spark):
    df = continuous_data(n=1600, effect=3, groups=4, seed=301)
    df["treat"] = df["treat"].map({0: 0, 1: 1, 2: 2, 3: 10})
    roles = {"id": InfoRole(), "treat": TreatmentRole(), "y": TargetRole()}
    output = HomogeneityTest().execute(make_dataset(df, roles, backend, spark))
    summary = to_pandas(output.summary)
    assert summary["group"].astype(str).tolist() == ["1", "2", "10"]
    for group in (1, 2, 10):
        row = row_for(summary, group=group)
        control = df.loc[df.treat == 0, "y"]
        test = df.loc[df.treat == group, "y"]
        np.testing.assert_allclose(
            float(row["difference"]), test.mean() - control.mean(), atol=1e-6
        )
        assert float(row["TTest p-value"]) < 1e-10
    null_df = continuous_data(n=1200, effect=0, seed=302)
    null_output = HomogeneityTest().execute(
        make_dataset(null_df, roles, backend, spark)
    )
    null_p = float(row_for(to_pandas(null_output.summary))["TTest p-value"])
    assert null_p > 0.01, null_p
    return {"groups": summary["group"].astype(str).tolist(), "null_p": null_p}


def homogeneity_categorical(backend, spark):
    df = categorical_data(n=1800, seed=303)
    roles = {"id": InfoRole(), "treat": TreatmentRole(), "y_cat": TargetRole(str)}
    output = HomogeneityTest().execute(make_dataset(df, roles, backend, spark))
    summary = to_pandas(output.summary)
    assert len(summary) == 2
    assert set(summary["feature"]) == {"y_cat"}
    assert summary["Chi2Test p-value"].astype(float).between(0, 1).all()
    assert (summary["Chi2Test pass"] == "NOT OK").all()
    return {"p_values": summary["Chi2Test p-value"].astype(float).tolist()}


def assert_matching_result(output, df, full=False):
    summary = to_pandas(output.summary)
    assert set(summary.index) == {"ATT", "ATC", "ATE"}
    assert np.isfinite(summary.astype(float).to_numpy()).all()
    for effect in summary["Effect Size"].astype(float):
        assert abs(effect - TRUE_EFFECT) < 0.9, summary
    for _, row in summary.iterrows():
        midpoint = (float(row["CI Lower"]) + float(row["CI Upper"])) / 2
        width = float(row["CI Upper"]) - float(row["CI Lower"])
        assert abs(midpoint - float(row["Effect Size"])) < 0.03, summary
        assert abs(width - 2 * 1.96 * float(row["Standard Error"])) < 0.04, summary
    quality = to_pandas(output.quality_results)
    assert not quality.empty
    if full:
        indexes = to_pandas(output.indexes)
        assert len(indexes) == len(df)
        for column in indexes.columns:
            matched = indexes[column].astype(int)
            valid = matched >= 0
            original_treat = df.treat.to_numpy()[indexes.index.to_numpy()[valid]]
            matched_treat = df.treat.to_numpy()[matched[valid]]
            assert (original_treat != matched_treat).all()
        assert len(to_pandas(output.full_data)) == len(df)
    return summary["Effect Size"].astype(float).to_dict()


def matching_mahalanobis(backend, spark):
    df = matching_data(seed=401)
    roles = {
        "id": InfoRole(),
        "treat": TreatmentRole(),
        "x1": FeatureRole(),
        "x2": FeatureRole(),
        "segment": FeatureRole(str),
        "y": TargetRole(),
    }
    output = Matching(
        distance="mahalanobis",
        faiss_mode="base",
        n_neighbors=1,
        bias_estimation=True,
        quality_tests="auto",
        compute_indexes=True,
        extract_full_data=True,
    ).execute(make_dataset(df, roles, backend, spark))
    effects = assert_matching_result(output, df, full=True)
    return {"effects": effects}


def matching_l2_fast(backend, spark):
    df = matching_data(n=1200, seed=402)
    roles = {
        "id": InfoRole(),
        "treat": TreatmentRole(),
        "x1": FeatureRole(),
        "x2": FeatureRole(),
        "y": TargetRole(),
    }
    output = Matching(
        distance="l2",
        faiss_mode="fast",
        n_neighbors=3,
        bias_estimation=False,
        quality_tests=["t-test", "ks-test"],
        compute_indexes=True,
    ).execute(make_dataset(df, roles, backend, spark))
    effects = assert_matching_result(output, df)
    return {"effects": effects}


def matching_repeated_calibration(backend, spark):
    roles = {
        "id": InfoRole(),
        "treat": TreatmentRole(),
        "x1": FeatureRole(),
        "x2": FeatureRole(),
        "y": TargetRole(),
    }
    effects = []
    covered = []
    for seed in range(500, 540):
        df = matching_data(n=400, seed=seed)
        output = Matching(
            distance="l2",
            faiss_mode="base",
            quality_tests=["t-test"],
            bias_estimation=True,
        ).execute(make_dataset(df, roles, backend, spark))
        row = to_pandas(output.summary).loc["ATE"]
        effects.append(float(row["Effect Size"]))
        covered.append(float(row["CI Lower"]) <= TRUE_EFFECT <= float(row["CI Upper"]))
    mean_effect = float(np.mean(effects))
    coverage = float(np.mean(covered))
    assert abs(mean_effect - TRUE_EFFECT) < 0.35, mean_effect
    assert coverage > 0.5, coverage
    return {
        "runs": len(effects),
        "mean_ate": mean_effect,
        "ate_sd": float(np.std(effects)),
        "ci_coverage": coverage,
    }


def matching_single_quality_string(backend, spark):
    df = matching_data(n=400, seed=403)
    roles = {
        "id": InfoRole(),
        "treat": TreatmentRole(),
        "x1": FeatureRole(),
        "x2": FeatureRole(),
        "y": TargetRole(),
    }
    output = Matching(distance="l2", quality_tests="t-test").execute(
        make_dataset(df, roles, backend, spark)
    )
    effects = assert_matching_result(output, df)
    return {"effects": effects}


CASES = {
    "aa": [aa_null, aa_parameters, aa_three_groups, aa_reproducibility],
    "ab": [
        ab_welch_and_effect,
        ab_multigroup_order_and_bonferroni,
        ab_categorical_and_quantile,
        ab_cuped,
    ],
    "homo": [homogeneity_numeric, homogeneity_categorical],
    "matching": [
        matching_mahalanobis,
        matching_l2_fast,
        matching_repeated_calibration,
        matching_single_quality_string,
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=["pandas", "spark", "both"], default="both"
    )
    parser.add_argument("--only", choices=[*CASES, "all"], default="all")
    parser.add_argument("--aa-iterations", type=int, default=200)
    parser.add_argument(
        "--output", default="examples/experiments/alpha_agent_behavior_results.json"
    )
    args = parser.parse_args()

    backend_names = ["pandas", "spark"] if args.backend == "both" else [args.backend]
    spark = None
    if "spark" in backend_names:
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.builder.master("local[2]")
            .appName("HypEx-alpha-behavior-check")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .config("spark.sql.execution.arrow.pyspark.enabled", "false")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("ERROR")

    selected = list(CASES) if args.only == "all" else [args.only]
    results = []
    try:
        for backend_name in backend_names:
            backend = BackendsEnum[backend_name]
            for family in selected:
                for function in CASES[family]:
                    name = function.__name__
                    started = time.monotonic()
                    record = {"case": name, "family": family, "backend": backend_name}
                    try:
                        if function is aa_null:
                            metrics = function(backend, spark, args.aa_iterations)
                        else:
                            metrics = function(backend, spark)
                        record.update(status="PASS", metrics=metrics)
                    except Exception as error:
                        record.update(
                            status="FAIL",
                            error=f"{type(error).__name__}: {error}",
                            traceback=traceback.format_exc(),
                        )
                    record["seconds"] = round(time.monotonic() - started, 3)
                    results.append(record)
                    print(json.dumps(record, default=str), flush=True)
    finally:
        if spark is not None:
            spark.stop()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, default=str) + "\n")
    failures = [result for result in results if result["status"] == "FAIL"]
    print(
        f"Completed {len(results)} cases: {len(results) - len(failures)} passed, {len(failures)} failed"
    )
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
