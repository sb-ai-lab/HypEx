"""Synthetic data generators for HypEx tutorials and tests.

All generators return a :class:`pandas.DataFrame` and accept an
optional ``random_state`` seed for reproducibility.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd  # pyright: ignore[reportMissingModuleSource]
from numpy.typing import NDArray
from scipy import stats  # pyright: ignore[reportMissingImports]

# ── Module-level constants ───────────────────────────────────────────────────

_NUM_MONTHS: int = 12
_ANALYSIS_CUTOFF_MONTH: int = 3
_INDUSTRY_NAMES: tuple[str, ...] = ("Finance", "E-commerce", "Logistics")


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_rng(random_state: int | None = None) -> np.random.Generator:
    """Create a NumPy random generator with an optional seed.

    Args:
        random_state: Seed for reproducibility.  ``None`` gives a
            non-reproducible generator.

    Returns:
        A :class:`numpy.random.Generator` instance.
    """
    return np.random.default_rng(random_state)


def _as_list(value: Any, default_factory: Any = None) -> list:
    """Normalise a scalar / sequence / ``None`` into a list.

    Args:
        value: Input value.
        default_factory: Callable that produces the default list when
            *value* is ``None``.

    Returns:
        A list representation of *value*.
    """
    if value is None:
        return default_factory() if default_factory else []
    if isinstance(value, (str, int, float)):
        return [value]
    return list(value)


def sigmoid(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute the logistic sigmoid element-wise.

    Defined as ``sigmoid(x) = 1 / (1 + exp(-x))``.

    Args:
        x: Input array.

    Returns:
        Sigmoid of *x*, same shape.
    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_division(
    x: NDArray,
    dependent_division: bool = True,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int64]:
    """Generate a binary vector via a sigmoid-based probability.

    Args:
        x: Input array used to compute assignment probabilities.
        dependent_division: If ``True``, the binary vector depends on
            *x*; otherwise it is drawn with ``p = 0.5``.
        rng: Random generator.  Falls back to the global state when
            ``None`` (backward compatibility).

    Returns:
        Binary integer array of the same length as *x*.
    """
    rng = rng or np.random
    if dependent_division:
        prob = sigmoid((x - x.mean()) / x.std())
        return rng.binomial(1, prob)
    return rng.binomial(1, 0.5, size=len(x))


# ── NaN injection ────────────────────────────────────────────────────────────


def set_nans(
    data: pd.DataFrame,
    na_step: Sequence[int] | int | None = None,
    nan_cols: Sequence[str] | str | None = None,
) -> pd.DataFrame:
    """Return a copy of *data* with ``NaN`` inserted at regular intervals.

    Args:
        data: Input DataFrame (not mutated).
        na_step: Step interval(s) for inserting NaNs.  Defaults to
            ``[10]``.
        nan_cols: Column name(s) to affect.  Defaults to all columns.

    Returns:
        A **new** DataFrame with NaNs injected.

    Raises:
        ValueError: If *na_step* or *nan_cols* resolve to empty lists.
    """
    result = data.copy()

    na_step_list: list[int] = _as_list(na_step, default_factory=lambda: [10])
    nan_cols_list: list[str] = _as_list(
        nan_cols, default_factory=lambda: list(result.columns)
    )

    if not na_step_list or not nan_cols_list:
        raise ValueError("na_step and nan_cols must not be empty.")

    # Align lengths: pad shorter, truncate longer.
    if len(na_step_list) < len(nan_cols_list):
        na_step_list += [na_step_list[-1]] * (
            len(nan_cols_list) - len(na_step_list)
        )
    else:
        na_step_list = na_step_list[: len(nan_cols_list)]

    for col, step in zip(nan_cols_list, na_step_list):
        if col in result.columns:
            result.loc[step::step, col] = None

    return result


# ── Advanced generator ───────────────────────────────────────────────────────


class DataGenerator:
    """Advanced synthetic data generator with lagged targets and controlled
    correlation structure.

    Generates a panel-like dataset with treatment assignment, lagged
    covariates (``X1``, ``X2``), and a lagged outcome chain (``y0``).

    Args:
        n_samples: Number of observations.
        distributions: Per-variable distribution specs.  Defaults to
            normal ``X1``, Bernoulli ``X2``, normal ``y0``.
        time_correlations: Autocorrelation coefficient per variable.
        effect_size: Additive treatment effect on the outcome.
        seed: Random seed for reproducibility.

    Example:
        >>> gen = DataGenerator(n_samples=500, seed=42)
        >>> df = gen.generate()
        >>> df.shape
        (500, 11)
    """

    _DEFAULT_DISTRIBUTIONS: ClassVar[dict[str, dict[str, Any]]] = {
        "X1": {"type": "normal", "mean": 1, "std": 2},
        "X2": {"type": "bernoulli", "p": 0.4},
        "y0": {"type": "normal", "mean": 10, "std": 3},
    }
    _DEFAULT_CORRELATIONS: ClassVar[dict[str, float]] = {
        "X1": 0.7,
        "X2": 0.6,
        "y0": 0.8,
    }
    _COVARIATE_VARS: tuple[str, ...] = ("X1", "X2")

    def __init__(
        self,
        n_samples: int = 2000,
        distributions: dict[str, dict[str, Any]] | None = None,
        time_correlations: dict[str, float] | None = None,
        effect_size: float = 5.0,
        seed: int | None = None,
    ) -> None:
        self.n_samples = n_samples
        self.distributions = distributions or dict(self._DEFAULT_DISTRIBUTIONS)
        self.time_correlations = time_correlations or dict(self._DEFAULT_CORRELATIONS)
        self.effect_size = effect_size
        self.rng = _make_rng(seed)

    # ── private samplers ──────────────────────────────────────────────

    def _bernoulli_pair(
        self, p: float, rho: float
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Sample a correlated Bernoulli pair (current, lag).

        Args:
            p: Marginal probability.
            rho: Correlation between the two draws.

        Returns:
            Tuple ``(current, lag)`` of binary arrays.

        Raises:
            ValueError: If *rho* exceeds the feasible range for *p*.
        """
        rho_max = min(p / (1 - p), (1 - p) / p)
        if abs(rho) > rho_max:
            raise ValueError(f"Impossible correlation {rho} for p={p}")

        base = p * (1 - p)
        p11 = p * p + rho * base
        p10 = base - rho * base
        p01 = base - rho * base
        p00 = (1 - p) ** 2 + rho * base

        states = self.rng.choice(4, size=self.n_samples, p=[p00, p01, p10, p11])
        lag = ((states == 1) | (states == 3)).astype(int)
        current = ((states == 2) | (states == 3)).astype(int)
        return current, lag

    def _correlated_pair(
        self,
        dist_type: str,
        params: dict[str, Any],
        rho: float,
        u_shift: NDArray | float = 0,
    ) -> tuple[NDArray, NDArray]:
        """Sample a correlated pair from the specified distribution.

        Args:
            dist_type: One of ``"normal"``, ``"bernoulli"``, ``"gamma"``.
            params: Distribution parameters.
            rho: Correlation coefficient.
            u_shift: Optional additive shift (confounding).

        Returns:
            Tuple ``(current, lag)`` arrays.

        Raises:
            ValueError: For unsupported distribution types.
        """
        if dist_type == "normal":
            var = params["std"] ** 2
            cov = [[var, rho * var], [rho * var, var]]
            mean = [params["mean"], params["mean"]]
            return (
                self.rng.multivariate_normal(mean, cov, self.n_samples).T
                + u_shift
            )

        if dist_type == "bernoulli":
            return self._bernoulli_pair(params["p"], rho)

        if dist_type == "gamma":
            z = self.rng.multivariate_normal(
                [0, 0], [[1, rho], [rho, 1]], self.n_samples
            )
            u = stats.norm.cdf(z)
            shape, scale = params["shape"], params["scale"]
            current = stats.gamma.ppf(u[:, 0], a=shape, scale=scale)
            lag = stats.gamma.ppf(u[:, 1], a=shape, scale=scale)
            return current, lag

        raise ValueError(f"Unsupported distribution: {dist_type}")

    def _correlated_chain(
        self,
        params: dict[str, Any],
        rho: float,
        n_points: int,
    ) -> NDArray:
        """Sample an AR(1)-like correlated chain of length *n_points*.

        Args:
            params: Must contain ``"mean"`` and ``"std"``.
            rho: Lag-1 autocorrelation.
            n_points: Number of time points.

        Returns:
            Array of shape ``(n_points, n_samples)``.
        """
        var = params["std"] ** 2
        indices = np.arange(n_points)
        cov = var * rho ** np.abs(indices[:, None] - indices[None, :])
        mean = np.full(n_points, params["mean"])
        return self.rng.multivariate_normal(mean, cov, self.n_samples).T

    # ── public API ────────────────────────────────────────────────────

    def generate(self) -> pd.DataFrame:
        """Generate the full synthetic dataset.

        Returns:
            DataFrame with treatment, covariates, lagged features,
            and outcome columns.
        """
        data: dict[str, NDArray] = {}

        # Treatment assignment
        data["z"] = self.rng.binomial(1, 0.5, self.n_samples)
        data["U"] = self.rng.normal(0, 1, self.n_samples)
        propensity = np.clip(0.3 + 0.4 * data["z"] + 0.3 * data["U"], 0, 1)
        data["D"] = self.rng.binomial(1, propensity)
        data["d"] = data["D"] * data["z"]

        # Covariates with lags
        for var in self._COVARIATE_VARS:
            spec = self.distributions[var]
            current, lag = self._correlated_pair(
                spec["type"], spec, self.time_correlations[var], data["U"]
            )
            data[var] = current
            data[f"{var}_lag"] = lag

        # Outcome chain
        y_spec = self.distributions["y0"]
        y_rho = self.time_correlations["y0"]
        if y_spec["type"] == "normal":
            chain = self._correlated_chain(y_spec, y_rho, n_points=3)
            data["y0"], data["y0_lag_1"], data["y0_lag_2"] = (
                chain[2], chain[1], chain[0]
            )
        else:
            current, lag1 = self._correlated_pair(y_spec["type"], y_spec, y_rho)
            lag2, _ = self._correlated_pair(y_spec["type"], y_spec, y_rho)
            data["y0"], data["y0_lag_1"], data["y0_lag_2"] = current, lag1, lag2

        # Treatment effect on outcome
        noise = self.rng.normal(0, 0.01, self.n_samples)
        data["y1"] = data["y0"] + self.effect_size * (1 + data["U"]) + noise
        data["y"] = np.where(data["d"] == 1, data["y1"], data["y0"])

        df = pd.DataFrame(data)
        return df.rename(
            columns={
                "X1": "X1_lag1",
                "X2": "X2_lag1",
                "X1_lag": "X1_lag2",
                "X2_lag": "X2_lag2",
            }
        )


# ── Tutorial generators ──────────────────────────────────────────────────────


def create_test_data(
    num_users: int = 10_000,
    na_step: Sequence[int] | int | None = None,
    nan_cols: Sequence[str] | str | None = None,
    file_name: str | None = None,
    output_dir: str | Path | None = None,
    exact_ATT: int = 100,
    rs: int | None = None,
) -> pd.DataFrame:
    """Create a synthetic A/B-test panel dataset for tutorials.

    Simulates user spending behaviour over 12 months with a known
    treatment effect, then aggregates into pre/post signup means.

    Args:
        num_users: Number of synthetic users.
        na_step: Step interval(s) for NaN injection.
        nan_cols: Column(s) to inject NaNs into.
        file_name: If provided, saves the CSV to *output_dir*.
        output_dir: Directory for CSV output.  Defaults to ``"."``.
        exact_ATT: Exact additive treatment effect to embed.
        rs: Random seed.

    Returns:
        Aggregated DataFrame with per-user features.
    """
    rng = _make_rng(rs)

    panel = _build_panel(num_users, rng)
    panel = _apply_treatment_effect(panel, exact_ATT)
    data = _aggregate_spends(panel)
    data = _add_demographics(data, rng)
    data = set_nans(data, na_step, nan_cols)

    if file_name is not None:
        out = Path(output_dir or ".") / f"{file_name}.csv"
        data.to_csv(out, index=False)

    return data


def _build_panel(num_users: int, rng: np.random.Generator) -> pd.DataFrame:
    """Build the raw user × month panel.

    Args:
        num_users: Number of users.
        rng: Random generator.

    Returns:
        Long-format panel DataFrame.
    """
    signup_months = rng.choice(
        np.arange(1, _NUM_MONTHS), num_users
    ) * rng.integers(0, 2, size=num_users)

    panel = pd.DataFrame(
        {
            "user_id": np.repeat(np.arange(num_users), _NUM_MONTHS),
            "signup_month": np.repeat(signup_months, _NUM_MONTHS),
            "month": np.tile(np.arange(1, _NUM_MONTHS + 1), num_users),
            "spend": rng.poisson(500, num_users * _NUM_MONTHS),
        }
    )
    panel["treat"] = panel["signup_month"] > 0
    # Monotonically decreasing seasonal effect
    panel["spend"] -= panel["month"] * 10
    return panel


def _apply_treatment_effect(panel: pd.DataFrame, att: int) -> pd.DataFrame:
    """Add the treatment effect to post-signup spending.

    Args:
        panel: Raw panel.
        att: Additive treatment effect.

    Returns:
        Panel with modified ``spend``.
    """
    mask = (panel["signup_month"] < panel["month"]) & panel["treat"]
    panel.loc[mask, "spend"] += att
    return panel


def _aggregate_spends(panel: pd.DataFrame) -> pd.DataFrame:
    """Aggregate panel into pre/post signup mean spends per user.

    Args:
        panel: Raw panel with treatment effects applied.

    Returns:
        One row per user with ``pre_spends`` and ``post_spends``.
    """
    cutoff = _ANALYSIS_CUTOFF_MONTH

    def _agg(x: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "pre_spends": x.loc[x.month < cutoff, "spend"].mean(),
                "post_spends": x.loc[x.month > cutoff, "spend"].mean(),
            }
        )

    return (
        panel.groupby(["user_id", "signup_month", "treat"])
        .apply(_agg)
        .reset_index()
    )


def _add_demographics(
    data: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Attach random demographic columns.

    Args:
        data: Aggregated user-level DataFrame.
        rng: Random generator.

    Returns:
        DataFrame with ``age``, ``gender``, ``industry`` appended.
    """
    n = data["user_id"].nunique()

    data["age"] = rng.integers(18, 70, size=n)
    data["gender"] = rng.choice(["M", "F"], size=n)
    data["industry"] = rng.choice(_INDUSTRY_NAMES, size=n)
    data["treat"] = data["treat"].astype(int)
    return data


# ── Domain-specific generators ───────────────────────────────────────────────


def gen_special_medicine_df(
    data_size: int = 100,
    *,
    dependent_division: bool = True,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Generate a synthetic medical trial dataset.

    Args:
        data_size: Number of patients.
        dependent_division: If ``True``, treatment assignment depends
            on disease severity.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with ``age``, ``disease_degree``,
        ``experimental_treatment``, ``residual_lifetime``.
    """
    rng = _make_rng(random_state)

    disease_degree = rng.choice(
        [1, 2, 3, 4, 5], p=[0.3, 0.3, 0.2, 0.1, 0.1], size=data_size
    ).astype(int)

    age = rng.normal(40, scale=8, size=data_size).astype(int)
    age_effect = (age ^ 2 - 400) / 1000

    experimental_treatment = sigmoid_division(
        disease_degree, dependent_division, rng=rng
    )

    rate = 17 - 2.5 * disease_degree + experimental_treatment - age_effect
    residual_lifetime = rng.exponential(rate)

    return pd.DataFrame(
        {
            "age": age,
            "disease_degree": disease_degree,
            "experimental_treatment": experimental_treatment,
            "residual_lifetime": residual_lifetime,
        }
    )


def gen_oracle_df(
    data_size: int = 8,
    *,
    dependent_division: bool = True,
    factual_only: bool = False,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Generate an oracle dataset with factual / counterfactual outcomes.

    Args:
        data_size: Number of observations.
        dependent_division: If ``True``, feature assignment depends on
            treatment.
        factual_only: If ``True``, counterfactual outcomes are masked
            with ``NaN``.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with ``X``, ``Target_untreated``, ``Target_treated``,
        ``Treatment``, ``Target``, ``TE``.
    """
    rng = _make_rng(random_state)

    treatment = rng.binomial(1, 0.5, size=data_size)

    if dependent_division:
        target_feature = rng.binomial(1, 0.3 + 0.4 * treatment)
    else:
        target_feature = rng.binomial(1, 0.5, size=data_size)

    target_untreated = rng.uniform(300, 800, size=data_size).round(-2).astype(int)
    target_treated = target_untreated + 50 + target_feature * 100

    if factual_only:
        target_untreated = np.where(1 - treatment, target_untreated, np.nan)
        target_treated = np.where(treatment, target_treated, np.nan)

    y_factual = np.where(treatment, target_treated, target_untreated).astype(int)
    treatment_effect = target_treated - target_untreated

    return pd.DataFrame(
        {
            "X": target_feature,
            "Target_untreated": target_untreated,
            "Target_treated": target_treated,
            "Treatment": treatment,
            "Target": y_factual,
            "TE": treatment_effect,
        }
    )


def gen_control_variates_df(
    data_size: int = 1000,
    *,
    dependent_division: bool = True,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Generate a dataset for CUPED / control-variate experiments.

    Features have zero variation in the outcome, mixed with a linear
    dependency on ``X``.

    Args:
        data_size: Number of observations.
        dependent_division: If ``True``, treatment depends on the
            lagged feature.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with ``X_lag_1``, ``Target_lag_1``, ``X``,
        ``Treatment``, ``Target``.
    """
    rng = _make_rng(random_state)

    mean_x = rng.uniform(0, 5, size=data_size)
    x_lag = rng.normal(mean_x, 2)
    x_curr = rng.normal(mean_x, 2)

    treatment = sigmoid_division(x_lag, dependent_division, rng=rng)

    target_lag = 200 + x_lag * 100
    target_factual = 200 + x_curr * 100 + treatment * 10

    return pd.DataFrame(
        {
            "X_lag_1": x_lag,
            "Target_lag_1": target_lag,
            "X": x_curr,
            "Treatment": treatment,
            "Target": target_factual,
        }
    )
