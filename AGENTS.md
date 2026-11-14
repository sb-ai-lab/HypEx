# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this is

HypEx (2.0.0a1, alpha) is a causal-inference / A-B-testing library built as a pipeline of composable `Executor` blocks that read/write a shared `ExperimentData` state, on a backend-agnostic `Dataset` (pandas **and** Spark). Poetry, Python `>=3.8, <3.13` (no 3.13: pyspark 3.5 imports `np.NaN`), PySpark pinned to `3.5.1`.

Every subpackage of `hypex/` has its own `README.md`. Start with `hypex/README.md` (architecture overview + reading order), then `hypex/dataset/README.md`. `factory/` and `hypotheses/` are dormant — do not build on them.

## Commands

```bash
poetry install --all-extras --with dev   # setup; Spark tests also need Java 8/11/17

pytest -m "not spark"        # fast suite (~25 s; full suite ~10 min, Spark is ~96% of it)
pytest                       # everything incl. Spark
tox -- -m "not spark"        # same inside tox env (posargs after `--` reach pytest)

tox -e lint                  # ruff check + ruff format --check (read-only)
tox -e format                # ruff check --fix, then ruff format (writes files)
tox -e typing                # mypy (advisory in CI: ~239 known errors, do not "fix the build" here)
tox -e docs                  # sphinx -W; needs pandoc installed (non-Python dep)
tox -e notebooks             # nbmake over examples/tutorials + examples/advanced_tutorials
tox -e coverage              # suite with coverage; gate is fail_under=88
```

- Ruff is pinned to the same version in three places (`tox.ini`, `.pre-commit-config.yaml`, `pyproject.toml`); CI checks they agree. Update all three together.
- `ruff check --fix` must run **before** `ruff format` (a fix can leave code needing reformat). CI never runs `--fix`.
- Ignore lists live in `pyproject.toml` (`[tool.ruff.lint]`), not on the command line. E501 is enforced by format, not lint.
- `.flake8` is stale — Ruff is the linter; don't add flake8 config.
- Editing `pyproject.toml` dependencies requires re-running `poetry lock` (CI runs `poetry check --lock`).
- Local autofix: `poetry run pre-commit run --all-files`.

## Testing

- `pyproject.toml` turns `DeprecationWarning`/`FutureWarning` from `hypex.*` into **errors**. New deprecation warnings break the suite; fix the caller or update the deprecated path.
- Markers: `spark` (needs local Spark session, slow), `slow`, `catboost`. `tests/conftest.py` auto-marks anything using `spark_session`.
- On this macOS machine, Java 11 is installed via Homebrew at `/opt/homebrew/opt/openjdk@11`, but is not registered with `/usr/libexec/java_home`. Do not infer that Java is absent when `/usr/bin/java` reports "Unable to locate a Java Runtime". Set up Spark commands with:

  ```bash
  export JAVA_HOME=/opt/homebrew/opt/openjdk@11
  export PATH="$JAVA_HOME/bin:$PATH"
  ```

- Backend parity is tested via the `backend` fixture (parametrizes pandas/spark) and the `make_dataset` factory fixture — new `Dataset`/executor features should be tested on both backends with these.
- Known open bugs are encoded as `xfail(strict=True)` with a `reason` referencing `tests/KNOWN_LIBRARY_BUGS.md` (in Russian). When you fix such a bug, the strict xfail XPASSes and fails — remove the marker and update the doc. `grep -rn xfail tests` to find them.
- `unitests/` + `coverage.sh` are a legacy unittest harness — not part of CI. The real suite is `pytest` in `tests/`.

## Architecture rules that break things silently

- **Roles, not column names.** Blocks never hardcode column names; they ask `Dataset` for columns by role (`TargetRole`, `TreatmentRole`, `FeatureRole`, …).
- **Executor result ids are load-bearing.** Every result is keyed by `executor.id` = `ClassName + params_hash + key` joined by `ID_SPLIT_SYMBOL` (`┴`), with `NAME_BORDER_SYMBOL` (`┆`) separating parts inside a segment. Reporters parse these ids back apart — changing the format breaks reporting.
- **Immutability by convention:** only `Transformer`s replace `ExperimentData.ds`; everything else appends to `additional_fields` / `analysis_tables` / `variables` / `groups`.
- **New executor subclasses must satisfy `StrictABCMeta`** (`hypex/utils/strict_abc.py`): concrete method signatures are checked against the abstract declarations at class-definition time (LSP). Caveat: in files with `from __future__ import annotations` the return-type check silently doesn't work; the `test_strict_abc.py` tests deliberately omit that import.
- **Backend dispatch in extensions** goes through `BACKEND_MAPPING: dict[BackendsEnum, str]` on the extension class; the Spark backend may reuse a pandas implementation or provide its own. On Spark, KS/U tests are histogram approximations (`n_bins`), not exact scipy — numeric differences vs pandas are expected and documented.
- `factory/` and `hypotheses/` are excluded from coverage (`[tool.coverage.run]`) because they are inactive.

## Repo quirks

- Bare `pytest` is safe: `testpaths = ["tests"]` exists because examples/experiments scripts used to break collection. Keep new experiments/scripts out of `tests/`.
- Notebooks in repo root are duplicates of `examples/tutorials/`; `tox -e notebooks` skips `*CUPAC*` and `DatasetTutorial.ipynb` (known broken) and forces `--nbmake-kernel=python3` because committed kernelspecs name a nonexistent kernel.
- `examples/advanced_tutorials/` is the long-form methodology textbook (design, A/B inference, matching, custom pipelines, Spark); it is included in the default `tox -e notebooks` paths.
- `__partition_indexes_*` directories at repo root are gitignored runtime artifacts — ignore and never commit them.
- Line endings are forced to LF via `.gitattributes` (CI matrix includes Windows).
- `poetry.lock` is `merge=binary`/generated — don't hand-edit conflict hunks.

## CI shape (`.github/workflows/ci.yml`)

lint + codespell, lockfile sync, test matrix (py3.8–3.12 on Linux full-Spark; macOS/Windows smoke + Spark smoke files `tests/dataset_spark_backend/`), locked-deps coverage, docs, build. Advisory (do not gate on): `typing`, `audit`. Branches: PRs target `master` or `dev/master`. Contributing flow and PR templates: `.github/CONTRIBUTING.md`.
