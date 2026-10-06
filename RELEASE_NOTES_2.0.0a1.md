# HypEx 2.0.0a1

First alpha of HypEx 2.0. The API may still change before the stable release.

```bash
pip install --pre -U hypex
# or pin the alpha
pip install hypex==2.0.0a1
```

## Highlights

HypEx 2.0 is built on a new architecture: experiments are pipelines of small composable blocks working on a
backend-agnostic, role-tagged `Dataset`.

- **Spark backend.** `Dataset` runs on pandas or Spark with the same API for AA tests, AB tests, homogeneity tests and
  matching.
- **CUPED and CUPAC** variance reduction in `ABTest`, with a variance reduction report.
- **Multiple-testing corrections** in `ABTest` (Holm by default, plus Bonferroni, Sidak, FDR and others).
- **Matching** options to control full data extraction and index computation.

## Breaking changes

- Import paths, class names and result objects differ from 0.1.x. See the
  [tutorials](https://github.com/sb-ai-lab/HypEx/tree/master/examples/tutorials) for migration.
- Version 0.1.x is no longer supported (`pip install hypex==0.1.10`).

## Requirements

- Python `>=3.8, <3.13`.
- PySpark `3.5.1` is installed with the library; running on Spark also needs Java.
- CatBoost and LightGBM are optional: `pip install "hypex[cat]"`, `"hypex[lgbm]"`, `"hypex[all]"`.

## Known limitations

- Python 3.13 is not supported yet (PySpark 3.5 is incompatible with NumPy 2).
- Alpha status: some internal modules still carry TODOs and the public API may change.
