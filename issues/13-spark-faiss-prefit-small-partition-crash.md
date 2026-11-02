> **Status: OPEN — the merged fix introduces a different crash (re-checked 2026-11-02 @ `5d8d4beb`).**
> Batch buffering was added at `hypex/extensions/faiss.py:944-966`, but `np.append(cache, np_batch)`
> flattens the feature matrix. Once buffered rows are passed to `partial_fit`, scikit-learn receives
> a 1-D array. Direct execution of the current cache logic for 20 two-row, two-feature batches and
> `k=3` raises `ValueError: Expected 2D array, got 1D array instead` with cache shape `(80,)`.
> Accumulate batches in a list and use `np.concatenate(batches, axis=0)` (or initialize and append
> with `axis=0`) instead. Spark was not rerun because this environment has no Java runtime.

## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`SparkFaissExtension._prefit` (`hypex/extensions/faiss.py:855-920`) trains a `MiniBatchKMeans` with
`n_clusters = k = int(sqrt(N / 4))` by calling `partial_fit` on every batch coming out of
`rdd.mapPartitions(...).toLocalIterator()`. Batches are accumulated **per partition** (up to
`FAISS_DRIVER_INDEX_LIMIT` rows), so when Spark has many partitions, some batches contain very few
rows and `partial_fit` raises:

```
ValueError: n_samples=2 should be >= n_clusters=3.
```

(sklearn requires `n_samples >= n_clusters` on the first `partial_fit`.)

This is why `tests/unit/test_ml_faiss.py::test_spark_extension_matches_brute_force` fails on
machines where `local[*]` creates many cores (40 rows over 14 partitions → 2-row partitions), while
passing on 2-core CI runners (fewer, larger partitions). Trigger condition: number of Spark
partitions ≳ `sqrt(4N)` for the control dataset of size N — realistic for large clusters processing
moderate data.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

1. Minimal, library-free:

```python
import numpy as np
from sklearn.cluster import MiniBatchKMeans
m = MiniBatchKMeans(n_clusters=3)
m.partial_fit(np.array([[0., 0.], [1., 1.]], dtype=np.float32))
# ValueError: n_samples=2 should be >= n_clusters=3.
```

2. End-to-end: run `pytest tests/unit/test_ml_faiss.py::test_spark_extension_matches_brute_force`
   on a machine with ≥ 8 cores (`local[*]` fixture, 40-row control set). Fails with the error above.
   The same test passes with a `local[2]` session because both partitions hold ~20 rows.

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

The prefit must succeed regardless of partitioning: batches fed to `partial_fit` must contain at
least `k` rows (accumulated across partitions if necessary), and/or `k` must be capped by the
minimum viable batch size.

### Screenshots

<!-- If applicable, add screenshots to help explain your problem. -->

N/A

### Environment

<!-- Please complete the following information -->

- HypEx Version: 2.0.0a1 (dev/master @ 20298227)
- Python Version: 3.11
- Operating System: macOS (Spark backend, `local[*]` session)

### Additional Context

<!-- Add any other context about the problem here. -->

- This is currently the **only non-xfail test failure** in the unit suite
  (`pytest tests/unit tests/scenario` → 1 failed, 1687 passed), which makes the suite red on
  multi-core dev machines while CI stays green — an unfortunate combination for early regression
  detection.
- `k` is computed in `_compute_cluster_params` as `int(sqrt(_data_size / 4))`; with N = 40 → k = 3.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Buffer batches in `_prefit` until they hold ≥ `k` rows before calling `partial_fit` (flush the
remainder at the end, skipping undersized leftovers), and/or cap `k = min(k, FAISS_SAMPLE_TARGET
bound, rows of the first batch)`. A unit test that runs the Spark faiss fit with a highly fragmented
`spark.sql.shuffle.partitions` setting would cover the partitioning regime.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)
