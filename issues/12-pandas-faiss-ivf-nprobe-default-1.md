> **Status: RESOLVED on `review/hidden-bugs-audit` @ `5d8d4beb` (re-checked 2026-11-02).**
> `PandasFaissExtension._fit` now computes the shared cluster parameters and assigns
> `_index.nprobe = min(self._nprobe, nlist)` at `hypex/extensions/faiss.py:360-363`.
> A runtime check on the IVF path returned `nprobe=10` with `nlist=16`. The report below is retained
> as the original reproduction and should not be filed as a new issue.

## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`PandasFaissExtension._fit` (`hypex/extensions/faiss.py:313-329`) builds an IVF index for large
datasets and **never sets `nprobe`**:

```python
_index = faiss.IndexIVFFlat(quantizer, X.shape[1], nlist)
_index.train(X)
self.index = faiss.IndexIDMap(_index)
```

FAISS's default is `nprobe = 1` (verified on the installed version), so each query searches only
**one** of `nlist` cells — the returned "nearest neighbors" are frequently far from the true nearest
neighbors. Match quality silently degrades, biasing effect estimates. The Spark path sets a tuned
`nprobe` (`faiss.py:1061, 1083`), so the two backends are numerically inconsistent for the same
inputs.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
import faiss, numpy as np

rng = np.random.default_rng(0)
X = rng.normal(size=(20000, 4)).astype("float32")
Q = rng.normal(size=(200, 4)).astype("float32")

flat = faiss.IndexFlatL2(4); flat.add(X)
D_ref, I_ref = flat.search(Q, 5)

nlist = int(np.sqrt(20000 / 4))                       # 70 — same heuristic as _fit
ivf = faiss.IndexIVFFlat(faiss.IndexFlatL2(4), 4, nlist)
ivf.train(X); ivf.add(X)
D1, I1 = ivf.search(Q, 5)                             # nprobe stays 1, as in _fit
print(np.mean([len(set(I_ref[i]) & set(I1[i])) / 5 for i in range(len(Q))]))
# recall@5 = 0.753

ivf.nprobe = 10
D10, I10 = ivf.search(Q, 5)
print(np.mean([len(set(I_ref[i]) & set(I10[i])) / 5 for i in range(len(Q))]))
# recall@5 = 1.0
```

Reachable via `Matching(faiss_mode="fast")` with > 1 000 control rows, or `faiss_mode="auto"` with
> 1 000 000 rows. The existing test `test_faiss_modes_give_same_result_on_small_data` never hits the
IVF path (its data is ≤ 1 000 rows).

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

The IVF index must set a reasonable `nprobe` (e.g. the same heuristic the Spark path uses:
`max(n_neighbors, min(max(nlist // 10, 10), 50))`), or expose it via configuration.

### Screenshots

<!-- If applicable, add screenshots to help explain your problem. -->

N/A

### Environment

<!-- Please complete the following information -->

- HypEx Version: 2.0.0a1 (dev/master @ 20298227)
- Python Version: 3.11
- Operating System: macOS (pandas backend)

### Additional Context

<!-- Add any other context about the problem here. -->

- The `nprobe = 1` default is not a FAISS guarantee surfaced to users anywhere; the degraded recall
  is fully silent (no warning, no quality flag).
- Spark faiss additionally computes `self._nprobe` in `_compute_cluster_params` — reusing that logic
  keeps backends consistent.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Set `nprobe` before `add_with_ids`/search on the pandas IVF path, mirroring
`_compute_cluster_params`/`_nprobe` from the Spark implementation; consider a parity test comparing
pandas-IVF and pandas-flat matches on a moderately large dataset.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)
