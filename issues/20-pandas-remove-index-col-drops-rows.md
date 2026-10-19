## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

`PandasDataset.remove_index_col` (`hypex/dataset/backends/pandas_backend.py:1039-1040`) drops
**rows**, not the column:

```python
def remove_index_col(self, index_col_name: str | None = UTILITY_INDEX_COL_NAME):
    self.data = self.data.drop(index_col_name)      # default axis=0 -> rows!
```

Its counterpart `add_index_col` (lines 1036-1037) adds a **column**, so `remove_index_col` can never
succeed after `add_index_col` — it raises `KeyError: "['⏣index'] not found in axis"` (pandas looks
for the label in the row index).

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
import pandas as pd
from hypex.dataset.backends.pandas_backend import PandasDataset

b = PandasDataset(data=pd.DataFrame({"x": [1, 2, 3]}))
b.add_index_col()
print(b.data.columns.tolist())     # ['x', '⏣index']
b.remove_index_col()
# KeyError: "['⏣index'] not found in axis"
```

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

`remove_index_col` removes the utility index **column**, restoring the frame to its pre-
`add_index_col` shape.

### Screenshots

<!-- If applicable, add screenshots to help explain your problem. -->

N/A

### Environment

<!-- Please complete the following information -->

- HypEx Version: 2.0.0a1 (dev/master @ 20298227)
- Python Version: 3.11
- Operating System: macOS (pandas backend; the Spark backend has no such method)

### Additional Context

<!-- Add any other context about the problem here. -->

- Latent: no in-repo callers today, but it is public backend API and the asymmetry with
  `add_index_col` makes it an obvious trap.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

```python
self.data = self.data.drop(columns=[index_col_name])
```

Add a round-trip test (`add_index_col` → `remove_index_col` → columns/roles unchanged).

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)
