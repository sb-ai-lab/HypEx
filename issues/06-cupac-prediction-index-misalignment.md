## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

CUPAC predictions are paired with the wrong rows when the dataset index is not a `RangeIndex`.

`CupacExtension._predict_pandas` (`hypex/extensions/cupac.py:144-148`) wraps the model output in a
fresh frame with a default `RangeIndex`:

```python
predictions = pd.DataFrame(model.predict(X_df), columns=["predict"])
return Dataset(roles={"predict": AdditionalTargetRole()}, data=predictions)
```

`CUPACExecutor.execute` (`hypex/ml/cupac.py:403-408`) then computes
`target_cupac = data.ds[target] - (prediction - prediction.mean()) * theta`, which aligns on pandas
indexes. If `data.ds` carries a user (filtered/shuffled) index, each prediction is subtracted from
the **wrong row**.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
import numpy as np, pandas as pd
from hypex.dataset import Dataset, TargetRole, AdditionalTargetRole
from hypex.utils.cuped_theta import cuped_theta

np.random.seed(2)
n = 50
x = np.random.normal(size=n)
y = 3.0 * x + np.random.normal(scale=0.3, size=n)
perm = np.random.permutation(n)
y_ds = Dataset(roles={"y": TargetRole()}, data=pd.DataFrame({"y": y}, index=perm))
# predictions computed in row order but wrapped in a RangeIndex (as _predict_pandas does):
pred = Dataset(roles={"predict": AdditionalTargetRole()}, data=pd.DataFrame({"predict": 3.0 * x}))

theta = cuped_theta(y_ds["y"].raw_data.values.flatten(), pred.raw_data.values.flatten())
adjusted = y_ds["y"] - (pred - pred.mean()) * theta
print(np.nanvar(adjusted.raw_data["y"]))
```

Observed: variance 9.49 before "adjustment", **17.75 after** (variance increased).
With an aligned prediction index the same computation gives 0.09.

The same values differ by up to 7.0 from a `RangeIndex` run through the real executor with an
equivalent dataset.

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

The adjusted target must have substantially lower variance than the original regardless of the
dataset's index; each prediction must be paired with the row it was predicted for.

### Screenshots

<!-- If applicable, add screenshots to help explain your problem. -->

N/A

### Environment

<!-- Please complete the following information -->

- HypEx Version: 2.0.0a1 (dev/master @ 20298227)
- Python Version: 3.11
- Operating System: macOS (pandas backend; Spark path uses the same alignment downstream)

### Additional Context

<!-- Add any other context about the problem here. -->

- Note that the theta computation at `hypex/ml/cupac.py:403-406` uses `.values.flatten()` on both
  sides and is therefore row-order-correct; only the final subtraction at lines 407-408 is
  index-aligned and broken.
- `CUPEDTransformer` is unaffected (both operands share the same index).
- Full CUPAC end-to-end is currently blocked by a separate known issue (xfail
  `test_cupac_runs_end_to_end`); this bug will surface immediately once that is fixed, and already
  affects direct use of `CupacExtension`.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Build the prediction frame on the feature frame's index:
`pd.DataFrame(model.predict(X_df), columns=["predict"], index=X_df.index)`. Adding a regression test
that runs CUPAC on shuffled-index data and asserts variance decreases would lock the fix in.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)
