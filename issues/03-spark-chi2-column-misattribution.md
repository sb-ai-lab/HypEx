## 🐛 Bug Description

<!-- A clear and concise description of what the bug is. -->

On the Spark backend, `StatsChi2TestExtension._calc_spark`
(`hypex/extensions/stats_hypothesis_testing.py:425-441`) builds the per-group result dict in
Spark `collect()` arrival order, and `StatsComparator._execute_inner_function`
(`hypex/comparators/abstract.py:937`) iterates the target columns in that dict order. But
`StatsComparator.execute` (line 1136-1140) builds the result index from `target_cols` order.
When the per-column dict order ≠ `target_cols` order, per-column results are reported under the
**wrong column labels**.

### Steps To Reproduce

<!-- Steps to reproduce the behavior: -->

```python
import os
os.environ["PYSPARK_PYTHON"] = "<venv python>"
os.environ["PYSPARK_DRIVER_PYTHON"] = "<venv python>"
import numpy as np, pandas as pd
from pyspark.sql import SparkSession
spark = SparkSession.builder.master("local[2]").config("spark.ui.enabled", "false") \
    .config("spark.sql.shuffle.partitions", "2").getOrCreate()

from hypex.dataset import Dataset, TargetRole, TreatmentRole, ExperimentData
from hypex.comparators.stats_hypothesis_testing import StatsChi2Test
from hypex.utils import BackendsEnum
from scipy.stats import chi2_contingency

rng = np.random.default_rng(0)
n = 900
g = rng.integers(0, 2, n)
df = pd.DataFrame({"g": g})
df["y1"] = np.where(df["g"] == 1, rng.choice(["a","b"], n, p=[.2,.8]), rng.choice(["a","b"], n, p=[.9,.1]))
df["y2"] = rng.choice(["a","b"], n, p=[.5,.5])   # independent of group
df["y3"] = rng.choice(["a","b"], n, p=[.4,.6])   # weakly dependent

ds = Dataset(data=df, roles={"g": TreatmentRole(), "y1": TargetRole(str), "y2": TargetRole(str),
                             "y3": TargetRole(str)}, backend=BackendsEnum.spark, session=spark)
StatsChi2Test(grouping_role=TreatmentRole()).execute(ExperimentData(ds))

# true p-values from the per-column ┆stats tables:
#   y1 -> ~7.3e-89,  y2 -> 0.7026,  y3 -> 0.2838
# but the combined table reports, under labels [y1, y2, y3]:
#   [0.2838, 0.7026, 7.3e-89]
```

The strongly significant column (y1, p≈7e-89) is reported under the label of the weak column (y3),
and vice versa. y2 happens to land correctly.

### Expected Behavior

<!-- A clear and concise description of what you expected to happen. -->

Row *i* of the combined result must correspond to `target_cols[i]`, regardless of the order in
which Spark's `collect()` returns rows.

### Screenshots

<!-- If applicable, add screenshots to help explain your problem. -->

N/A

### Environment

<!-- Please complete the following information -->

- HypEx Version: 2.0.0a1 (dev/master @ 20298227)
- Python Version: 3.11
- Operating System: macOS (Spark backend, local session; any Spark deployment)

### Additional Context

<!-- Add any other context about the problem here. -->

- The per-column `StatsChi2Test┴┴<col>┆stats` tables are keyed correctly; only the combined pairwise
  table misattributes. Any consumer that reads the combined table (e.g. ABAnalyzer on Spark) gets
  per-feature verdicts attached to the wrong features — for an A/B test this publishes decisions
  about the wrong metrics.
- The arrival order is stable for a given plan but not guaranteed by Spark, so it can change with
  partitioning/shuffle configuration.

### Possible Solution

<!-- If you have suggestions on a fix for the bug, please describe it here. -->

Either restore `target_cols` order in the extension's result dict (insert placeholders for all
`target_cols` before filling from collected rows), or iterate `target_cols` instead of the dict in
`StatsComparator._execute_inner_function` so the value order always matches the label order.

### Checklist

- [x] I have described the bug in detail
- [x] I have provided steps to reproduce
- [x] I have provided the expected behavior
- [x] I have provided screenshots (if applicable)
- [x] I have provided my environment details
- [x] I have suggested a possible solution (if applicable)
