from .constants import ID_SPLIT_SYMBOL, NAME_BORDER_SYMBOL, TEST_NAME_NORMALIZATION

METRIC_SUFFIXES: tuple[str, ...] = (
    "control mean",
    "test mean",
    "difference %",
    "difference",
    "p-value",
    "pass",
)

def normalize_test_name(raw: str) -> str:
    """Normalize internal test class name to display name."""
    return TEST_NAME_NORMALIZATION.get(raw, raw)

# ── Metric column parser ──────────────────────────────────────────────────────

#: Multi-word metric suffixes used in composite column names.
#: Order matters: longer phrases must precede shorter ones
#: (e.g. "control mean" before "difference").

def _parse_metric_col(col: str) -> tuple[str, str, str, str]:
    """Parse ``'{feature} {TestName} {metric} {group}'`` → (feature, test, metric, group).

    Handles:
    * Multi-word metrics: ``'control mean'``, ``'test mean'``,
      ``'difference %'``, ``'p-value'``, ``'pass'``.
    * Columns containing :data:`NAME_BORDER_SYMBOL` (intermediate stats
      columns produced by :class:`StatsComparator`) — returned as four
      empty strings so callers can skip them.
    * Both legacy ``ID_SPLIT_SYMBOL``-separated and space-separated formats.

    Returns:
        A 4-tuple ``(feature, test_name, metric, group)``.  Any part that
        cannot be determined is returned as an empty string.
    """
    # Skip intermediate stats columns (e.g. "pre_spends┆stats GroupDifference mean┆...")
    if NAME_BORDER_SYMBOL in col:
        return "", "", "", ""

    # ── Legacy format: feature┆TestName┆metric┆group ──────────────
    if ID_SPLIT_SYMBOL in col:
        parts = col.split(ID_SPLIT_SYMBOL)
        if len(parts) >= 4:
            return parts[0], parts[1], parts[2], parts[3]
        if len(parts) == 3:
            return parts[0], parts[1], parts[2], ""
        return "", "", "", ""

    # ── Space-separated format ────────────────────────────────────
    parts = col.split()
    for metric in METRIC_SUFFIXES:
        metric_parts = metric.split()
        mlen = len(metric_parts)
        for i in range(len(parts) - mlen + 1):
            if parts[i : i + mlen] == metric_parts:
                before = parts[:i]
                after = parts[i + mlen :]
                if len(before) >= 2:
                    feature = before[0]
                    test = " ".join(before[1:])
                elif len(before) == 1:
                    feature = ""
                    test = before[0]
                else:
                    feature = ""
                    test = ""
                group = " ".join(after) if after else ""
                return feature, test, metric, group

    # Fallback: nothing matched
    return "", "", "", ""