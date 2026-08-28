from __future__ import annotations

ID_SPLIT_SYMBOL = "\u2534"
NAME_BORDER_SYMBOL = "\u2506"
MATCHING_INDEXES_SPLITTER_SYMBOL = "\u256f"
UTILITY_COL_SYMBOL = "\u23e3"
UTILITY_INDEX_COL_NAME = "\u23e3index"

UTILITY_INDEX_COL_NAME = "\u23e3index"
UTILITY_PHYSICAL_INDEX_COL_NAME = "\u23e3_physical_index"

UTILITY_INDEX_COL_NAME = "\u23e3index"
UTILITY_PHYSICAL_INDEX_COL_NAME = "\u23e3_physical_index"

NUMBER_TYPES_LIST = [int, float]
CATEGORICAL_TYPES_LIST = [str]

TEST_NAME_NORMALIZATION: dict[str, str] = {
    "StatsTTest": "TTest",
    "StatsKSTest": "KSTest",
    "StatsChi2Test": "Chi2Test",
    "StatsZTest": "ZTest",
    "GroupTTest": "TTest",
    "GroupKSTest": "KSTest",
    "GroupChi2Test": "Chi2Test",
    "GroupUTest": "UTest",
    "TTest": "TTest",
    "KSTest": "KSTest",
    "Chi2Test": "Chi2Test",
    "UTest": "UTest",
}