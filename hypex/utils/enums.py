import enum

@enum.unique
class ExperimentDataEnum(enum.Enum):
    additional_fields = "additional_fields"
    analysis_tables = "analysis_tables"
    stats_fields = "stats_fields"


class BackendsEnum(enum.Enum):
    pandas = "pandas"
