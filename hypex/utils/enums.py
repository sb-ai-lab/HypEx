import enum

@enum.unique
class ExperimentDataEnum(enum.Enum):
    additional_fields = "additional_fields"
    analysis_tables = "analysis_tables"
    stats_fields = "stats_fields"

@enum.unique
class BackendsEnum(enum.Enum):
    pandas = "pandas"

@enum.unique
class SpaceEnum(enum.Enum):
    auto = "auto"
    additional = "additional"
    data = "data"
