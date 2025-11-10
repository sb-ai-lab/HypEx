import enum



@enum.unique
class ExperimentDataEnum(enum.Enum):
    variables = "variables"
    additional_fields = "additional_fields"
    analysis_tables = "analysis_tables"
    groups = "groups"
    ml = "ml"


@enum.unique
class BackendsEnum(enum.Enum):
    pandas = "pandas"


@enum.unique
class SpaceEnum(enum.Enum):
    auto = "auto"
    additional = "additional"
    data = "data"


@enum.unique
class ABNTestMethodsEnum(enum.Enum):
    bonferroni = "bonferroni"
    sidak = "sidak"
    holm_sidak = "holm-sidak"
    holm = "holm"
    simes_hochberg = "simes-hochberg"
    hommel = "hommel"
    fdr_bh = "fdr_bh"
    fdr_by = "fdr_by"
    fdr_tsbh = "fdr_tsbh"
    fdr_tsbky = "fdr_tsbky"
    quantile = "quantile"


@enum.unique
class ABTestTypesEnum(enum.Enum):
    t_test = "t-test"
    ks_test = "ks-test"
    u_test = "u-test"
    chi2_test = "chi2-test"


@enum.unique
class RenameEnum(enum.Enum):
    all = "all"
    columns = "columns"
    index = "index"