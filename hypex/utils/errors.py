from typing import Literal


class RoleColumnError(Exception):
    def __init__(self, roles, columns):
        super().__init__(
            "Check your roles. All of them must be names of data columns. \n"
            f"Now roles have {roles} values and columns have {columns} values"
        )


class ConcatDataError(Exception):
    def __init__(self, data_type):
        super().__init__(f"Can only append Dataset to Dataset. Got {data_type}")


class ConcatBackendError(Exception):
    def __init__(self, other_backend, backend):
        super().__init__(
            f"Can only append data with the same backends. Got {other_backend} expected {backend}"
        )


class SpaceError(Exception):
    def __init__(self, space):
        super().__init__(f"{space} is not a valid space")


class NoColumnsError(Exception):
    def __init__(self, role):
        super().__init__(f"No columns found by role {role}")


class NotSuitableFieldError(Exception):
    def __init__(self, field, field_role: Literal["Grouping", "Target", "Baseline"]):
        super().__init__(f"{field_role} field {field} is not suitable for comparison")


class NotFoundInExperimentDataError(Exception):
    def __init__(self, class_: str):
        super().__init__(f"{class_} id is not found in ExperimentData")


class AbstractMethodError(NotImplementedError):
    def __init__(self):
        super().__init__(
            "This method is abstract and will be overridden in derived class."
        )


class DataTypeError(Exception):
    def __init__(self, data_type):
        super().__init__(
            f"Can only perform the operation for Dataset and Dataset. Got {data_type}"
        )


class BackendTypeError(Exception):
    def __init__(self, other_backend, backend):
        super().__init__(
            f"Can only perform the operation with the same backends. Got {other_backend} expected {backend}"
        )


class MergeOnError(Exception):
    def __init__(self, on):
        super().__init__(f"Can only merge on one of the columns data. Got {on}")


class NoRequiredArgumentError(Exception):
    def __init__(self, argument_name):
        super().__init__(f"The required argument {argument_name} has not been passed.")


class NoneArgumentError(Exception):
    def __init__(self, arg, process):
        super().__init__(f"Argument {arg} is None in process {process}.")


class InvalidArgumentError(Exception):
    def __init__(self, arg, possible_type):
        super().__init__(
            f"Invalid type for argument {arg}, possible type is is {possible_type}."
        )


class PairsNotFoundError(Exception):
    def __init__(self):
        super().__init__(
            "Pairs are not found. Check your input data and try execute preprocessing pipeline before matching estimation."
        )
