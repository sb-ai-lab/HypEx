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
            f"Can only append datas with the same backends. Got {other_backend} expected {backend}"
        )


class SpaceError(Exception):
    def __init__(self, space):
        super().__init__(f"{space} is not a valid space")


class NoColumnsError(Exception):
    def __init__(self, role):
        super().__init__(f"No columns found by role {role}")


class ComparisonNotSuitableFieldError(Exception):
    def __init__(self, group_field):
        super().__init__(f"Group field {group_field} is not suitable for comparison")


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
