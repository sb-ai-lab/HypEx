class RoleError(Exception):
    def __init__(self, role, task):
        super().__init__("Invalid role {} for task {}".format(role, task))

class MethodError(Exception):
    def __init__(self, func_name, data_type):
        super().__init__("No implementation of {} for dataset {}".format(func_name, data_type))
