class RoleError(Exception):
    def __init__(self, role, task):
        super().__init__("Invalid role {} for task {}".format(role, task))
