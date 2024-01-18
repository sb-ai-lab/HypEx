from hypex.dataset.roles import MatchingRoles, AARoles, ABRoles

class Hypothesis:
    def __init__(self, name):
        self.name = self.get_name_from_str(name)
        self.attributes_for_test = self.parse_task_roles() # ?

    @staticmethod
    def get_name_from_str(name):
        if name.lower() in ['matching', 'aa', 'ab']:
            return name
        raise ValueError("Not a valid name for the experiment")
    def parse_task_roles(self):
        if self.name == 'matching':
            return MatchingRoles # list with roles?
        if self.name == 'aa':
            return AARoles
        if self.name == 'ab':
            return ABRoles