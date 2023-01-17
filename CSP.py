from copy import deepcopy
from random import choice, uniform, sample, choices
from operator import methodcaller, attrgetter
from collections import deque


class Variable:

    def __init__(self, domain: list, value=None, name = None) -> None:
        if type(domain) is not str:
            my_domain = domain
        else:
            my_domain = str(domain).split()[0]
        self.__domain = list(my_domain)
        self.__value = None
        self.__name = name
        if len(self.__domain) == 1:
            self.assign(self.__domain[0])
        if value is not None:
            self.assign(value)

    @classmethod
    def from_names_to_equal_domain(cls, names: set, domain: list, value=None) -> dict:
        name_to_variable_map = dict()
        for name in names:
            # if value is not None:
            #     name_to_variable_map[name] = cls(domain, value)
            # else:
            name_to_variable_map[name] = cls(domain,value,name)
        return name_to_variable_map

    def __bool__(self) -> bool:
        return self.__value is not None

    def unique_assignment(self) -> bool:
        return len(self.__domain) == 1

    def __get_domain(self) -> list:
        return self.__domain
    def __set_domain(self, domain: set) -> None:
        self.__domain = domain
    domain = property(__get_domain, __set_domain)

    def __get_value(self):
        return self.__value
    value = property(__get_value)

    def __get_name(self):
        return self.__name
    name = property(__get_name)

    def assign(self, value):
        if self.__value is not None:
            raise OverAssignmentError(self)
        if value not in self.__domain:
            raise UncontainedValueError(self, value)
        self.__value = value

    def unassign(self):
        self.__value = None

    # def remove_from_domain(self, value):
    #     self.__domain.remove(value)

    def __str__(self) -> str:
        name =""
        if self.__name: name = str(self.__name)
        return name + "  (value: " + str(self.value) + "\t domain: " + str(self.__domain) + ")"
    def __repr__(self) -> str:
        return self.__str__()

class VariableError(Exception):
    """ Base class for various Variable Errors. """

class UncontainedValueError(VariableError):
    def __init__(self, variable: Variable, value):
        msg = "Cannot assign variable: " + str(variable) + " with value: " + str(value) + \
              " since it is not contained in variable's domain."
        super(UncontainedValueError, self).__init__(msg)

class OverAssignmentError(VariableError):
    def __init__(self, variable: Variable):
        msg = "Over-assignment of an assigned variable: " + str(variable) + \
              ". variable must be unassigned before assignment."
        super(OverAssignmentError, self).__init__(msg)


class Constraint:
    def __init__(self, variables: list, evaluate_constraint):
        """ evaluate_constraint: a function  tuple -> bool"""
        self.__variables = tuple(variables)

        # if len(self.__variables) != len(frozenset(self.__variables)):
        #     self.__variables = list()
        #     seen_variables = set()
        #     for var in variables:
        #         if var not in seen_variables:
        #             self.__variables.append(var)
        #             seen_variables.add(var)
        #     self.__variables = tuple(self.__variables)

        self.__evaluate_constraint = evaluate_constraint
        self.__i_consistent_assignments = set()
        # if len(self.__variables) == 1:
        #     self.__enforce_unary_constraint()

    # def __enforce_unary_constraint(self):
    #     variable, *_ = self.__variables
    #     variable.domain = list(self.get_consistent_domain_values(variable))

    def __get_variables(self):
        return self.__variables
    variables = property(__get_variables)

    # @classmethod
    # def from_domains(cls, evaluate_constraint, *domains):
    #     variables = list()
    #     for elem in domains:
    #         try:
    #             if type(elem) is str:
    #                 raise TypeError
    #             elem_iterator = iter(elem)
    #             variables.append(Variable(elem))
    #         except TypeError:
    #             last_variable = variables[-1]
    #             if elem in last_variable.domain:
    #                 last_variable.assign(elem)
    #     return cls(variables, evaluate_constraint)

    def __bool__(self) -> bool:
        # if self.__i_consistent_assignments:
        #     return all(self.__variables) and self.is_consistent() and self.__is_i_consistent_assignment()
        return all(self.__variables) and self.is_consistent()

    __value_getter = attrgetter("value")

    def is_consistent(self) -> bool:
        assigned_variables = []
        for var in self.__variables:
            if var:
                assigned_variables.append(var)
        # all_values = map(Constraint.__value_getter, self.__variables)
        # values_of_assigned_variables = tuple(filter(None.__ne__, all_values))
        if self.__i_consistent_assignments:
            return self.__evaluate_constraint(assigned_variables) and self.__is_i_consistent_assignment()
        return self.__evaluate_constraint(assigned_variables)
        # if self.__i_consistent_assignments:
        #     return self.__evaluate_constraint(self.__variables) and self.__is_i_consistent_assignment()
        # return self.__evaluate_constraint(self.__variables)

    def get_consistent_domain_values(self, variable: Variable) -> set:
        if variable not in self.__variables:
            raise UncontainedVariableError(self, variable)

        if variable.unique_assignment():
            return variable.domain

        original_value = variable.value
        variable.unassign()
        consistent_domain = set()
        for value in variable.domain:
            variable.assign(value)
            if self.is_consistent():
                consistent_domain.add(value)
            variable.unassign()

        if original_value is not None and variable.domain:
            variable.assign(original_value)
        return consistent_domain

    def update_i_consistent_assignments(self, i_consistent_assignments: set) -> None:
        if not i_consistent_assignments:
            self.__i_consistent_assignments.add(frozenset())
        for assignment in i_consistent_assignments:
            self.__i_consistent_assignments.add(frozenset(assignment))

    def __is_i_consistent_assignment(self) -> bool:
        all_values = map(Constraint.__value_getter, self.__variables)
        current_assignment = set(filter(None.__ne__, all_values))
        for assignment in self.__i_consistent_assignments:
            if assignment.issubset(current_assignment):
                return True
        return False

    def __str__(self) -> str:
        state = "\n  constraint is completely assigned: " + str(all(self.__variables)) + \
                ". constraint is consistent: " + str(self.is_consistent()) + ". constraint is satisfied: " + \
              str(bool(self)) + ". ]\n"
        return "[ " + "\n  ".join(map(str, self.variables)) + state

class ConstraintError(Exception):
    """ Base class for various Constraint Errors. """
class UncontainedVariableError(ConstraintError):
    def __init__(self, constraint: Constraint, variable: Variable):
        msg = "Cannot return consistent domain of " + str(variable) + " since variable is not contained in\n" \
              + str(constraint) + "variables."
        super(UncontainedVariableError, self).__init__(msg)


class ConstraintProblem:

    # __is_consistent_method_caller = methodcaller("is_consistent")

    def __init__(self, constraints):
        self.__constraints = tuple(constraints)
        self.__variables_to_constraints_map = _build_variables_to_constraints_mapping(self.__constraints)
        self.__constraint_graph = _build_constraint_graph_as_adjacency_list(self.__variables_to_constraints_map)
        # self.__name_to_variable_map = name_to_variable_map
        # if name_to_variable_map is not None:
        #     assert frozenset(self.__name_to_variable_map.values()).issubset(self.get_variables()), \
        #         "name_to_variable_map.values() is not a subset of the variables given in constraints. "

    # def get_name_to_variable_map(self):
    #     return self.__name_to_variable_map

    # def is_completely_unassigned(self) -> bool:
    #     return not any(self.__variables_to_constraints_map.keys())

    def is_completely_assigned(self) -> bool:
        return all(self.__variables_to_constraints_map.keys())

    def is_consistently_assigned(self) -> bool:
        is_consistent_results = map(methodcaller("is_consistent"), self.__constraints)
        return all(is_consistent_results)

    # def is_completely_consistently_assigned(self) -> bool:
    #     return all(self.__constraints)

    def get_variables(self):
        return self.__variables_to_constraints_map.keys()

    # def get_assigned_variables(self):
    #     assigned_variables = filter(None, self.__variables_to_constraints_map.keys())
    #     return frozenset(assigned_variables)

    def get_unassigned_variables(self):
        unassigned_variables = []
        for var in self.__variables_to_constraints_map.keys():
            if not var:
                unassigned_variables.append(var)
        return unassigned_variables

    def get_neighbors(self, variable: Variable):
        return self.__constraint_graph[variable]

    # def get_assigned_neighbors(self, variable: Variable):
    #     assigned_neighbors = filter(None, self.__constraint_graph[variable])
    #     return frozenset(assigned_neighbors)

    def get_unassigned_neighbors(self, variable: Variable):
        unassigned_neighbors = []# filterfalse(None, self.__constraint_graph[variable])
        for const in self.__constraint_graph[variable]:
            if not const:
                unassigned_neighbors.append(const)
        return tuple(unassigned_neighbors)

    def get_constraints(self):
        return self.__constraints

    # def get_consistent_constraints(self):
    #     consistent_constraints = filter(methodcaller("is_consistent"), self.__constraints)
    #     return frozenset(consistent_constraints)

    # def get_inconsistent_constraints(self):
    #     inconsistent_constraints = [] #filterfalse(methodcaller("is_consistent"), self.__constraints)
    #     for const in self.__constraints:
    #         if not const.isconsistent():
    #             inconsistent_constraints.append(const)
    #     return frozenset(inconsistent_constraints)

    # def get_satisfied_constraints(self):
    #     satisfied_constraints = filter(None, self.__constraints)
    #     return frozenset(satisfied_constraints)

    def get_unsatisfied_constraints(self):
        unsatisfied_constraints = [] #filterfalse(None, self.__constraints)
        for const in self.__constraints:
            if not const:
                unsatisfied_constraints.append(const)
        return frozenset(unsatisfied_constraints)

    # def get_constraints_containing_variable(self, variable: Variable):
    #     return frozenset(self.__variables_to_constraints_map[variable])

    def get_consistent_domain(self, variable: Variable) -> set:
        consistent_domains = map(methodcaller("get_consistent_domain_values", variable),
                                 self.__variables_to_constraints_map[variable])
        return set.intersection(*consistent_domains)

    # def get_current_assignment(self):
    #     return {variable: variable.value for variable in self.__variables_to_constraints_map.keys()}

    # def unassign_all_variables(self, read_only_variables = None):
    #     if read_only_variables is None:
    #         for variable in self.__variables_to_constraints_map.keys():
    #             variable.unassign()
    #     else:
    #         for variable in self.__variables_to_constraints_map.keys():
    #             if variable not in read_only_variables:
    #                 variable.unassign()

    # def assign_variables_from_assignment(self, assignment):
    #     for variable in self.__variables_to_constraints_map.keys():
    #         if assignment[variable] is not None:
    #             variable.assign(assignment[variable])
    #         else:
    #             variable.unassign()

    # def assign_variables_with_random_values(self, read_only_variables = None, action_history = None):
    #     for variable in self.__variables_to_constraints_map.keys():
    #         if read_only_variables is None or variable not in read_only_variables:
    #             value = choice(variable.domain)
    #             variable.assign(value)
    #             if action_history is not None:
    #                 action_history.append((variable, value))
    #     return action_history

    # def get_constraint_graph_as_adjacency_list(self):
    #     return self.__constraint_graph

    # def add_constraint(self, constraint: Constraint):
    #     new_constraints = self.__constraints | {constraint}
    #     self.__variables_to_constraints_map = _build_variables_to_constraints_mapping(new_constraints)
    #     self.__constraint_graph = _build_constraint_graph_as_adjacency_list(self.__variables_to_constraints_map)

    def __str__(self):
        state = "\n  constraint_problem is completely assigned: " + str(all(self.__variables_to_constraints_map)) + \
                ". constraint_problem is consistent: " + str(self.is_consistently_assigned()) + \
                ". constraint_problem is satisfied: " + str(all(self.__constraints)) + ". }\n"
        return "{ " + "\n  ".join(map(str, self.__constraints)) + state

def _build_variables_to_constraints_mapping(constraints: list) -> dict:
    """ returns a dictionary   var : list(constraints it is in) """
    variables_to_constraints_map = dict()  #dict(set)
    for const in constraints:
        for var in const.variables:
            try:
                variables_to_constraints_map[var].add(const)
            except:
                variables_to_constraints_map[var] = set()
                variables_to_constraints_map[var].add(const)
    return variables_to_constraints_map

def _build_constraint_graph_as_adjacency_list(variables_to_constraints_map: dict) -> dict:
    """ returns a dictionary   var : set(variables that share constraints with var) """
    constraints_graph = dict()
    for variable in variables_to_constraints_map:
        for constraint in variables_to_constraints_map[variable]:
            try:
                constraints_graph[variable].update(constraint.variables)
            except:
                constraints_graph[variable] = set()
                constraints_graph[variable].update(constraint.variables)
        constraints_graph[variable].discard(variable)
    return constraints_graph


def least_constraining_value(constraint_problem: ConstraintProblem, variable: Variable) -> list:
    unassigned_neighbors = constraint_problem.get_unassigned_neighbors(variable)

    def neighbors_consistent_domain_lengths(value) -> int:
        variable.assign(value)
        consistent_domain_lengths = map(lambda neighbor: len((constraint_problem.get_consistent_domain(neighbor))),
                                        unassigned_neighbors)
        variable.unassign()
        return sum(consistent_domain_lengths)

    return sorted(constraint_problem.get_consistent_domain(variable), key=neighbors_consistent_domain_lengths,
                  reverse=True)


def minimum_remaining_values(constraint_problem: ConstraintProblem, variables = None):
    if variables is not None:  # then we're using minimum_remaining_values as secondary key
        min_variable = min(variables, key=lambda variable: len(constraint_problem.get_consistent_domain(variable)))
        return frozenset({min_variable})

    unassigned_variables = constraint_problem.get_unassigned_variables()
    min_variable = min(unassigned_variables, key=lambda var: len(constraint_problem.get_consistent_domain(var)))
    min_remaining_values = len(constraint_problem.get_consistent_domain(min_variable))
    min_variables = filter(lambda var: len(constraint_problem.get_consistent_domain(var)) == min_remaining_values,
                           unassigned_variables)
    return list(min_variables)

def degree_heuristic(constraint_problem: ConstraintProblem, variables):
    if variables is not None:  # then we're using degree_heuristic as secondary key
        max_variable = max(variables, key=lambda var: len(constraint_problem.get_unassigned_neighbors(var)))
        return max_variable

__actions_history = deque()

def classic_heuristic_backtracking_search(constraint_problem: ConstraintProblem, with_history: bool = False):
    __actions_history.clear()
    __classic_heuristic_backtrack(constraint_problem, with_history)
    if with_history:
        return __actions_history

def __classic_heuristic_backtrack(constraint_problem: ConstraintProblem, with_history: bool = False) -> bool:
    if constraint_problem.is_completely_assigned():
        if constraint_problem.is_consistently_assigned():
            return True
        return False
    unassigned_variables = constraint_problem.get_unassigned_variables()
    max_r = len(unassigned_variables)
    if max_r == 1: n_choice = 1
    else: n_choice = choice(range(1,max_r))
    selected_unassigned_vars = set(choices(unassigned_variables,k = n_choice))
    selected_variable = degree_heuristic(constraint_problem, selected_unassigned_vars)

    selected_domain = constraint_problem.get_consistent_domain(selected_variable)
    for value in selected_domain:
        selected_variable.assign(value)
        if with_history:
            __actions_history.append((selected_variable, value))

        if __classic_heuristic_backtrack(constraint_problem, with_history):
            return True

        selected_variable.unassign()
        if with_history:
            __actions_history.append((selected_variable, None))

    return False


# def always_satisfied(values: tuple) -> bool:
#     return True

# def never_satisfied(values: tuple) -> bool:
#     return False

# def all_diff_constraint_evaluator(values: tuple) -> bool:
#     seen_values = set()
#     for val in values:
#         if val in seen_values:
#             return False
#         seen_values.add(val)
#     return True

