import glob, os

import torch
import torch.nn.functional as F
import quimb.tensor as qtn
import problog


class CNF:
    """
    A formula in Conjunctive Normal Form
    with (probabilistic) weights on the variables.
    """

    def __init__(self, clauses: list, params: torch.Tensor, binary_vars: list, nb_vars: int) -> None:
        if not isinstance(clauses, list):
            clauses = list(clauses)
        assert torch.is_tensor(params), "params must be a tensor"

        self.clauses = clauses
        self.params = params
        self.binary_vars = binary_vars
        self.nb_vars = nb_vars
        self.compute_weights()  # (nb_vars, 2)

    def nb_params(self):
        return len(self.params)

    def nb_clauses(self):
        return len(self.clauses)

    def __str__(self) -> str:
        return f"CNF(nb_vars={self.nb_vars}, nb_clauses={self.nb_clauses()})"

    def get_grads(self):
        return self.params.grad.data

    def get_binary_weights(self):
        return self.weights[self.binary_vars, :]

    def var_to_param(self, var):
        try:
            return self.binary_vars.index(abs(var) - 1)
        except ValueError:
            return None

    def clause_iterator(self):
        for clause in self.clauses:
            yield self.get_lit_weights(clause)

    def get_lit_weights(self, lits):
        vrs = tuple(abs(lit) - 1 for lit in lits)
        pos = tuple(int(lit > 0) for lit in lits)
        return self.weights[vrs, pos]

    def partially_weighted(self):
        return self.nb_params() < self.nb_vars

    @classmethod
    def from_dimacs(self, file_path: str):
        instance = qtn.cnf_file_parse(file_path)
        weights_dict = instance["weights"]
        _validate_weights(weights_dict)
        params = _get_params(instance)
        return CNF(instance["clauses"], *params)

    def to_problog(self):
        formula = problog.cnf_formula.CNF()
        formula._clauses = self.clauses
        formula._weights = {v: float(self.weights[v, 1].exp()) for v in self.binary_vars}
        formula._clausecount = self.nb_clauses()
        formula._atomcount = self.nb_vars
        return formula

    def compute_weights(self):
        self.weights = torch.zeros(self.nb_vars, 2, dtype=torch.float64)
        self.weights[:] = torch.tensor(0.5, dtype=torch.float64).log()
        self.weights[self.binary_vars, 1] = F.logsigmoid(self.params)
        self.weights[self.binary_vars, 0] = F.logsigmoid(-self.params)


def _get_params(instance):
    nb_vars = instance["num_variables"]
    weights_dict = instance["weights"]

    # detect binary variables
    binary_vars = []
    for k in weights_dict.keys():
        if k < 0:
            continue
        if weights_dict[k] + weights_dict[-k] == 1:
            binary_vars.append(k)
        else:  # unweighted
            assert weights_dict[k] == 1 and weights_dict[-k] == 1, \
                f"Invalid weights: {weights_dict[k]}, {weights_dict[-k]}"

    params = torch.tensor([weights_dict[k] for k in binary_vars], dtype=torch.float64)
    params = torch.logit(params, eps=1e-99)
    params.requires_grad = True

    binary_vars = [k - 1 for k in binary_vars]
    return params, binary_vars, nb_vars


def _validate_weights(weights_dict):
    assert weights_dict, f"No weights found..."
    for weight in weights_dict.values():
        assert 0 <= weight <= 1, f"Invalid weight: {weight}"


def instance_path_iterator(problem_set: str):
    yield from sorted(glob.glob(f"data/{problem_set}/*cnf"))


def cnf_instance_iterator(problem_set: str = "**", only_solved: bool = True):
    for path in instance_path_iterator(problem_set):
        print("Loading", path)
        name = "/".join(path.split("/")[-2:])

        if only_solved and not os.path.exists(f"results/{name}/exact.pt"):
            print("-> Not solved, skipping")
            continue

        try:
            yield name, CNF.from_dimacs(path)
        except AssertionError as e:
            print('Loading error:', e)
            # os.remove(path)
            continue
