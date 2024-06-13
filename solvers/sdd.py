from functools import reduce

import torch
from pysdd.iterator import SddIterator
from pysdd.sdd import SddManager, SddNode
from tqdm import tqdm


class SddAlgebra:
    def __init__(self):
        self.sdd_manager = None

    def disjoin_all(self, sdds):
        return reduce(self.sdd_manager.disjoin, sdds)

    def clause_val(self, lits):
        literals = [self.sdd_manager.literal(lit) for lit in lits]
        return self.disjoin_all(literals)

    def cnf_val(self, cnf):
        self.sdd_manager = SddManager(var_count=cnf.nb_vars)
        cnf_sdd = self.sdd_manager.true()
        for clause in tqdm(cnf.clauses):
            cnf_sdd &= self.clause_val(clause)
        return eval_sdd(self.sdd_manager, cnf_sdd, cnf.weights)

    def wmc(self, sdd, weights):
        return eval_sdd(self.sdd_manager, sdd, weights)

    def get_name(self):
        return "sdd"


def eval_sdd(manager: SddManager, root_node: SddNode, weights):
    iterator = SddIterator(manager, smooth=False)

    def _formula_evaluator(node: SddNode, r_values, *_):
        if node is not None:
            if node.is_literal():
                return weights[abs(node.literal) - 1, int(node.literal > 0)]
            elif node.is_true():
                return torch.tensor(0.0)
            elif node.is_false():
                return torch.tensor(float("-inf"))
        # Decision node
        return torch.logsumexp(torch.stack([v[0] + v[1] for v in r_values]), dim=0)

    result = iterator.depth_first(root_node, _formula_evaluator)
    return result
