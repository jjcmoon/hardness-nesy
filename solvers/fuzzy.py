from functools import reduce
import torch
from tqdm import tqdm

from utils import log_neg


class GodelAlgebra:
    def cnf_val(self, cnf):
        x = [clause.max() for clause in cnf.clause_iterator()]
        return torch.stack(x).min()

    def get_name(self):
        return "godel_tnorm"


class ProductAlgebra:
    def cnf_val(self, cnf):
        x = [log_neg(log_neg(clause).sum(0)) for clause in cnf.clause_iterator()]
        return torch.stack(x).sum(0)
    
    def get_name(self):
        return "product_tnorm"
    

class NilPotentAlgebra:
    def cnf_val(self, cnf):
        x = [clause.max() for clause in cnf.clause_iterator()]
        return torch.stack(x).min()
    
    def get_name(self):
        return f"nilpotent_tnorm"


class IndecaterProductAlgebra:
    def lit_val(self, lit, cnf, weight):
        lit_v = torch.full_like(cnf.weights, weight)
        if lit > 0:
            lit_v[abs(lit) - 1, :] = torch.tensor([float('-inf'), 0])
        else:
            lit_v[abs(lit) - 1, :] = torch.tensor([0, float('-inf')])
        return log_neg(lit_v)

    def clause_val(self, lits, cnf):
        clause_weights = 0
        lit_weights = cnf.get_lit_weights(lits).detach()
        for lit, weight in zip(lits, lit_weights):
            clause_weights += self.lit_val(lit, cnf, weight)
        return log_neg(clause_weights)

    def cnf_val(self, cnf):
        result = sum(self.clause_val(clause, cnf) for clause in tqdm(cnf.clauses, disable=True))
        result = torch.logsumexp(result + cnf.weights, dim=(0, 1))
        # correct for number of vars
        result = result - torch.tensor(cnf.nb_vars).log()
        return result

    def get_name(self):
        return "indecater_product_tnorm"


class IndecaterGodelAlgebra:
    def lit_val(self, lit, cnf, weight):
        lit_v = torch.full_like(cnf.weights, weight)
        if lit > 0:
            lit_v[abs(lit) - 1, :] = torch.tensor([float('-inf'), 0])
        else:
            lit_v[abs(lit) - 1, :] = torch.tensor([0, float('-inf')])
        return lit_v

    def clause_val(self, lits, cnf):
        clause_weights = torch.tensor(float('-inf'))
        lit_weights = cnf.get_lit_weights(lits).detach()
        for lit, weight in zip(lits, lit_weights):
            clause_weights = torch.maximum(self.lit_val(lit, cnf, weight), clause_weights)
        return clause_weights

    def cnf_val(self, cnf):
        result = (self.clause_val(clause, cnf) for clause in tqdm(cnf.clauses, disable=True))
        result = reduce(lambda x, y: torch.minimum(x, y), result)
        result = torch.logsumexp(result + cnf.weights, dim=(0, 1))
        # correct for number of vars
        result = result - torch.tensor(cnf.nb_vars).log()
        return result

    def get_name(self):
        return "indecater_product_tnorm"


if __name__ == "__main__":
    clauses = [[1, 2], [1, -3]]
    weights = torch.tensor([0, 0.2, 0.3, 0.4])
    algebra = IndecaterProductAlgebra()
    print(algebra.cnf_val(clauses, weights))
