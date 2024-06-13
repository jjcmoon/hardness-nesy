from collections import defaultdict
import torch
from torch import Tensor
import torch.nn.functional as F
import itertools
from tqdm import tqdm

from solvers.sdd import SddAlgebra, eval_sdd
from pysdd.sdd import SddManager


class SemanticStrengthAlgebra:
    """
    Semantic strenghtening.
    Cf. "Semantic Strengthening of Neuro-Symbolic Learning"
        by K. Ahmed et al. (2023)
    """

    def __init__(self, k: int):
        self.k = k
        self.sdd_algebra = SddAlgebra()

    def get_name(self):
        return f"semantic_strength_k={self.k}"

    def eval_queue(self, queue, weights):
        result = 0
        for clause_sdd in queue:
            result += eval_sdd(self.sdd_algebra.sdd_manager, clause_sdd, weights)
        print("RESULT", result)
        return result

    def merge_queue(self, queue, var_map, weights):
        weights = weights.detach()
        pwmi = []
        possible_combinations = get_possible_combos(queue, var_map)
        for c1, c2 in tqdm(possible_combinations):
            if var_map[c1].isdisjoint(var_map[c2]):
                continue
            mi = mutual_information(c1, c2, weights, self.sdd_algebra.sdd_manager)
            pwmi.append([*mi, set([c1, c2])])

        queue = get_new_queue(pwmi, self.k, queue)
        return queue

    def cnf_val(self, cnf):
        self.sdd_algebra.sdd_manager = SddManager(var_count=cnf.nb_vars)

        # Construct sdd queue
        sdd_queue = []
        var_map = dict()
        for clause in cnf.clauses:
            clause_sdd = self.sdd_algebra.clause_val(clause)
            clause_sdd.ref()
            sdd_queue.append(clause_sdd)
            var_map[clause_sdd] = {abs(lit) for lit in clause}

        sdd_queue = self.merge_queue(sdd_queue, var_map, cnf.weights)
        return self.eval_queue(sdd_queue, cnf.weights)


def get_possible_combos(constraints, var_map):
    """
    Returns a list with all the possible combinations of
    constraints, sharing at least one variable.
    """
    constraint_map = defaultdict(set)
    for c in constraints:
        for v in var_map[c]:
            constraint_map[v].add(c)
    combos = set()
    for cs in constraint_map.values():
        for c1, c2 in itertools.combinations(cs, 2):
            combos.add((c1, c2))
    return combos
    

def get_new_queue(pwmi, k, constraints):
    """
    Taken from: https://github.com/UCLA-StarAI/Semantic-Strengthening
    """

    # Merge the K constraints with highest MI
    pwmi = sorted(pwmi, key=lambda tup: tup[0], reverse=True)
    to_merge = pwmi[:k]

    # Consolidate common constraints
    out = []
    while len(to_merge) > 0:
        curr, *rest = to_merge

        lcurr = -1
        while len(curr[-1]) > lcurr:
            lcurr = len(curr[-1])

            rest_ = []
            for other in rest:
                if not curr[-1].isdisjoint(other[-1]):
                    # print("Consolidated")
                    curr[-2] |= other[-2]
                    curr[-1] |= other[-1]
                else:
                    rest_.append(other)
            rest = rest_

        out.append(curr)
        to_merge = rest

    to_merge = out
    print(len(to_merge))

    # Sanity-check
    for i in range(len(to_merge)):
        for j in range(len(to_merge)):
            if i != j:
                assert (to_merge[i][-1].isdisjoint(to_merge[j][-1]))

    # Dereference unused constraints
    for elem in pwmi[k:]:
        for e in elem[1]: e.deref()
    
    # A set to track the constraints to be removed
    to_remove = set()
    for _, joint, indiv in to_merge:

        # if we have decided to model c1 and c2 jointly,
        # and c1 and c3 jointly, then we conjoin the two
        # constraints to get a single constraint for
        # c1 & c2 & c3
        first, *rest = joint
        for r in rest:
            old_first = first
            first = first & r
            first.ref()
            old_first.deref()

        # append the resulting constraint the list of independent
        # constraints
        constraints.append(first)

        # Update the constraints to remove
        to_remove |= indiv

    for c in to_remove:
        constraints.remove(c)
        c.deref()

    return constraints
    


def mutual_information(c1, c2, lit_weights, sdd_manager):
    """
    Mutual information between two sdd formulas.
    Taken from: https://github.com/UCLA-StarAI/Semantic-Strengthening
                 c2            -c2
          +-----------+------------+
          |           |            |
     c1   |     a     |      b     |
          |           |            |
          +------------------------+
          |           |            |
    -c1   |     c     |      d     |
          |           |            |
          +-----------+------------+
    """
    #TODO
    #if cache[c1+c2]:
    # Conjoin the two constraints
    c1_c2 = c1 & c2
    c1_c2.ref()

    # marginals
    p_c1 = eval_sdd(sdd_manager, c1, lit_weights)[None] #a+b
    p_c2 = eval_sdd(sdd_manager, c2, lit_weights)[None] #a+c

    # Calculate the probabilities: a, b, c, and d
    p_c1_c2 = eval_sdd(sdd_manager, c1_c2, lit_weights)[None] #a
    p_c1_nc2 = logsubexp(p_c1, p_c1_c2) #b
    p_nc1_c2 = logsubexp(p_c2, p_c1_c2) #c
    tmp = torch.stack((p_c1_c2, p_c1_nc2,  p_nc1_c2), dim=-1).logsumexp(dim=-1).clamp(max=-1.1920928955078125e-07)
    p_nc1_nc2 = log1mexp(-tmp) #d


    p_c1 = [log1mexp(-p_c1), p_c1]
    p_c2 = [log1mexp(-p_c2), p_c2]
    p_c1_c2 = [[p_nc1_nc2, p_nc1_c2],[p_c1_nc2, p_c1_c2]]

    mi = 0.0
    for x, y in itertools.product([0,1], repeat=2):
        a = p_c1_c2[x][y].exp()
        b = p_c1_c2[x][y] - (p_c1[x] + p_c2[y])
        mi += xty(a, b)

        if torch.any(torch.isnan(mi)) or torch.any(torch.isinf(mi)):
            import pdb; pdb.set_trace()
            print("Crap! nan or inf")

    mi.clamp(min=0)
    return (mi.mean().item(), set([c1_c2]))


@torch.jit.script
def log1mexp(x):
    lt = (x < 0.6931471805599453094).logical_and(x > 0)
    gt = x >= 0.6931471805599453094
    res = torch.empty_like(x)
    res[lt] = torch.log(-torch.expm1(-x[lt]))
    res[gt] = torch.log1p(-torch.exp(-x[gt]))
    res = res.masked_fill_(x == 0, -float('inf'))
    return res


@torch.jit.script
def logsubexp(x, y):
    delta = torch.where((x == y) & (x.isfinite() | (x < 0)), torch.zeros_like(x - y), (x-y).abs())
    return torch.max(x, y) + log1mexp(delta)


@torch.jit.script
def logsumexp(tensor: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    with torch.no_grad():
        m, _ = torch.max(tensor, dim=dim, keepdim=True)
        m = m.masked_fill_(torch.isneginf(m), 0.)

    z = (tensor - m).exp_().sum(dim=dim, keepdim=True)
    mask = z == 0
    z = z.masked_fill_(mask, 1.).log_().add_(m)
    z = z.masked_fill_(mask, -float('inf'))

    if not keepdim:
        z = z.squeeze(dim=dim)
    return z.clamp(max=-1.1920928955078125e-07)


@torch.jit.script
def xty(x, y):
    return torch.where(x == 0, torch.zeros_like(x), x * y)