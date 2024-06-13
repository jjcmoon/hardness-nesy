import random
import torch
import subprocess
import problog
import os


class KOptimal:
    def __init__(self, k: int = 1, partial: bool = False):
        self.k = k
        self.partial = partial
        self.solver = problog.maxsat.get_solver()

    def cnf_val(self, cnf):
        models = []
        for _ in range(self.k):
            model = self.solve_evalmaxsat(cnf)
            if model is None:
                return None
            models.append(model)
            cnf.clauses.append([-x for x in model])

        if not self.partial or self.k == 1:
            result = eval_models(models, cnf)
        else:
            raise NotImplementedError()
        
        return result

    def solve_maxsatz(self, cnf):
        cnf_formula = cnf.to_problog()
        solution = self.solver.evaluate(cnf_formula, partial=self.partial)
        if self.partial:
            solution = cnf_formula.from_partial(solution)
        # print("Solution", len(solution), solution)
        return solution

    def solve_evalmaxsat(self, cnf):
        dimacs = cnf.to_problog().to_dimacs(
            weighted=int, smart_constraints=True, partial=self.partial)
        tmp_file_name = f"tmp_{random.randint(0, 100000)}.cnf"
        with open(tmp_file_name, "w") as f:
            f.write(dimacs)

        cmd = f"timeout {300//self.k}s ../EvalMaxSAT/build/EvalMaxSAT_bin {tmp_file_name} --old"
        try:
            output = subprocess.check_output(cmd, shell=True)
        except subprocess.CalledProcessError:
            os.remove(tmp_file_name)
            return None
        os.remove(tmp_file_name)
        solution = process_output(output)
        if solution is None:
            return None

        if self.partial:
            solution = from_partial(solution, cnf.nb_vars)
        solution = solution[:cnf.nb_vars]
        # print("Solution", len(solution), solution)
        return solution
    
    def get_name(self):
        return f"opt_evalmaxsat_k{self.k}_p{self.partial}"
    

def process_output(output):
    output = output.decode("utf-8")
    for line in output.split("\n"):
        if line.startswith("v "):
            return list(map(int, line.split()[1:-1]))
    return None


def eval_models(models, cnf):
    results = [cnf.get_lit_weights(model).sum() for model in models]
    results = torch.logsumexp(torch.stack(results), dim=0)
    return results
    


def from_partial(atoms: list, nb_vars: int):
    """Translates a (complete) conjunction in the partial formula back to the complete formula.

    For example: given an original formula with one atom '1',
        this atom is translated to two atoms '1' (pt) and '2' (ct).

    The possible conjunctions are:

        * [1, 2]    => [1]  certainly true (and possibly true) => true
        * [-1, -2]  => [-1] not possibly true (and certainly true) => false
        * [1, -2]   => []   possibly true but not certainly true => unknown
        * [-1, 2]   => INVALID   certainly true but not possible => invalid

    :param atoms: complete list of atoms in partial CNF
    :return: partial list of atoms in full CNF
    """
    explanation = []
    atoms = set(atoms)
    for i in range(nb_vars):
        pos, neg = 2 * i + 1, 2 * i + 2
        if pos in atoms and neg in atoms:
            explanation.append(i)
        elif -pos in atoms and -neg in atoms:
            explanation.append(-i)
        elif -pos in atoms and neg in atoms:
            assert False, "INVALID"
    return explanation


