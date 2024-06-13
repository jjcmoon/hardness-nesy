import os, time
import random

import torch
from tqdm import tqdm

from cnf import cnf_instance_iterator


def eval_nnf(nnf_string, weights, solver_name):
    if solver_name == "exact":
        return wmc_nnf(nnf_string, weights)
    elif solver_name == "mpe":
        return mpe_nnf(nnf_string, weights)
    else:
        raise NotImplementedError()


def wmc_nnf(nnf_string, weights, verbose=True) -> float:
    ONE = torch.tensor(0., dtype=torch.float64)
    ZERO = torch.tensor(-1e300, dtype=torch.float64)

    lines = [s.split(" ")[:-1] for s in nnf_string.split("\n")]
    nodes = [None]
    for line in tqdm(lines, desc="WMC NNF", disable=not verbose):
        if not line:
            continue
        if line[0] == "o" or line[0] == "f":
            nodes.append([ZERO, line[0]])
        elif line[0] == "a" or line[0] == "t":
            nodes.append([ONE, line[0]])
        else:
            source, target, *literals = [int(x) for x in line]
            if len(literals) == 0:
                lits_val = nodes[target][0]
            else:
                ix1 = [abs(lit) - 1 for lit in literals]
                ix2 = [int(lit > 0) for lit in literals]
                lit_weights = weights[ix1, ix2]
                lits_val = nodes[target][0] + lit_weights.sum()

            if nodes[source][1] == 'o':
                nodes[source][0] = torch.logaddexp(nodes[source][0], lits_val)
            elif nodes[source][1] == 'a':
                nodes[source][0] = nodes[source][0] + lits_val
    return nodes[1][0]


def mc_nnf(nnf_string, verbose=True) -> float:
    lines = [s.split(" ")[:-1] for s in nnf_string.split("\n")]
    nodes = [None]
    for line in tqdm(lines, desc="WMC NNF", disable=not verbose):
        if not line:
            continue
        if line[0] == "o" or line[0] == "f":
            nodes.append([0, line[0]])
        elif line[0] == "a" or line[0] == "t":
            nodes.append([1, line[0]])
        else:
            source, target, *literals = [int(x) for x in line]
            lits_val = nodes[target][0]

            if nodes[source][1] == 'o':
                nodes[source][0] += lits_val
            elif nodes[source][1] == 'a':
                nodes[source][0] *= lits_val
    return nodes[1][0]


class Explanation:
    def __init__(self, content) -> None:
        self.content = content

    def __and__(self, other):
        if self.content is None or other.content is None:
            return None
        return Explanation(self.content | other.content)

    def __or__(self, other):
        if self.content is None:
            return other
        elif other.content is None:
            return self
        return random.choice([self, other])

    def add(self, x):
        if self.content is None:
            return self
        return Explanation(self.content | set(x))

    @classmethod
    def zero(self):
        return Explanation(None)

    @classmethod
    def one(self):
        return Explanation(set())


def sample_nnf(nnf_string, verbose=True) -> float:
    lines = [s.split(" ")[:-1] for s in nnf_string.split("\n")]
    nodes = [None]
    for line in tqdm(lines, desc="SAMPLE NNF", disable=not verbose):
        if not line:
            continue
        if line[0] == "o" or line[0] == "f":
            nodes.append([Explanation.zero(), line[0]])
        elif line[0] == "a" or line[0] == "t":
            nodes.append([Explanation.one(), line[0]])
        else:
            source, target, *literals = [int(x) for x in line]
            lits_val = nodes[target][0].add(literals)
            if nodes[source][1] == 'o':
                nodes[source][0] |= lits_val
            elif nodes[source][1] == 'a':
                nodes[source][0] &= lits_val
    if nodes[1][0].content is None:
        return None
    return tuple(sorted(nodes[1][0].content, key=lambda x: abs(x)))


def mpe_nnf(nnf_string, weights) -> float:
    ONE = torch.tensor(0., dtype=torch.float64)
    ZERO = torch.tensor(-1e300, dtype=torch.float64)

    lines = [s.split(" ")[:-1] for s in nnf_string.split("\n")]
    nodes = [None]
    for line in tqdm(lines, desc="MPE NNF"):
        if not line:
            continue
        if line[0] == "o" or line[0] == "f":
            nodes.append([ZERO, line[0]])
        elif line[0] == "a" or line[0] == "t":
            nodes.append([ONE, line[0]])
        else:
            source, target, *literals = [int(x) for x in line]
            if len(literals) == 0:
                lits_val = nodes[target][0]
            else:
                ix1 = [abs(lit) - 1 for lit in literals]
                ix2 = [int(lit > 0) for lit in literals]
                lit_weights = weights[ix1, ix2]
                lits_val = nodes[target][0] + lit_weights.sum()

            if nodes[source][1] == 'o':
                nodes[source][0] = torch.maximum(nodes[source][0], lits_val)
            elif nodes[source][1] == 'a':
                nodes[source][0] = nodes[source][0] + lits_val
    return nodes[1][0]


def generate_exact_grads(solver_name: str, max_len: int = None):
    for name, cnf in cnf_instance_iterator(problem_set="reweight*", only_solved=False):
        nnf_path = f"results/{name}/d4.nnf"
        save_path = f"results/{name}/{solver_name}.pt"
        print("Loading", nnf_path)
        if not os.path.exists(nnf_path):
            print("-> Not solved, skipping")
            continue
        if os.path.exists(save_path):
            print("-> Already exists, skipping")
            continue

        with open(nnf_path) as f:
            nnf = f.read()
        if len(nnf) == 0 or (max_len is not None and max_len < len(nnf)):
            print("-> Skipping", name, "bad size")
            continue

        t1 = time.time()
        result = eval_nnf(nnf, cnf.weights, solver_name)
        try:
            result.backward()
        except RuntimeError as e:
            print("WARNING: skipping", name, e)
            continue

        delta = time.time() - t1
        grads = cnf.get_grads()
        print(grads)
        assert not torch.any(torch.isnan(grads))
        if solver_name == "exact" and torch.all(grads == 0):
            print("Zero gradient, skipping...")
            continue

        results = {"grads": grads, "result": result.item(), "time": delta}
        print(f"RESULT {result.item():.4g}")
        torch.save(results, save_path)


if __name__ == "__main__":
    generate_exact_grads("exact", None)
