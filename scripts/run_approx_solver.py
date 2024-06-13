from pathlib import Path
from time import time

import multiprocessing
from solvers.sampling import ReinforceFuzzyAlgebra
import torch

from cnf import cnf_instance_iterator
from solvers import (
    GodelAlgebra,
    IndecaterProductAlgebra,
    ReinforceSamplingAlgebra,
    ProductAlgebra,
    SddAlgebra,
    SemanticStrengthAlgebra,
    KOptimal,
    InterpretationSamplingAlgebra,
    StraightThroughAlgebra,
    GumbelSoftMaxAlgebra,
    CmsGenAlgebra,
    IndecaterGumbelAlgebra,
    WeightMeAlgebra
)


def solve_wrapper(solver, instance, return_dict):
    t1 = time()
    result = solver.cnf_val(instance)
    if result is None:
        return
    print("BACKWARD")
    try:
        result.backward()
    except RuntimeError as e:
        print(f"-> Failed {solver.get_name()} {e}")
        return

    grads = instance.get_grads()

    delta = time() - t1
    print(f"DONE {solver.get_name()} {delta:.2f}s")

    return_dict["result"] = result.item()
    return_dict["grads"] = grads
    return_dict["time"] = delta
    return_dict["done"] = True


def solve(mngr, solver, instance, timeout: int):
    if timeout is None:
        result = {"done": False}
        solve_wrapper(solver, instance, result)
        return result

    else:
        result = mngr.dict()
        result["done"] = False
        p = multiprocessing.Process(
            target=solve_wrapper, args=(solver, instance, result)
        )
        p.start()
        p.join(timeout)
        if not result["done"]:
            print(f"-> Killed")
            p.terminate()

    print(f"-> Solved to {result}")
    return result


def run_solver(solver, timeout: int, rerun: bool = False):
    mngr = multiprocessing.Manager()
    for name, instance in cnf_instance_iterator(problem_set="road*"):
        save_path = f"results/{name}/{solver.get_name()}.pt"
        if not rerun and Path(save_path).exists():
            print(f"-> Already solved {name}, skipping")
            continue

        result = solve(mngr, solver, instance, timeout)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        if result["done"]:
            if not torch.isnan(result["grads"]).any():
                torch.save(dict(result), save_path)
            else:
                print(f"NaN gradient: {result['grads']}")


if __name__ == "__main__":
    run_solver(InterpretationSamplingAlgebra(nb_samples=10), None)
