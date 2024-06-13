import json
import random
from cnf import CNF, cnf_instance_iterator
from nnf_to_grads import wmc_nnf, sample_nnf
from solvers import ProductAlgebra, IndecaterProductAlgebra, GodelAlgebra, StraightThroughAlgebra, GumbelSoftMaxAlgebra, \
    IndecaterGumbelAlgebra, CmsGenAlgebra, GodelGumbelAlgebra, IndecaterGodelAlgebra, CombinedAlgebra, \
    SemanticStrengthAlgebra, ReinforceFuzzyAlgebra
import torch
import time
import numpy as np
from pathlib import Path


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_cnf_params(cnf: CNF, model: list, pretrain: float):
    model = [x for x in model if abs(x) - 1 in cnf.binary_vars]  # weighted submodel
    model = np.array(model)
    sampled_model = torch.zeros_like(cnf.params, dtype=torch.bool)
    nb_pos, nb_neg = (model > 0).sum(), (model < 0).sum()
    sampled_model[model > 0] = torch.rand(nb_pos) < pretrain  # pretrain is scaled (0.5 to 1)
    sampled_model[model < 0] = torch.rand(nb_neg) > pretrain
    params = torch.zeros_like(cnf.params)
    params[sampled_model] = pretrain
    params[~sampled_model] = 1 - pretrain
    params = torch.logit(params, eps=1e-99)
    cnf.params.data = params
    cnf.params.requires_grad = True
    cnf.compute_weights()


def train(name: str, solver, max_iters: int, pretrain: float, seed: int, lr: float = 0.1):
    # load in everything
    cnf = CNF.from_dimacs(f'data/{name}')
    with open(f'results/{name}/d4.nnf') as f:
        nnf_string = f.read()
    model = np.loadtxt(f'results/{name}/model.txt', dtype=int)
    set_cnf_params(cnf, model, pretrain)
    progress = 0
    best_loss = float('-inf')

    optimizer = torch.optim.AdamW([cnf.params], lr=lr)
    deltas = []
    print(f"Starting training with {solver.get_name()} (pretain={pretrain}, lr={lr})")

    for i in range(max_iters):
        optimizer.zero_grad()

        t1 = time.time()
        loss = solver.cnf_val(cnf)
        (-loss).backward()
        deltas.append(time.time() - t1)

        if i % 10 == 0:
            progress += 1
            t1 = time.time()
            with torch.no_grad():
                exact_loss = wmc_nnf(nnf_string, cnf.weights, verbose=False).item()
            print(
                f"[{i}/{max_iters}] loss: {loss.item():.4f} real loss: {exact_loss:.4f} ({np.mean(deltas):.2f}s/it + {time.time() - t1:.2f}s)")
            deltas = []
            if exact_loss > best_loss:
                best_loss, progress = exact_loss, 0
            if exact_loss > -10 or progress > 5:
                break

        optimizer.step()
        cnf.compute_weights()
    return best_loss


def extend_explanation(cnf: CNF, model: list):
    model = set(model)
    for v in range(1, cnf.nb_vars + 1):
        if v not in model and -v not in model:
            model.add(random.choice([v, -v]))
    model = sorted(model, key=lambda x: abs(x))
    return list(model)


def pick_instances():
    names = []
    for name, cnf in cnf_instance_iterator("mcc**"):
        exact_path = f'results/{name}/exact.pt'
        exact_result = torch.load(exact_path)['result']
        nnf_path = Path(f'results/{name}/d4.nnf')
        if exact_result > -10 or cnf.nb_vars > 1000:
            continue
        if nnf_path.stat().st_size > 10_000_000:
            continue

        with open(nnf_path) as f:
            nnf_string = f.read()
            model = sample_nnf(nnf_string)
            if model is None:  # UNSAT
                continue
            print("found", model)
            model = extend_explanation(cnf, model)
            print("Saving to", f"results/{name}/model.txt")
            np.savetxt(f'results/{name}/model.txt', np.array(model, dtype=int), fmt='%i')

        names.append(name)
    print(len(names))
    print(names)


def get_instances():
    for path in Path('data').glob('**/*cnf'):
        name = str(path).replace('data/', '')
        model_path = Path(f'results/{name}/model.txt')
        if model_path.exists():
            yield name


def get_current_results(instance, solver):
    file_name = Path(f'results/{instance}/{solver.get_name()}.json')
    if file_name.exists():
        with open(file_name, 'r') as f:
            current_results = json.load(f)
    else:
        current_results = {}
    return current_results


def main(solver, pretrain: float, lr: float):
    for instance in get_instances():
        current_results = get_current_results(instance, solver)
        if f'lr={lr}_pretrain={pretrain}' in current_results:
            continue

        print("Starting training loop on", instance)
        best_loss = train(instance, solver, 10_000, pretrain=pretrain, seed=1337, lr=lr)

        # might have changed
        current_results = get_current_results(instance, solver)
        current_results[f'lr={lr}_pretrain={pretrain}'] = best_loss
        with open(f'results/{instance}/{solver.get_name()}.json', 'w') as f:
            json.dump(current_results, f)


if __name__ == "__main__":
    # pick_instances()
    solver = ReinforceFuzzyAlgebra(nb_samples=32_000)
    main(solver, pretrain=0.5, lr=0.05)

