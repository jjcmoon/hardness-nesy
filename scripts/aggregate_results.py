import glob, os
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')


def compare_grads(v1, v2):
    assert not torch.isnan(v1).any() and not torch.isnan(v2).any()
    if v1.norm(1) == 0 or v2.norm(1) == 0:
        return torch.nan

    v1 = v1 / v1.norm(1)
    v2 = v2 / v2.norm(1)
    sim = torch.cosine_similarity(v1, v2, dim=0)
    return sim.item()


def load_results(path, unsafe=True):
    if unsafe:
        return torch.load(path)
    try:
        return torch.load(path)
    except Exception as e:
        print(e)
        # corrupted results, delete the file
        os.remove(path)


def main(exact_name="exact.pt"):
    metrics = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(f"results/*/*cnf"):
        exact_path = path + "/" + exact_name
        if not os.path.exists(exact_path):
            continue

        problem_set = path.split("/")[-2]
        exact_results = load_results(exact_path)
        if exact_results is None or 'grads' not in exact_results:
            continue

        weightme_path = Path(path + "/" + 'weightme2_100.pt')
        if not weightme_path.exists() or load_results(weightme_path)['time'] > 300:
            continue

        exact_grads = exact_results["grads"]
        if torch.all(exact_grads == 0) or torch.any(torch.isnan(exact_grads)):
            print("WARNING: removing invalid instance", path)
            os.remove(exact_path)
            continue

        for approx_path in glob.glob(path + "/*.pt"):
            if approx_path.endswith(exact_name):
                continue

            solver_name = approx_path.split("/")[-1][:-3]
            approx_results = load_results(approx_path)

            if approx_results is None or 'grads' not in approx_results:
                continue

            if approx_results['time'] > 300:
                continue  # time-out
            sim = compare_grads(approx_results['grads'], exact_grads)
            metrics[problem_set][solver_name].append(sim)

    return metrics


def print_metrics(metrics, latex=False):
    mm = defaultdict(dict)
    print("### RESULTS ###")
    for problem_set, problem_metrics in sorted(metrics.items()):
        print(problem_set)
        for solver_name, solver_metrics in sorted(problem_metrics.items()):
            safe_metrics = np.nan_to_num(solver_metrics)
            mean, std = safe_metrics.mean(), safe_metrics.std()
            nans = np.isnan(solver_metrics).sum()
            n = len(solver_metrics)
            mm[solver_name][problem_set] = [mean, std, n]
            nanmean = np.nanmean(solver_metrics) if nans != n else 0.
            q1, q2, q3 = np.quantile(safe_metrics, [0., 0.5, 1.])

            print(
                f"  {solver_name: <30}: {mean:.3f} ± {std:.3f}   (n={n: <2}, nanmean={nanmean:.3f}, nans={nans: <2}, min={q1:.3f}, q2={q2:.3f}, max={q3:.3f})"
            )

    if latex:
        latex_pprint(mm)


problem_trans = {
    'mcc2021': 'MCC2021',
    'mcc2022': 'MCC2022',
    'mcc2023': 'MCC2023',
    'road-r': "ROAD-R",
}
solver_trans = {
    # 'mpe': 'MPE',
    'godel_tnorm': 'Gödel t-norm',
    'product_tnorm_unweighted': 'Product t-norm',
    "indecater_product_tnorm": "IndeCateR + Product t-norm",
    "rloo2_10k": "SFE (s=10k)",
    'explain_sample_1000': 'Explanation sampling (s=1k)',
    'semantic_strength_k=10': 'Semantic strengthening ($\\kappa$=10)',
    'opt_evalmaxsat_k1_pFalse': 'MPE',
    'cmsgen_100': 'Weighted model sampling (k=100)',
    'cmsgen_100_unweighted': 'Uniform model sampling (k=100)',
    "indecater2_gumbel_tau1": "IndeCateR + Gumbel (s=1, $\\tau$=1)",
    'gumbel_softmax_tau2_s10': 'Gumbel (s=10, $\\tau$=2)',
    'ST_10': 'Straight-through estimator (s=10)',
    'rloo_fuzzy_10k': 'SFE+Product t-norm (s=10k)',
}


def latex_pprint(mm):
    print("Latex table:")
    # print table header
    print("\\begin{tabular}{l", end="")
    for _ in mm["godel_tnorm"]:
        print(" c", end="")
    print("}")
    print('\\toprule')

    # print solvers header
    for problem_set, _ in sorted(mm["godel_tnorm"].items()):
        print(f" & {problem_trans[problem_set]}", end="")
    print(r" \\")
    print("\\midrule")

    # print number of instances
    print("Nb of Instances", end="")
    for _, problem_metrics in sorted(mm['godel_tnorm'].items()):
        print(f" & {problem_metrics[-1]:d}", end="")
    print(r" \\")
    print("\\midrule")

    # print body
    for solver_name, problem_metrics in sorted(mm.items()):
        if solver_name not in solver_trans:
            continue
        print(solver_trans[solver_name], end="")
        for problem_set, (mean, std, n) in sorted(problem_metrics.items()):
            if n != mm['godel_tnorm'][problem_set][-1]:
                print(" & -", end="")
            else:
                print(f" & {mean:.3f} ± {std:.3f}", end="")
        print(r" \\")
    print('\\bottomrule')
    print("\\end{tabular}")


def cactus_plot(plot_list):
    timings = defaultdict(list)
    for path in glob.glob(f"results/*/*cnf"):
        problem_set = path.split("/")[-2]
        if "mcc" not in problem_set:
            continue

        for approx_path in glob.glob(path + "/*.pt"):
            if approx_path.endswith("exact.pt"):
                continue

            solver_name = approx_path.split("/")[-1][:-3]
            approx_results = load_results(approx_path)
            if approx_results is None or 'time' not in approx_results:
                continue
            timings[solver_name].append(approx_results['time'])

    plt.figure(figsize=(4, 3))
    for solver_name, ts in timings.items():
        if solver_name not in plot_list:
            continue
        ts = np.array(list(sorted(ts)))
        ts = np.cumsum(ts)
        ts = ts / 60
        label = solver_trans[solver_name]
        label = " ".join([x for x in label.split() if not x.startswith("(")])
        plt.plot(ts, label=label)
    plt.legend(ncol=1)
    plt.xlabel("Number of instances solved")
    plt.ylabel("Cumulative time (minutes)")
    plt.yscale("log")
    plt.gca().set_ylim(bottom=1e-4, top=150)
    plt.xlim(0, len(timings['godel_tnorm']) - 1)
    plt.savefig("cactus.pdf", bbox_inches='tight')


if __name__ == "__main__":
    print_metrics(main())
    plot_list = [
        'product_tnorm_unweighted',
        'cmsgen_100',
        'cmsgen_100_unweighted',
        'semantic_strength_k=10',
        'opt_evalmaxsat_k1_pFalse',
    ]
    # cactus_plot(plot_list)
