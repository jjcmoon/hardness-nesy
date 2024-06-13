from train import get_instances
from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import scienceplots


plt.style.use('science')


def parse_str(s):
    s = [x.split('=') for x in s.split("_")]
    return {k: float(v) for k, v in s}


name_trans = {
    "product_tnorm_unweighted_lr0.03": 'Product t-norm',
    "ST_100_lr0.03": "Straight-through estimator (s=100)",
    "gumbel_softmax_tau0.1_s1000_lr0.03": "Gumbel-Softmax (s=1k, $\\tau$=0.1)",
    "rloo_fuzzy_10k_lr0.03": "Fuzzy + SFE (s=10k)",
    "indecater_product_tnorm_lr0.03": "CatLog + Product t-norm",
    "indecater2_gumbel_tau1_lr0.03": "CatLog + Gumbel-Softmax (s=1, $\\tau$=1)",
    "godel_tnorm_lr0.03": "Gödel t-norm",
}

def main(pretrain=0.5):
    print(f"### RESULTS (pretrain={pretrain})")
    matrix = defaultdict(list)
    for i, name in enumerate(get_instances()):
        for result_file in Path(f"results/{name}").glob("*.json"):
            method_name = result_file.stem
            with open(result_file, 'r') as f:
                for k, v in json.load(f).items():
                    k = parse_str(k)
                    if k['pretrain'] == pretrain:
                        matrix[f"{method_name}_lr{k['lr']}"].append(v)

        # if i > 3:
        #     break


    for name, values in matrix.items():
        values = np.array(list(sorted(values, reverse=True)))
        values = np.minimum(values, -10)
        print(f"{name:<35} {np.mean(values):.2f} +- {np.std(values):.2f}  (n={len(values)})")
        if name in name_trans:
            plt.plot(values, label=name_trans[name])
    plt.plot([-10] * len(values), label="Exact", linestyle="--")
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.xlim(0, len(values)-1)
    plt.xlabel("Instances")
    plt.ylabel("Log-likelihood")
    plt.savefig(f"results/plot_pretrain{pretrain}.pdf")
    plt.clf()


if __name__ == "__main__":
    main(0.5)
    main(0.9)