from pathlib import Path

from cnf import cnf_instance_iterator


def reweight_instance(instances="mcc**"):
    for name, cnf in cnf_instance_iterator(problem_set=instances):
        out_path = Path(f"data/reweighted_{name}")
        in_path = Path(f"data/{name}")
        print(out_path)
        if out_path.exists():
            print(f"-> Already exists, skipping {in_path}")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"-> Reweighting {in_path}")
        # generate random weights
        cnf.weights.detach_()
        cnf.weights.uniform_()
        cnf.weights[:, 0] = 1 - cnf.weights[:, 1]

        with open(in_path, "r") as f_in:
            with open(out_path, "w") as out:
                for line in f_in:
                    if line.startswith('c p weight'):
                        vr = int(line.split(" ")[3])
                        if abs(vr) - 1 not in cnf.binary_vars:
                            continue
                        w = cnf.weights[abs(vr) - 1, int(vr > 0)].item()
                        out.write(f"c p weight {vr} {w:.4f}\n")
                    else:
                        out.write(line)


if __name__ == "__main__":
    reweight_instance()