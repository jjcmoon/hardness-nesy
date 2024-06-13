import os, glob
from pathlib import Path

def run_d4(instances="*", d4_path="../d4/d4"):
    for in_path in sorted(glob.glob(f"data/{instances}/*cnf")):
        name = "/".join(in_path.split("/")[-2:])
        out_path = Path(f"results/{name}/d4.nnf")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            print(f"-> Already exists, skipping {name}")
            continue

        print(f"\n-> Running d4 on {name}")
        os.system(f"timeout 300s {d4_path} -dDNNF \"{in_path}\" -out={out_path}")


if __name__ == "__main__":
    run_d4("road-r")
