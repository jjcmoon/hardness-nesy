# On the Hardness of Probabilistic Neurosymbolic Learning

[[paper](https://arxiv.org/pdf/2406.04472)]

Code to the reproduce the experiments of the paper "On the Hardness of Probabilistic Neurosymbolic Learning" (ICML 2024).

> [!TIP]
> If you just want to use WeightME, take a look at [Kompyle](https://github.com/jjcmoon/kompyle).

## Installation

Install [d4](https://github.com/crillab/d4) and [EvalMaxSAT](https://github.com/FlorentAvellaneda/EvalMaxSAT), for exact compilation and MaxSat solving, respectively.
Furthermore, make sure the following python libraries are installed.
```bash
pip install numpy torch pycmsgen problog pysdd tqdm matplotlib scienceplots
```

## Data generation

The MCC benchmarks can be downloaded in dimacs format at:
- [MCC2021](https://zenodo.org/records/10012857)
- [MCC2022](https://zenodo.org/records/10012860)
- [MCC2023](https://zenodo.org/records/10012864)

Note that we only use the weighted model counting track (`track2`). The different benchmarks should be saved in the `data` folder.
The [Road-R benchmark](https://github.com/EGiunchiglia/ROAD-R/tree/main/requirements) is already included.

Note that Windows is not supported. The random weights of the MCC benchmarks are generated using
```bash
python scripts/reweight_instances.py
```


## Usage

First, generate the d-DNNF circuits with d4. (This will take a while.)
```bash
python scripts/run_d4.py
```
Next, we can evaluate these circuits in d4 to get the exact gradients. (This will take a while. For the largest circuits, 256GB RAM is necessary.)
```bash
python scripts/nnf_to_grads.py
```

Finally, we can run one of the approximate methods.
```bash
python scripts/run_approx_solver.py
```

You can select the specific method and hyperparameters at the bottom of the file.
(To run all methods and with multiple hyperparameters, its recommended to parallize this over multiple machines.)

Similarly, the training experiments can be run using
```bash
python scripts/run_training.py
```

Once all experiments have finished running, the figures and table of the paper can be generated using
```bash
python scripts/aggregate_results.py
```

## Paper

If you use WeightME in your work, consider citing our paper:

```
@inproceedings{maene2024hardness,
  title={On the Hardness of Probabilistic Neurosymbolic Learning},
  author={Maene, Jaron and Derkinderen, Vincent and De Raedt, Luc},
  booktitle={Proceedings of the 41th International Conference on Machine Learning},
  year={2024}
}
```
