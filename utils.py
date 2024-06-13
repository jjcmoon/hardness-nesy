import torch


def lits_to_ix(lits):
    vrs = tuple(abs(lit) - 1 for lit in lits)
    pos = tuple(int(lit > 0) for lit in lits)
    return vrs, pos


def log_neg(x, eps=1e-7):
    return torch.log1p(-torch.exp(x - eps))
