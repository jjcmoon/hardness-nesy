import random
from solvers.fuzzy import log_neg
import torch
from tqdm import tqdm


class InterpretationSamplingAlgebra:
    """
    Sampling interpretations with IndeCateR gradient estimation.
    """

    def __init__(self, nb_samples):
        self.nb_samples = nb_samples

    def clause_val(self, lits, samples):
        clause_samples = False
        for lit in lits:
            lit_samples = samples[lit - 1] if lit > 0 else ~samples[-lit - 1]
            clause_samples |= lit_samples
        return clause_samples

    def eval_clauses(self, samples, clauses):
        true_samples = torch.ones(
            samples.shape[1:], dtype=torch.bool, device=samples.device
        )
        for clause in tqdm(clauses):
            true_samples &= self.clause_val(clause, samples)
        return true_samples.float()  # .mean()

    def get_samples(self, cnf) -> torch.Tensor:
        shape = (cnf.nb_vars, cnf.nb_vars, 2, self.nb_samples)
        samples = torch.rand(shape, device=cnf.weights.device)
        weights = cnf.weights[:, 1].exp()
        print(samples.shape, weights.shape, cnf.nb_vars)
        samples = samples <= weights[:, None, None, None]

        cnf_vars = tuple(range(cnf.nb_vars))
        print(max(cnf_vars), samples.shape)
        samples[cnf_vars, cnf_vars, 0, :] = 0
        samples[cnf_vars, cnf_vars, 1, :] = 1
        return samples

    def cnf_val(self, cnf):
        samples = self.get_samples(cnf)  # (V, V, 2, S)
        true_samples = self.eval_clauses(samples, cnf.clauses)  # (V, 2, S)
        samples = true_samples.mean(-1)  # (V, 2)
        weight_tensor = cnf.weights.exp()  # (V, 2)
        result = torch.sum(samples * weight_tensor, dim=(0, 1)) / len(cnf.weights)
        return result

    def get_name(self):
        return f"sampling_indecater_{self.nb_samples}"


class ReinforceSamplingAlgebra:
    """Interpretation sampling with REINFORCE Leave-One-Out gradient estimation."""
    def __init__(self, nb_samples: int):
        self.nb_samples = nb_samples

    def cnf_val(self, cnf):
        samples = self.get_samples(cnf)  # (P, S)
        sample_probs = self.get_sample_probs(samples, cnf)  # (S,)
        samples = self.eval_clauses(samples, cnf.clauses).double()  # (S,)
        
        samples = samples - samples.mean()  # baseline correction
        result = torch.dot(sample_probs, samples)
        return result / (self.nb_samples - 1)

    def get_sample_probs(self, samples, cnf):
        sample_probs = 0
        for i in range(cnf.nb_vars):
            sample_probs += samples[i] * cnf.weights[i, 1] + ~samples[i] * cnf.weights[i, 0]
        return sample_probs

    def get_samples(self, cnf) -> torch.Tensor:
        samples = torch.rand(cnf.nb_vars, self.nb_samples)
        weights = cnf.weights[:, 1].exp()
        return samples <= weights[:, None]
    
    def eval_clauses(self, samples, clauses):
        true_samples = torch.ones(self.nb_samples, dtype=torch.bool)
        for clause in tqdm(clauses):
            true_samples &= self.clause_val(clause, samples)
        return true_samples

    def clause_val(self, lits, samples):
        clause_samples = False
        for lit in lits:
            ix = abs(lit) - 1
            clause_samples |= samples[ix] if lit > 0 else ~samples[ix]
        return clause_samples

    def get_name(self):
        return f"rloo2_{self.nb_samples//1000}k"


class StraightThroughAlgebra:
    """ Interpretation sampling with straight through gradient estimation. """
    def __init__(self, nb_samples: int = 10):
        self.nb_samples = nb_samples

    def cnf_val(self, cnf):
        samples = self.get_samples(cnf)  # (P, S)
        samples = self.eval_clauses(samples, cnf.clauses)  # (S,)
        return torch.logsumexp(samples, dim=0) - torch.tensor(cnf.nb_vars).log()

    def get_samples(self, cnf) -> torch.Tensor:
        samples = torch.rand(cnf.nb_vars, self.nb_samples)
        weights = cnf.weights.float()
        samples = samples <= weights[:, 1, None].exp()

        result = torch.zeros_like(samples, dtype=torch.float)
        result[~samples] = -1e30
        neg_params = weights[:, 0, None] * ~samples
        result += neg_params - neg_params.detach()
        pos_params = weights[:, 1, None] * samples
        result += pos_params - pos_params.detach()
        return result
    
    def eval_clauses(self, samples, clauses):
        true_samples = torch.zeros(self.nb_samples, dtype=torch.float)
        for clause in tqdm(clauses, disable=True):
            true_samples += self.clause_val(clause, samples)
        return true_samples

    def clause_val(self, lits, samples):
        clause_samples = torch.tensor(0.)
        for lit in lits:
            ix = abs(lit) - 1
            lit_value = log_neg(samples[ix]) if lit > 0 else samples[ix]
            clause_samples = clause_samples + lit_value
        return log_neg(clause_samples)

    def get_name(self):
        return f"ST_{self.nb_samples}"
    

class GumbelSoftMaxAlgebra(StraightThroughAlgebra):
    def __init__(self, tau=1, nb_samples=100) -> None:
        super().__init__(nb_samples)
        self.tau = tau
        self.gumbel = torch.distributions.Gumbel(0, 1)

    def get_samples(self, cnf) -> torch.Tensor:
        gumbels = self.gumbel.sample((cnf.nb_vars, 2, self.nb_samples))
        params = cnf.weights.float()
        samples = (gumbels + params[:, :, None]) / self.tau
        samples = torch.log_softmax(samples, dim=1)
        return samples[:, 1, :]

    def get_name(self):
        return f"gumbel_softmax_tau{self.tau}_s{self.nb_samples}"


class IndecaterGumbelAlgebra:
    def __init__(self, tau: float = 1):
        self.gumbel = torch.distributions.Gumbel(0, 1)
        self.tau = tau

    def clause_val(self, lits, weights):
        clause_weights = torch.zeros_like(weights)
        lit_weights = get_lit_weights(weights, lits)
        for lit, weight in zip(lits, lit_weights):
            neg_weight = log_neg(weight)
            clause_weights += neg_weight
            if lit > 0:
                clause_weights[abs(lit) - 1, :] += torch.tensor([-neg_weight, float('-inf')]) 
            else:
                clause_weights[abs(lit) - 1, :] += torch.tensor([float('-inf'), -neg_weight])
        return log_neg(clause_weights)

    def cnf_val(self, cnf):
        weights = self.get_samples(cnf)
        result = sum(self.clause_val(clause, weights.detach()) for clause in tqdm(cnf.clauses, disable=True))
        result = torch.logsumexp(result + cnf.weights, dim=(0, 1))
        # correct for number of vars
        result = result - torch.tensor(cnf.nb_params()).log()
        return result
    
    def get_samples(self, cnf) -> torch.Tensor:
        gumbels = self.gumbel.sample((cnf.nb_vars, 2))
        params = cnf.weights.float()
        samples = (gumbels + params) / self.tau
        samples = torch.log_softmax(samples, dim=1)
        return samples

    def get_name(self):
        return f"indecater2_gumbel_tau{self.tau}"


class GodelGumbelAlgebra:
    """ Interpretation sampling with straight through gradient estimation. """
    def __init__(self, nb_samples: int = 10, tau: float = 1):
        self.nb_samples = nb_samples
        self.tau = tau
        self.gumbel = torch.distributions.Gumbel(0, 1)

    def cnf_val(self, cnf):
        samples = self.get_samples(cnf)  # (P, S)
        samples = self.eval_clauses(samples, cnf.clauses)  # (S,)
        return torch.logsumexp(samples, dim=0) - torch.tensor(cnf.nb_vars).log()

    def get_samples(self, cnf) -> torch.Tensor:
        gumbels = self.gumbel.sample((cnf.nb_vars, 2, self.nb_samples))
        params = cnf.weights.float()
        samples = (gumbels + params[:, :, None]) / self.tau
        samples = torch.log_softmax(samples, dim=1)
        return samples[:, 1, :]
    
    def eval_clauses(self, samples, clauses):
        true_samples = torch.zeros(self.nb_samples, dtype=torch.float)
        for clause in tqdm(clauses, disable=True):
            true_samples = torch.minimum(self.clause_val(clause, samples), true_samples)
        return true_samples

    def clause_val(self, lits, samples):
        clause_samples = torch.tensor(float('-inf'))
        for lit in lits:
            ix = abs(lit) - 1
            lit_value = samples[ix] if lit > 0 else log_neg(samples[ix])
            clause_samples = torch.maximum(clause_samples, lit_value)
        return clause_samples

    def get_name(self):
        return f"Godel_gumbel_s{self.nb_samples}_tau{self.tau}"


class ReinforceFuzzyAlgebra:
    """Interpretation sampling with REINFORCE Leave-One-Out gradient estimation."""
    def __init__(self, nb_samples: int):
        self.nb_samples = nb_samples

    def cnf_val(self, cnf):
        samples = self.get_samples(cnf)  # (P, S)
        sample_probs = self.get_sample_probs(samples, cnf)  # (S,)

        total_result = torch.tensor(0.)
        true_samples = torch.ones(self.nb_samples, dtype=torch.bool)
        random.shuffle(cnf.clauses)
        restarts = 0
        for clause in cnf.clauses:
            clause_samples = self.clause_val(clause, samples) 
            new_true_samples = true_samples & clause_samples
            if torch.any(new_true_samples):
                true_samples = new_true_samples
            else:
                restarts += 1
                result = torch.dot(sample_probs, true_samples.double())
                # print("RESULT", result)
                total_result += result
                true_samples = clause_samples
        result = torch.dot(sample_probs, true_samples.double())
        total_result += result
        print("DONE WITH", restarts, "RESTARTS for", len(cnf.clauses), "clauses")
        # print(total_result)
        return total_result

    def get_sample_probs(self, samples, cnf):
        sample_probs = 0
        for i in range(cnf.nb_vars):
            sample_probs += samples[i] * cnf.weights[i, 1] + ~samples[i] * cnf.weights[i, 0]
        return sample_probs

    def get_samples(self, cnf) -> torch.Tensor:
        samples = torch.rand(cnf.nb_vars, self.nb_samples)
        weights = cnf.weights[:, 1].exp()
        return samples <= weights[:, None]
    
    def clause_val(self, lits, samples):
        clause_samples = False
        for lit in lits:
            ix = abs(lit) - 1
            clause_samples |= samples[ix] if lit > 0 else ~samples[ix]
        return clause_samples

    def get_name(self):
        return f"rloo_fuzzy_{self.nb_samples//1000}k"
    

def get_lit_weights(weights, lits):
    vrs = tuple(abs(lit) - 1 for lit in lits)
    pos = tuple(int(lit > 0) for lit in lits)
    return weights[vrs, pos]
