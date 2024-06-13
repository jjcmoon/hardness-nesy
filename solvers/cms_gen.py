import pycmsgen
import torch


class CmsGenAlgebra:
    """
    Weighted-like model sampling from a CNF using CMSGen.

    Cf. "Designing Samplers is Easy: The Boon of Testers" by
        Golia, Soos, Chakraborty, and Meel.
    """
    def __init__(self, nb_samples: int, unweighted: bool = False) -> None:
        self.nb_samples = nb_samples
        self.unweighted = unweighted

    def cnf_val(self, cnf):
        # convert to cmsgen format
        s = pycmsgen.Solver()
        s.add_clauses(cnf.clauses)
        if not self.unweighted:
            for v in cnf.binary_vars:
                s.set_var_weight(v+1, float(cnf.weights[v, 1].exp()))
        
        # sample models
        models = set()
        for _ in range(self.nb_samples):
            sat, _ = s.solve(time_limit=30)
            if not sat:
                break
            
            model = tuple(s.get_model())
            s.add_clause([-l for l in model])
            models.add(model)

        # print("FOUND", len(models), '/', self.nb_samples , "MODELS")
        
        # evaluate models
        total = None
        for model in models:
            result = cnf.get_lit_weights(model).sum()
            total = torch.logaddexp(total, result) if total is not None else result
        
        return total

    def get_name(self):
        return f"cmsgen_{self.nb_samples}" + ("_unweighted" if self.unweighted else "")


class WeightMeAlgebra:
    def __init__(self, nb_samples: int) -> None:
        self.nb_samples = nb_samples

    def cnf_val(self, cnf):
        # convert to cmsgen format
        s = pycmsgen.Solver()
        s.add_clauses(cnf.clauses)
        for v in cnf.binary_vars:
            s.set_var_weight(v+1, float(cnf.weights[v, 1].exp()))
        
        # sample models
        models = list()
        for _ in range(self.nb_samples):
            sat, _ = s.solve(time_limit=30)
            if not sat:
                return torch.tensor(0.0)
            
            model = tuple(s.get_model())
            # s.add_clause([-l for l in model])
            models.append(model)


        print(f"FOUND {len(models)} / {self.nb_samples} MODELS ({len(set(models))} unique)")

        # evaluate models
        total = sum(cnf.get_lit_weights(model).sum() for model in models)        
        return total

    def get_name(self):
        return f"weightme2_{self.nb_samples}"
