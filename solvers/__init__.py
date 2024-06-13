from .fuzzy import ProductAlgebra, GodelAlgebra, IndecaterProductAlgebra, IndecaterGodelAlgebra
from .sdd import SddAlgebra
from .sampling import InterpretationSamplingAlgebra, ReinforceSamplingAlgebra, StraightThroughAlgebra, GumbelSoftMaxAlgebra, IndecaterGumbelAlgebra, GodelGumbelAlgebra, ReinforceFuzzyAlgebra
from .semantic_strength import SemanticStrengthAlgebra
from .kbest import KOptimal
from .cms_gen import CmsGenAlgebra, WeightMeAlgebra


class CombinedAlgebra:
    def __init__(self, a1, a2):
        self.a1 = a1
        self.a2 = a2

    def cnf_val(self, cnf):
        raise self.a1.cnf_val(cnf) + self.a2.cnf_val(cnf)
