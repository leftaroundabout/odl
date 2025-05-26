from .new_weighting import Weighting

import array_api_compat.torch as xp

class PytorchWeighting(Weighting):
    def __init__(self, **kwargs):
        Weighting.__init__(self, **kwargs)
    
    @property
    def array_namespace(self):
        return xp
    
    @property
    def impl(self):
        return 'pytorch'
    