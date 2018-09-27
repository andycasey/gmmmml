



import logging
import numpy as np
from .base import Policy
from ..operations import iteratively_operate_components

logger_name, *_ = __name__.split(".")
logger = logging.getLogger(logger_name)


class BaseRepartitionPolicy(Policy):

    def repartition(self, y, K, **kwargs):
        raise NotImplementedError("should be implemented by sub-classes")



class RepartitionFromNearestMixturePolicy(BaseRepartitionPolicy):
    
    def repartition(self, y, K, **kwargs):

        # Get nearest mixture.
        K_trialled = np.hstack(self.model._results.keys())
        diff = np.abs(K - K_trialled)

        index = K_trialled[np.argmin(diff)]
        state, R, ll, I = self.model._results[index]
        
        return (K, iteratively_operate_components(y, *state, K, **kwargs))
