
"""
Policies for prediction and search.
"""

import logging
import numpy as np

from .base import Policy
from .. import operations as op

logger_name, *_ = __name__.split(".")
logger = logging.getLogger(logger_name)


class BaseInitialisationPolicy(Policy):

    def __init__(self, model, K_init=1, **kwargs):

        self.model = model
        self.K_init = K_init
        return None


    def initialise(self, y):
        raise NotImplementedError("should be implemented by sub-classes")



class DefaultInitialisationPolicy(BaseInitialisationPolicy):

    def initialise(self, y, **kwargs):

        y = np.atleast_2d(y)
        N, D = y.shape

        K_inits = np.logspace(0, np.log10(N), self.K_init, dtype=int)

        for i, K in enumerate(K_inits):

            try:
                *state, R = op.kmeans_pp(y, K, **kwargs)
                yield (K, self.model.expectation_maximization(y, *state, **kwargs))

            except ValueError:
                logger.warn(f"Failed to initialise at K = {K}")
                raise
                break


        return None




