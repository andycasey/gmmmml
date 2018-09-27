
import logging
import numpy as np
from .base import Policy

logger_name, *_ = __name__.split(".")
logger = logging.getLogger(logger_name)

class DefaultConvergencePolicy(Policy):
    
    @property
    def converged(self):

        if len(set(self.model._state_K)) < 2:
            return False

        # Get the minimum message length from the last two K trials.
        K_all = np.array(self.model._state_K)
        I_all = np.array(self.model._state_I)


        # Unique-ify but keep order.
        _, __ = np.unique(K_all, return_index=True)
        K_unique = K_all[np.sort(__)]

        prev_K, K = K_unique[-2:]
        
        
        I = np.min(I_all[np.where(K_all == K)[0]])
        prev_I = np.min(I_all[np.where(K_all == prev_K)[0]])

        delta = I - prev_I
        
        # Stop if the current message length is more than the best from the
        # previous mixture.
        converged = delta > 0

        logger.info(f"Convergence threshold: {I:.0f} (@ K = {K}) < {prev_I:.0f}"\
                    f" (@ prev_K = {prev_K}) ({delta:.0f})")
        
        return converged
