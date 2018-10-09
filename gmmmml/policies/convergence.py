
import logging
import numpy as np
from .base import Policy

logger_name, *_ = __name__.split(".")
logger = logging.getLogger(logger_name)

class BaseConvergencePolicy(Policy):

    @property
    def converged(self):
        raise NotImplementedError("this should be implemented by sub-classes")




class DefaultConvergencePolicy(BaseConvergencePolicy):
    
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


class ConvergedWithSuccessivelyWorseIterations(BaseConvergencePolicy):

    @property
    def converged(self):

        N = 5

        T = len(set(self.model._state_K)) - self.model._num_initialisations
        if T < N:
            return False

        ml = lambda I: I if isinstance(I, float) else np.sum(np.hstack(I.values()))

        # Check to see if the last X iterations were worse.
        K = np.hstack(self.model._results.keys())[-N:]
        I = np.hstack([ml(self.model._results[k][-1]) for k in K])

        min_I = np.min(self.model._state_I)
        converged = np.all(I > min_I)

        return converged

