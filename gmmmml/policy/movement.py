
import logging
import numpy as np
from .base import Policy

logger_name, *_ = __name__.split(".")
logger = logging.getLogger(logger_name)


class BaseMovementPolicy(Policy):
    pass



class DefaultMovementPolicy(BaseMovementPolicy):

    def move(self, y, **kwargs):

        K_failures = []

        # Keep moving until the convergence policy says we should stop.
        while not self.converged:

            # Predict before we move.
            Kp, I, I_var, I_lower = self.predict(y)

            # TODO: Can we use a restricted bayes optimisation (eg within bounds
            # of K)

            """
            # Calculate the acquisition function.
            chi = (min_I - I) / np.sqrt(I_var)
            Phi = 0.5 * (1.0 + erf(chi / np.sqrt(2)))
            phi = np.exp(-0.5 * chi**2) / np.sqrt(2 * np.pi * I_var)
            K_ei = (min_I - I) * Phi + I_var * phi
            """

            idx = np.argsort(I)
            K_nexts = Kp[idx]

            # Unique-ify but keep order.
            _, __ = np.unique(K_nexts, return_index=True)
            K_nexts = K_nexts[np.sort(__)]

            for K in K_nexts:
                if K not in self.model._state_K and K not in K_failures: break

            else:
                # No predictions!
                logger.warn("MovementPolicy has no new places to move to."\
                            "Convergence may not be reached.")
                break

            logger.info(f"Moving to K = {K}")
            yield K

        return None



class SingleMovementPolicy(DefaultMovementPolicy):

    def move(self, y, **kwargs):

        # Make a prediction using the DefaultMovementPolicy, but then just move
        # one step in that direction.
        for K in super(SingleMovementPolicy, self).move(y):

            diff = np.array(self.model._state_K) - K
            abs_diff = np.abs(diff)

            K_nearest = self.model._state_K[np.argmin(abs_diff)]
            K_actual = K_nearest + 1 if K > K_nearest else K_nearest - 1

            logger.info(f"Overriding movement policy. Moving to K = {K_actual}")

            yield K_actual




