
from .base import Policy

class DefaultConvergencePolicy(Policy):
    
    @property
    def converged(self):
        # HACK TESTING
        return max(self.model._state_K) > 40
