

import logging
import numpy as np
from collections import defaultdict
from tqdm import tqdm

from .base import Policy
from .. import operations as op

logger_name, *_ = __name__.split(".")
logger = logging.getLogger(logger_name)





class BaseRepartitionPolicy(Policy):

    def repartition(self, y, K, **kwargs):
        raise NotImplementedError("should be implemented by sub-classes")



class IterativelyRepartitionFromNearestMixturePolicy(BaseRepartitionPolicy):
    
    def repartition(self, y, K, **kwargs):

        # Get nearest mixture.
        state, R, ll, I = _nearest_mixture(self.model._results, K)
        return (K, op.iteratively_operate_on_components(y, *state, K, **kwargs))


class RepartitionMixtureUsingKMeansPP(BaseRepartitionPolicy):

    def repartition(self, y, K, **kwargs):
        return (K, op.kmeans_pp(y, K, **kwargs))


class SimultaneousRepartitionFromNearestMixturePolicy(BaseRepartitionPolicy):

    def repartition(self, y, K, **kwargs):

        # Get nearest mixture.
        state, R, ll, I = _nearest_mixture(self.model._results, K)
        result = op.simultaneously_operate_on_components(y, *state, K, **kwargs)
        
        return (K, result)

        

class GreedilyPerturbNearestMixturePolicy(BaseRepartitionPolicy):


    def repartition(self, y, K=None, **kwargs):

        # Get nearest (or most recent) mixture.
        if K is None:
            K = list(self.model._results.keys())[-1]
            state, R, ll, prev_I = self.model._results[K]

        else:
            state, R, ll, prev_I = _nearest_mixture(self.model._results, K)
            K = _K_from_state(state)

        # Exhaustively try all perturbations.
        best_perturbations = defaultdict(lambda: [np.inf])

        # TODO: Prevent things gonig into _results unless we have the full dictionary of message lengths
        ml = lambda I: I if isinstance(I, float) else np.sum(np.hstack(I.values()))

        tqdm_format = lambda f: None if kwargs.get("quiet", False) else f

        # Exhaustively split all components.
        for k in tqdm(range(K), desc=tqdm_format(f"Splitting K={K}")):
            p = op.split_component(y, *state, R, k, **kwargs)
            
            # Keep best split component.
            I = ml(p[-1])
            if I < best_perturbations["split"][0]:
                best_perturbations["split"] = [I, k] + list(p)

        if K > 1:
            # Exhaustively delete all components.
            for k in tqdm(range(K), desc=tqdm_format(f"Deleting K={K}")):
                p = op.delete_component(y, *state, R, k, **kwargs)
                
                # Keep best delete component.
                I = ml(p[-1])
                if I < best_perturbations["delete"][0]:
                    best_perturbations["delete"] = [I, k] + list(p)

            # Exhaustively merge all components.
            for k in tqdm(range(K), desc=tqdm_format(f"Merging K={K}")):
                p = op.merge_component(y, *state, R, k, **kwargs)
                
                # Keep best merged component.
                I = ml(p[-1])
                if I < best_perturbations["merge"][0]:
                    best_perturbations["merge"] = [I, k] + list(p)


        bop, bp = min(best_perturbations.items(), key=lambda x: x[1][0])
        logger.debug(f"Best operation is {bop} on index {bp[1]}")

        if ml(bp[-1]) >= ml(prev_I):
            logger.info("All perturbations are worse.")
            return (K, (state, R, ll, I))

        _, __, state, R, ll, I = bp

        K = _K_from_state(state)
        
        return (K, (state, R, ll, I))


def _K_from_state(state):
    return len(state[-1])

def _nearest_mixture(results, K):
    K_trialled = np.hstack(results.keys())
    diff = np.abs(K - K_trialled)

    index = K_trialled[np.argmin(diff)]
    return results[index]

    