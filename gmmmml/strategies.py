

from .policies import (initialisation, prediction, movement, repartition, 
                       convergence)

__all__ = ["MessageBreaking", "MessageJumping", "GreedyKMeans", "KasarapuAllison2015"]


class MessageBreaking(initialisation.DefaultInitialisationPolicy,
                      prediction.DefaultPredictionPolicy,
                      movement.StepTowardsMMLMixtureMovementPolicy,
                      repartition.IterativelyRepartitionFromNearestMixturePolicy,
                      convergence.DefaultConvergencePolicy):
    pass


class StrictMessageBreaking(initialisation.DefaultInitialisationPolicy,
                            prediction.DefaultPredictionPolicy,
                            movement.StepTowardsMMLMixtureMovementPolicy,
                            repartition.IterativelyRepartitionFromNearestMixturePolicy,
                            convergence.StrictConvergencePolicy):
    pass


class MessageJumping(initialisation.DefaultInitialisationPolicy,
                     prediction.LookaheadFromInitialisationPredictionPolicy,
                     movement.ConvergeToMixtureWithMaximumExpectationImprovementPolicy,
                     repartition.RepartitionMixtureUsingKMeansPP):
    
    # note -- no convergence policy because it is governed by movement.ConvergeToMixtureWithMaximumExpectationImprovementPolicy

    def __init__(self, *args, **kwargs):
        super(MessageJumping, self).__init__(*args, **kwargs)
        self.meta.update(K_init=10)
        return None
  

class GreedyKMeans(initialisation.DefaultInitialisationPolicy,
                   prediction.DefaultPredictionPolicy,
                   movement.IncreasingComponentsPolicy,
                   repartition.GreedilyPerturbNearestMixturePolicy,
                   convergence.DefaultConvergencePolicy):
    pass


class KasarapuAllison2015(initialisation.DefaultInitialisationPolicy,
                          prediction.NoPredictionPolicy,
                          movement.GreedyMovementPolicy,
                          repartition.GreedilyPerturbNearestMixturePolicy,
                          convergence.DefaultConvergencePolicy):
    pass