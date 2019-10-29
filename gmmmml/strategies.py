

from .policies import (initialisation, prediction, movement, repartition, 
                       convergence)

__all__ = ["MessageBreaking", "MessageJumping", "GreedyKMeans", "KasarapuAllison2015", "MessageBreakingNoPredictions"]


class MessageBreaking(initialisation.DefaultInitialisationPolicy,
                      prediction.NoPredictionPolicy,
                      movement.IncreasingComponentsPolicy,
                      repartition.IterativelyRepartitionFromNearestMixturePolicy,
                      convergence.DefaultConvergencePolicy):
    pass



class StrictMessageBreaking(initialisation.DefaultInitialisationPolicy,
                            prediction.NoPredictionPolicy,
                            movement.IncreasingComponentsPolicy,
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



class DebugMessageBreaking(MessageBreaking):

    def __init__(self, *args, **kwargs):

        self._message_length_of_components_split = []
        super(DebugMessageBreaking, self).__init__(*args, **kwargs)


    def repartition(self, y, K, **kwargs):

        kwds = kwargs.copy()
        kwds["debug"] = True

        K, result = super(DebugMessageBreaking, self).repartition(y, K, **kwds)
        state, R, ll, I, meta = result

        self._message_length_of_components_split.append(meta["I_components_chosen_for_split"])

        return (K, (state, R, ll, I))


class DebugKasarapuAllison2015(KasarapuAllison2015):

    def __init__(self, *args, **kwargs):
      self._message_length_of_components_split = []
      super(KasarapuAllison2015, self).__init__(*args, **kwargs)


    def repartition(self, y, K, **kwargs):

        kwds = kwargs.copy()
        kwds["debug"] = True

        K, result = super(KasarapuAllison2015, self).repartition(y, K, **kwds)
        state, R, ll, I, meta = result

        if "I_component_chosen" in meta:

            self._message_length_of_components_split.append(meta["I_component_chosen"])

            # Just for running path traces! 
            assert meta["operation"] == "split"

        return (K, (state, R, ll, I))