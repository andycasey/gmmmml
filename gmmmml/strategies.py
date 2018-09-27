

from .policies import (initialisation, prediction, movement, repartition, 
                       convergence)

__all__ = ["BayesStepper", "BayesJumper", "GreedyKMeans", "KasarapuAllison2015"]


class BayesStepper(initialisation.DefaultInitialisationPolicy,
                   prediction.DefaultPredictionPolicy,
                   movement.StepTowardsMMLMixtureMovementPolicy,
                   repartition.IterativelyRepartitionFromNearestMixturePolicy,
                   convergence.DefaultConvergencePolicy):
    pass


class BayesJumper(initialisation.DefaultInitialisationPolicy,
                  prediction.DefaultPredictionPolicy,
                  movement.MoveTowardsMMLMixtureMovementPolicy,
                  repartition.SimultaneousRepartitionFromNearestMixturePolicy,
                  convergence.ConvergedWithSuccessivelyWorseIterations):
    pass


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