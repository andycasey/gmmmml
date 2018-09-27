

from .policy import \
    (initialisation, prediction, movement, repartition, convergence)

# import DefaultInitialisationPolicy


class BayesStepper(initialisation.DefaultInitialisationPolicy,
                   prediction.DefaultPredictionPolicy,
                   movement.SingleMovementPolicy,
                   repartition.RepartitionFromNearestMixturePolicy,
                   convergence.DefaultConvergencePolicy):

    pass