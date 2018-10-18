

import logging
import numpy as np
from .base import Policy

logger_name, *_ = __name__.split(".")
logger = logging.getLogger(logger_name)


class BasePredictionPolicy(Policy):
    
    def __init__(self, *args, **kwargs):
        super(BasePredictionPolicy, self).__init__(*args, **kwargs)
    

    def predict(self, y, **kwargs):
        raise NotImplementedError("should be implemented by the sub-classes")


class DefaultPredictionPolicy(BasePredictionPolicy):

    def predict(self, y, **kwargs):

        N, D = y.shape

        # Predict a little bit ahead.
        Kp = 1 + np.arange(2 * np.max(self.model._state_K))

        logger.info("Predicting between K = {0} and K = {1}".format(Kp[0], Kp[-1]))
        
        K, I, I_var, I_lower = self.model._predict_message_length(Kp, N, D, **kwargs)

        K_min = K[np.argmin(I)]
        logger.info(f"Predicted minimum message length at K = {K_min}")

        return (K, I, I_var, I_lower)



class LookaheadFromInitialisationPredictionPolicy(BasePredictionPolicy):

    def __init__(self, *args, **kwargs):
        super(LookaheadFromInitialisationPredictionPolicy, self).__init__(*args, **kwargs)


    def predict(self, y, **kwargs):
        """
        Predict the message length only up to the K value that was trialled
        during the initialisation procedure.
        """

        N, D = y.shape
        """
        K_inits = np.logspace(0, np.log10(N/2.0), self.meta["K_init"], dtype=int)
        K_max = K_inits[1 + self.model._num_initialisations]
        """
        K_max = int(np.ceil(1.5 * [*self.model._results][self.model._num_initialisations - 1]))


        Kp = np.arange(1, 1 + K_max).astype(int)
        logger.info("Predicting between K = {0} and K = {1}".format(Kp[0], Kp[-1]))
        
        K, I, I_var, I_lower = self.model._predict_message_length(Kp, N, D, **kwargs)

        K_min = K[np.argmin(I)]
        logger.info(f"Predicted minimum message length at K = {K_min}")

        return (K, I, I_var, I_lower)
            


class NoPredictionPolicy(BasePredictionPolicy):



    def predict(self, y, **kwargs):

        return None