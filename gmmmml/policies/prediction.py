

import logging
import numpy as np
from .base import Policy

logger_name, *_ = __name__.split(".")
logger = logging.getLogger(logger_name)


class BasePredictionPolicy(Policy):

    def predict(self, y, **kwargs):
        raise NotImplementedError("should be implemented by the sub-classes")


class DefaultPredictionPolicy(BasePredictionPolicy):

    def predict(self, y, **kwargs):

        N, D = y.shape

        # Predict a little bit ahead.
        Kp = 1 + np.arange(2 * np.max(self.model._state_K))
        return self.model._predict_message_length(Kp, N, D, **kwargs)


class NoPredictionPolicy(BasePredictionPolicy):

    def predict(self, y, **kwargs):

        return None