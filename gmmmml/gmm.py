
"""
Model data with a mixture of gaussians.
"""

import logging
import numpy as np
import scipy

from . import (em, mml)

logger = logging.getLogger(__name__)


class GaussianMixture(object):

    r"""
    Model data from many multivariate Gaussian distributions, using minimum 
    message length (MML) as the objective function.

    :param covariance_type: [optional]
        The structure of the covariance matrix for individual components.
        The available options are: `full` for a free covariance matrix, or
        `diag` for a diagonal covariance matrix (default: ``full``).

    :param covariance_regularization: [optional]
        Regularization strength to add to the diagonal of covariance matrices
        (default: ``0``).

    :param threshold: [optional]
        The relative improvement in message length required before stopping an
        expectation-maximization step (default: ``1e-5``).

    :param max_em_iterations: [optional]
        The maximum number of iterations to run per expectation-maximization
        loop (default: ``10000``).

    :param information_bound: [optional]
        Whether or not to use a strict calculation for the lower bound on the
        predictions for the sum of the log of the determinant of the covariance
        matrices.

        If 'strict' is employed, then the pair-wise distances between all points
        will be calculated, and this matrix will be used when predicting the
        entropy of future gaussian mixtures. This can be computationally
        expensive for large data sets.

        If 'loose' is employed, then no pair-wise distances will be calculated
        and the logarithm of the determinant of the covariance matrices will
        be approximated by a Gaussian kernel density estimator. This approach
        is computationally efficient and reasonaly accurate, but may result in
        poor or uncertain predictions when the variance of components in the 
        mixture vary by several orders of magnitude.
    """

    parameter_names = ("mean", "covariance", "weight")

    def __init__(self, covariance_type="full", covariance_regularization=0, 
        threshold=1e-3, max_em_iterations=1000, information_bound="strict",
        **kwargs):

        available = ("full", )
        covariance_type = covariance_type.strip().lower()
        if covariance_type not in available:
            raise ValueError("covariance type '{}' is invalid. "\
                             "Must be one of: {}".format(
                                covariance_type, ", ".join(available)))

        if 0 > covariance_regularization:
            raise ValueError(
                "covariance_regularization must be a non-negative float")

        if 0 >= threshold:
            raise ValueError("threshold must be a positive value")

        if 1 > max_em_iterations:
            raise ValueError("max_em_iterations must be a positive integer")

        available = ("strict", "loose")
        information_bound = information_bound.strip().lower()
        if information_bound not in available:
            raise ValueError("information bound type '{}' is invalid. Must be "\
                             "one of: {}".format(
                                information_bound, ", ".join(available)))

        self._threshold = threshold
        self._max_em_iterations = max_em_iterations
        self._covariance_type = covariance_type
        self._covariance_regularization = covariance_regularization
        self._information_bound = information_bound

        # Lists to record states for predictive purposes.
        self._state_K = []
        self._state_det_covs = []
        self._state_weights = []
        self._state_sum_log_likelihoods = []
        self._state_message_lengths = []

        self._state_meta = {}

        return None


    @property
    def covariance_type(self):
        r""" Return the type of covariance stucture assumed. """
        return self._covariance_type


    @property
    def covariance_regularization(self):
        r""" 
        Return the regularization applied to diagonals of covariance matrices.
        """
        return self._covariance_regularization


    @property
    def threshold(self):
        r""" Return the threshold improvement required in message length. """
        return self._threshold


    @property
    def max_em_iterations(self):
        r""" Return the maximum number of expectation-maximization steps. """
        return self._max_em_iterations


    def search_greedy_forgetful(self, y, K_max=None, random_state=None,
        **kwargs):
        r"""
        Fit the data using a greedy search algorithm, where we do not retain
        information about previous mixtures in order to initialise the next
        set of mixtures. Instead, each new trial of :math:`K` is initialised
        using the K-means++ algorithm.

        :param y:
            The data :math:`y`.

        :param K_max: [optional]
            The maximum number of Gaussian components to consider in the
            mixture (defaults to the number of data points).

        :param random_state: [optional]
            The state to provide to the random number generator.
        """

        y, kwds, K_max, K_predict, handler = self._prepare_search(y, K_max, 
                                                                  **kwargs)
            
        for K in range(1, K_max):

            # Initialise using k-means++.
            mu, cov, weight, responsibility = em._initialise_by_kmeans_pp(y, K,
                covariance_regularization=self.covariance_regularization,
                random_state=random_state)

            prev_I = np.inf
            for i in range(self.max_em_iterations):

                R, ll, I = em.expectation(y, mu, cov, weight, **kwds)
                mu, cov, weight = em.maximization(y, mu, cov, weight, R, **kwds)

                change = prev_I - I
                prev_I = I

                if self.threshold > abs(change):
                    break

            self._record_state(cov, weight, ll, I)

            # Make predictions for past and future mixtures.
            K_target = np.arange(1, weight.size + K_predict)
            predictions = self._predict_message_length(K_target, y, **kwds)

            if handler is not None:
                handler.emit("model", 
                    dict(mu=mu, cov=cov, weight=weight, nll=-np.sum(ll), I=I))
                handler.emit("prediction", predictions)


        raise a


    def _prepare_search(self, y, K_max=None, **kwargs):
        """
        Prepare keyword arguments and other miscellaneous things before starting
        the search.

        :param y:
            The data.

        :param K_max: [optional]
            The maximum number of :math:`K` values to try.
        """

        kwds = dict(
            threshold=self.threshold,
            max_em_iterations=self.max_em_iterations,
            covariance_type=self.covariance_type,
            covariance_regularization=self.covariance_regularization,
            visualization_handler=None)
        kwds.update(kwargs)

        y = np.atleast_1d(y)
        N, D = y.shape
        K_max = N if K_max is None else K_max

        K_predict = kwds.pop("K_predict", 50)
        visualization_handler = kwargs.get("visualization_handler", None)
        return (y, kwds, K_max, K_predict, visualization_handler)


    def search_simulated_annealing(self, y, K_max=None, random_state=None,
        **kwargs):
        r"""
        Search for the optimal number of mixtures using simulated annealing.

        :param y:
            The data :math:`y`.

        :param K_max: [optional]
            The maximum number of Gaussian components to consider in the
            mixture (defaults to the number of data points).

        :param random_state: [optional]
            The state to provide to the random number generator.
        """

        y, kwds, K_max, K_predict, handler = self._prepare_search(y, K_max, 
                                                                  **kwargs)

        temperature = 100000
        minimum_temperature = 1e-6
        cooling_rate = 0.003




        raise a


    def _record_state(self, cov, weight, log_likelihood, message_length):
        r"""
        Record the state of the model to make better predictions for the
        message lengths of future mixtures.

        :param cov:
            The covariance matrices of the current mixture.

        :param weight:
            The relative weights of the current mixture.

        :param log_likelihood:
            The log-likelihood of the current mixture.

        :param message_length:
            The length of the message required to encode the current state.
        """

        self._state_K.append(weight.size)

        # Record determinant of covariance matrices.
        self._state_det_covs.append(np.linalg.det(cov))

        # Record weights.
        self._state_weights.append(weight)
        
        # Record log-likelihood.
        self._state_sum_log_likelihoods.append(np.sum(log_likelihood))

        # Record message length.
        self._state_message_lengths.append(message_length)

        return None


    def _predict_message_length(self, K, y, **kwargs):
        """
        Predict the message length of past or future mixtures.

        :param K:
            An array-like object of the :math:`K`-th mixtures to predict the
            message elngths of.

        :param y:
            The data :math:`y`.
        """

        N, D = y.shape
        predictions, meta = mml.predict_message_length(K, N, D, 
            previous_states=(
                self._state_K,
                self._state_weights,
                self._state_det_covs,
                self._state_sum_log_likelihoods,
                self._state_message_lengths),
            state_meta=self._state_meta)

        # Update the metadata of our state so that future predictions are 
        # faster (e.g., the optimization functions start from values closer to
        # the true value).
        self._state_meta.update(meta)

        return predictions