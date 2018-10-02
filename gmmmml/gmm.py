
"""
Model data using a mixture of Gaussians.
"""

__all__ = ["GaussianMixture"]

import logging
import numpy as np
import scipy
from collections import (defaultdict, OrderedDict)
from time import time
from tqdm import tqdm
from scipy.special import erf
from scipy.signal import find_peaks_cwt
from sklearn import cluster
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms

from . import (mml, em, operations, strategies)

logger_name, *_ = __name__.split(".")
logger = logging.getLogger(logger_name)

class GaussianMixture(object):

    r"""
    Model data from an unknown number of multivariate Gaussian distribution(s), 
    using minimum message length (MML) as the objective function.

    :param covariance_type: [optional]
        The structure of the covariance matrix for individual components.
        The available options are: 

        - "full": for a full-rank covariance matrix with non-zero off-diagonal
                  terms,
        - "diag": for a diagonal covariance matrix.

    :param covariance_regularization: [optional]
        Regularization strength to add to the diagonal of covariance matrices
        (default: `0`).

    :param threshold: [optional]
        The relative improvement in message length required before stopping an
        expectation-maximization step (default: `1e-2`).

    :param max_em_iterations: [optional]
        The maximum number of iterations to run per expectation-maximization
        loop (default: `100`).

    :param visualization_handler: [optional]
        An optional visualization handler to illustrate the search progress.
    """

    def __init__(self, covariance_type="full", covariance_regularization=0, 
        threshold=1e-5, max_em_iterations=1000, visualization_handler=None,
        **kwargs):

        available = ("full", )
        covariance_type = covariance_type.strip().lower()
        if covariance_type not in available:
            raise ValueError(f"covariance type '{covariance_type}' is invalid."\
                             f" Must be one of: {available}")

        covariance_regularization = float(covariance_regularization)
        if 0 > covariance_regularization:
            raise ValueError("covariance_regularization must be non-negative")

        threshold = float(threshold)
        if 0 >= threshold:
            raise ValueError("threshold must be positive")

        max_em_iterations = int(max_em_iterations)
        if 1 > max_em_iterations:
            raise ValueError("max_em_iterations must be a positive integer")

        self._em_kwds = dict(threshold=threshold,
                             covariance_type=covariance_type,
                             max_em_iterations=max_em_iterations,
                             covariance_regularization=covariance_regularization,
                             visualization_handler=visualization_handler)

        # Lists to record states for predictive purposes.
        self._state_K = []
        self._state_I = []
        self._state_weights = []
        self._state_slog_weights = []
        self._state_det_covs = []
        self._state_slog_likelihoods = []

        self._state_meta = {}

        return None


    @property
    def covariance_type(self):
        r""" Return the type of covariance stucture assumed. """
        return self._em_kwds["covariance_type"]


    @property
    def covariance_regularization(self):
        r""" 
        Return the regularization applied to diagonals of covariance matrices.
        """
        return self._em_kwds["covariance_regularization"]


    @property
    def threshold(self):
        r""" Return the threshold improvement required in message length. """
        return self._em_kwds["threshold"]


    @property
    def max_em_iterations(self):
        r""" Return the maximum number of expectation-maximization steps. """
        return self._em_kwds["max_em_iterations"]


    def expectation(self, y, means, covs, weights, **kwargs):
        r"""
        Perform the expectation step of the expectation-maximization algorithm
        on the mixture, given the data.

        :param y:
            The data values, :math:`y`, which are expected to have :math:`N` 
            samples each with :math:`D` dimensions. Expected shape of :math:`y` 
            is :math:`(N, D)`.

        :param means:
            The current estimate of the multivariate means of the :math:`K`
            components. The expected shape of `means` is :math:`(K, D)`.

        :param covs:
            The current estimate of the covariance matrices of the :math:`K`
            components. The expected shape of `covs` is :math:`(K, D, D)`.

        :param weights:
            The current estimate of the relative weights :math:`w` of all 
            :math:`K` components. The sum of weights must equal 1. The expected 
            shape of `weights` is :math:`(K, )`.
        
        :returns:
            A three-length tuple containing:

            (1) the responsibility matrix

            (2) the log-likelihood of each data point, given the model,

            (3) a dictionary describing the lengths of each component of the
                message.
        """

        kwds = {**self._em_kwds, **kwargs}
        
        R, ll, I = em.expectation(y, means, covs, weights, **kwds)

        # Do visualization stuff.
        handler = kwargs.get("visualization_handler", None)
        if handler is not None:
            handler.emit("actual_I",
                dict(K=weights.size, I=np.sum(np.hstack(I.values()))))

            handler.emit("actual_I_weights", 
                         dict(K=weights.size, I=I["I_weights"]))

            handler.emit("actual_I_slogdetcovs", 
                         dict(K=weights.size, I=I["I_slogdetcovs"]))

            handler.emit("actual_I_data", 
                         dict(K=weights.size, I=I["I_data"]))

            # TODO: emit snapshot if needed

        return (R, ll, I)


    def maximization(self, y, means, covs, weights, responsibilities, **kwargs):
        r"""
        Perform the maximization step of the expectation-maximization algorithm
        on the mixture, given the data.

        :param y:
            The data values, :math:`y`, which are expected to have :math:`N` 
            samples each with :math:`D` dimensions. Expected shape of :math:`y` 
            is :math:`(N, D)`.

        :param means:
            The current estimate of the multivariate means of the :math:`K`
            components. The expected shape of `means` is :math:`(K, D)`.

        :param covs:
            The current estimate of the covariance matrices of the :math:`K`
            components. The expected shape of `covs` is :math:`(K, D, D)`.

        :param weights:
            The current estimate of the relative weights :math:`w` of all 
            :math:`K` components. The sum of weights must equal 1. The expected 
            shape of `weights` is :math:`(K, )`.

        :param responsibilities:
            The responsibility matrix for all :math:`N` observations being
            partially assigned to each :math:`K` component. The expected shape 
            of `responsibilities` is :math:`(N, K)`.        
        
        :returns:
            A three-length tuple containing:

            (1) an updated estimate of the means of the Gaussian components

            (2) an updated estimate of the covariance matrices of the Gaussian
                components

            (3) an updated estimate of the relative weights of the Gaussian
                components.
        """

        kwds = {**self._em_kwds, **kwargs}
        means, covs, weights = em.maximization(y, means, covs, weights, 
                                               responsibilities, **kwds)

        # Do visualization stuff.
        handler = kwargs.get("visualization_handler", None)
        if handler is not None:
            handler.emit("maximization", 
                         dict(means=means, covs=covs, weights=weights))

        return (means, covs, weights)


    def expectation_maximization(self, y, means, covs, weights, **kwargs):
        r"""
        Run the expectation-maximization algorithm on the given mixture.

        :param y:
            The data values, :math:`y`, which are expected to have :math:`N` 
            samples each with :math:`D` dimensions. Expected shape of :math:`y` 
            is :math:`(N, D)`.

        :param means:
            The current estimate of the multivariate means of the :math:`K`
            components. The expected shape of `means` is :math:`(K, D)`.

        :param covs:
            The current estimate of the covariance matrices of the :math:`K`
            components. The expected shape of `covs` is :math:`(K, D, D)`.

        :param weights:
            The current estimate of the relative weights :math:`w` of all 
            :math:`K` components. The sum of weights must equal 1. The expected 
            shape of `weights` is :math:`(K, )`.

        :param quiet: [optional]
            Optionally turn off progress bars.

        :returns:
            A four-length tuple containing:

            (1) a tuple containing the best estimate of the means, covariance 
                matrices, and relative weights of the :math:`K` components

            (2) the responsibility matrix

            (3) the log-likelihood of each data point, given the model

            (4) a dictionary containing the message length of various parts of 
                the best model.
        """

        kwds = {**self._em_kwds, **kwargs}

        # Overload the keywords so that we can do visualisation.
        kwds.update(__expectation_function=self.expectation,
                    __maximization_function=self.maximization)

        return em.expectation_maximization(y, means, covs, weights, **kwds)


    def search(self, y, search_strategy="BayesStepper", **kwargs):
        r"""
        Search for the optimal number of multivariate Gaussian components, and
        the parameters of that mixture, given the data.

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is the
            number of dimensions per observation.

        :param search_strategy: [optional]
            The search strategy to use. This can either be a string describing
            the strategy, or a `strategies.Strategy` class.

            Recall that a `Strategy` is a set of policies that govern the search
            heuristic. This set of policies includes policies that govern:

            - how to initialise the search,
            - how far to make predictions about the mixture message length,
            - how to decide where the next step in the search should be,
            - how to move to the next step in the search,
            - when the search has converged.

            Some strategies include:

            - `strategies.KasarapuAllison2015`
            - `strategies.BayesStepper`
            - `strategies.BayesJumper`
            - `strategies.GreedyKMeans`

            Defaults to `strategies.BayesStepper`, a fast and possibly convex
            optimisation search strategy.
        """

        y = np.atleast_2d(y)

        t_init = time()
        
        # TODO: We may not want to do this,....
        #       Or we may want to update the recording state so it does not get
        #       too bloated.
       
        kwds = {**self._em_kwds, **kwargs}
        kwds.update(__expectation_function=self.expectation,
                    __maximization_function=self.maximization,
                    __callback_function=lambda s, *a: \
                        self._record_state_for_predictions(*s[1:], *a[1:]))

        if isinstance(search_strategy, str):
            try:
                search_strategy_class = getattr(strategies, search_strategy)

            except AttributeError:
                raise ValueError("unrecognised strategy: '{}'".format(strategy))

        else:
            # Assume strategy is a strategy class.
            search_strategy_class = strategy

        logger.info(f"Using search strategy '{search_strategy_class.__name__}'")

        strategy = search_strategy_class(self, **kwds)
        self._results = OrderedDict(strategy.initialise(y, **kwds))

        # Chose the next K to trial.
        for K in strategy.move(y, **kwds):

            # Decide how to repartition to that K.

            # Note that here we take the K returned by the repartition function,
            # because some repartition functions will be greedy and not actually
            # go to the K they were instructed to (or K could be None for
            # greedy repartition policies).
            self._results.update([strategy.repartition(y, K, **kwds)])

            index = np.argmin(self._state_I)
            K_best, I_best = (self._state_K[index], self._state_I[index])
            logger.debug(f"Best so far is K = {K_best} with I = {I_best}")
            
        index = np.argmin(self._state_I)
        K_best, I_best = (self._state_K[index], self._state_I[index])
        logger.info(f"Best mixture has K = {K_best} and I = {I_best:.0f}")

        state, R, ll, I = self._results[K_best]
        meta = dict(strategy=search_strategy_class.__name__, 
                    t_search=time() - t_init)

        # Set the state attribuets.
        self.means_, self.covs_, self.weights_ = state
        self.responsibilities_, self.log_likelihoods_, self.message_lengths_ = (R, ll, I)
        self.meta_ = meta

        return self


    def _predict_message_length(self, K, N, D, **kwargs):
        """
        Predict the message lengths of target Gaussian mixtures.

        :param K:
            An array-like object of K-th mixtures to predict message lengths
            for.

        :param N:
            The number of data points.

        :param D:
            The dimensionality of each data point.
        """

        K = np.atleast_1d(K)

        # Prepare a dictionary that we will use for making predictions.
        data = dict(
            K=self._state_K,
            I=self._state_I,
            weights=self._state_weights,
            det_covs=self._state_det_covs,
            sum_log_weights=self._state_slog_weights,
            negative_log_likelihood=-np.array(self._state_slog_likelihoods))

        # Constant terms.
        I_other = mml.information_of_mixture_constants(K, N, D)

        # Sum of the log of the weights.
        I_sum_log_weights, I_sum_log_weights_var, \
        I_sum_log_weights_lower, I_sum_log_weights_upper \
            = mml.predict_information_of_sum_log_weights(K, N, D, data=data)

        # Sum of the log of the determinant of the covariance matrices.
        I_sum_log_det_covs, I_sum_log_det_covs_var, \
        I_sum_log_det_covs_lower, I_sum_log_det_covs_upper \
            = mml.predict_information_of_sum_log_det_covs(K, D, data=data)

        # Negative log-likelihood.
        I_data, I_data_var, I_data_lower \
            = mml.predict_negative_sum_log_likelihood(K, N, D, data=data)

        # Predict total, given other predictions.
        I = I_other + I_sum_log_weights + I_sum_log_det_covs + I_data
        I_var = I_sum_log_weights_var + I_sum_log_det_covs_var + I_data_var
        I_lower = I_other + I_sum_log_weights_lower \
                + I_sum_log_det_covs_lower + I_data_lower

        # TODO: Store the predictions somewhere?

        # Visualize the predictions.
        handler = kwargs.get("visualization_handler", None)
        if handler is not None:
            handler.emit("predict_I_weights", dict(
                K=K, I=I_sum_log_weights, 
                I_var=I_sum_log_weights_var,
                I_lower=I_sum_log_weights_lower, 
                I_upper=I_sum_log_weights_upper))

            handler.emit("predict_I_slogdetcovs", dict(
                K=K, I=I_sum_log_det_covs, 
                I_var=I_sum_log_det_covs_var,
                I_lower=I_sum_log_det_covs_lower,
                I_upper=I_sum_log_det_covs_upper))

            handler.emit("predict_I_data", dict(
                K=K, I=I_data, I_var=I_data_var, I_lower=I_data_lower))

            # Since this is the final prediction, create a snapshot image.
            handler.emit("predict_I", 
                         dict(K=K, I=I, I_var=I_var, I_lower=I_lower),
                         snapshot=True)

        return (K, I, I_var, I_lower)


    def _record_state_for_predictions(self, covs, weights, log_likelihoods,
                                      message_length):
        r"""
        Record the state of a Gaussian mixture in order to make predictions on
        future mixtures.

        :param covs:
            The current estimate of the covariance matrices of the :math:`K`
            components. The expected shape of `covs` is :math:`(K, D, D)`.

        :param weights:
            The current estimate of the relative weights :math:`w` of all 
            :math:`K` components. The sum of weights must equal 1. The expected 
            shape of `weights` is :math:`(K, )`.

        :param log_likelihoods:
            The log-likelihood of each data point, given the model.

        :param message_length:
            The message length of the current mixture.

        :returns:
            A boolean indicating whether the state was recorded.
        """

        # Check that the state *should* be saved.
        I = np.sum(np.hstack(message_length.values()))

        slog_weights = np.sum(np.log(weights))
        detcovs = np.linalg.det(covs)
        if not np.all(np.isfinite(np.hstack([covs.flatten(), weights,
            log_likelihoods, I, slog_weights]))) or np.any(detcovs <= 0):
            logger.warn("Ignoring invalid state.")
            return False

        self._state_K.append(weights.size)

        # Record determinates of covariance matrices.
        self._state_det_covs.append(detcovs)

        # Record sum of the log of the weights.
        self._state_weights.append(weights)
        self._state_slog_weights.append(slog_weights)

        # Record log likelihood
        self._state_slog_likelihoods.append(np.sum(log_likelihoods))
        self._state_I.append(I)

        return True

