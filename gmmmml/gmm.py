
"""
Model data using a mixture of Gaussians.
"""

__all__ = ["GaussianMixture"]

import logging
import numpy as np
import scipy
from collections import defaultdict
from time import time
from tqdm import tqdm
from scipy.special import erf
from scipy.signal import find_peaks_cwt
from sklearn import cluster
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms

from . import (mml, em, operations)

logger = logging.getLogger(__name__)

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
        threshold=1e-2, max_em_iterations=100, visualization_handler=None,
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


    def initialize(self, y, K, **kwargs):
        r"""
        Initialize the Gaussian mixture model, given some number of components
        :math:`K`, using the K-means++ method.

        :param y:
            The data values, :math:`y`, which are expected to have :math:`N` 
            samples each with :math:`D` dimensions. Expected shape of :math:`y` 
            is :math:`(N, D)`.

        :param K:
            The number of Gaussian components in the mixture.

        :returns:
            A four-length tuple containing:

            (1) an estimate of the means of the Gaussian components

            (2) an estimate of the covariance matrices of the Gaussian components

            (3) the relative mixing weights of each component

            (4) a responsibility matrix that assigns each datum to a component.
        """

        if K == 1 or len(self._results) == 0:
            return _initialize_with_kmeans_pp(y, K, **kwargs)

        kwds = {**self._em_kwds, **kwargs}

        # get closest in K
        K_trialled = np.hstack(self._results.keys())
        K_diff = np.abs(K - K_trialled)

        index = K_trialled[np.argmin(K_diff)]

        (means, covs, weights), responsibilities, ll, I = self._results[index]

        (means, covs, weights), responsibilities, ll, I \
            = operations.iteratively_operate_components(y, means, covs, weights, K, **kwds)

        return (means, covs, weights, responsibilities)

        #return _initialize_with_kmeans_pp(y, K, **kwargs)


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

        # Record state for predictions.
        self._record_state_for_predictions(covs, weights, ll, I)

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


    def search(self, y, search_strategy="bayes-jumper", **kwargs):
        r"""
        Search for the optimal number of multivariate Gaussian components, and
        the parameters of that mixture, given the data.

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is the
            number of dimensions per observation.

        :param search_strategy: [optional]
            The search strategy to use. The available search strategies include:

            - ``bayes-jumper``: SOME DESCRIPTION HERE TODO.

            - ``greedy-kmeans``: SOME DESCRIPTION HERE TODO.

            - ``kasarapu-allison-2015``: SOME DESCRIPTION TODO

        """

        y = np.atleast_2d(y)

        available_search_strategies = dict([
            ("bayes-jumper", self._search_bayes_jumper),
            ("greedy-kmeans", self._search_greedy_kmeans),
            ("kasarapu-allison-2015", self._search_kasarapu_allison_2015),
        ])

        search_strategy = f"{search_strategy}".lower()

        try:
            func = available_search_strategies[search_strategy]

        except KeyError:
            available_repr = ", ".join(available_search_strategies.keys())
            raise ValueError(
                f"Unknown search strategy provided ('{search_strategy}'). "\
                f"Available strategies include: {available_repr}")

        else:
            t_init = time()
            state, R, ll, I, meta = func(y, **kwargs)

        # Update the metadata.
        meta.update(search_strategy=search_strategy, t_search=time() - t_init)

        # Set the state attribuets.
        self.means_, self.covs_, self.weights_ = state
        self.responsibilities_, self.log_likelihoods_, self.message_lengths_ = (R, ll, I)
        self.meta_ = meta

        return self


    def _search_bayes_jumper(self, y, expected_improvement_fraction=0.01,
                             K_init=10, **kwargs):
        r"""
        Perform Bayesian optimisation to estimate the optimal number of Gaussian
        components :math:`K`, and the optimal parameters of that mixture.

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is the
            number of dimensions per observation.

        :param expected_improvement_fraction: [optional]
            The expected improvement fraction (in message length) before
            considering the Bayesian optimisation process as converged.
            Default is `0.01`.

        :param K_init: [optional]
            The number of initial trials to run in :math:`K`, distributed
            uniformly in log (base 10) space between :math:`K = 1` and 
            :math:`K = N`, where :math:`N` is the number of data points.
        """

        N, D = y.shape
        kwds = {**self._em_kwds, **kwargs}
        
        # Initial guesses.
        K_inits = np.logspace(0, np.log10(N), K_init, dtype=int)

        self._results = {}
        results = {}

        for i, K in enumerate(K_inits):

            try:
                *state, R = self.initialize(y, K, **kwds)

                # Run E-M.
                results[K] = self.expectation_maximization(y, *state, **kwds)

            except ValueError:
                logger.warn(f"Failed to initialize at K = {K}")
                break

            else:
                # Make predictions.
                self._predict_message_length(1 + np.arange(2 * K), N, D, **kwds)

        self._results.update(results)

        # Bayesian optimization.
        converged, prev_I, K_skip = (False, np.inf, [])

        # TODO: Go twice as far as what did work?
        # TODO: consider the time it would take to trial points?
        for iteration in range(1000):

            Kp = (1 + 1.1 * np.arange(np.max(np.hstack(self._results.keys()))))
            Kp = np.unique(Kp).astype(int)
            I, I_var, I_lower = self._predict_message_length(Kp, N, D, **kwds)

            min_I = np.min(self._state_I)

            # Calculate the acquisition function.
            chi = (min_I - I) / np.sqrt(I_var)
            Phi = 0.5 * (1.0 + erf(chi / np.sqrt(2)))
            phi = np.exp(-0.5 * chi**2) / np.sqrt(2 * np.pi * I_var)
            K_ei = (min_I - I) * Phi + I_var * phi

            # The next K values should be places where we expect the greatest
            # improvement in our predictions, and the greatest improvement in
            # the message length.

            # TODO ARGH PICK BETTER HEURISTICS BASED ON TIME TO EVALUATE?

            idx = np.hstack([
                find_peaks_cwt(K_ei, [3]) - 1,
                np.argmax(K_ei),
                np.argsort(I),

            ]).astype(int)

            K_nexts = Kp[idx]

            _, __ = np.unique(K_nexts, return_index=True)
            K_nexts = K_nexts[np.sort(__)]
            
            for K in K_nexts:
                if K in self._state_K or K in K_skip: continue

                logger.info(f"Trying K {K} on iteration {iteration}")

                try:
                    *state, R = self.initialize(y, K, **kwds)

                    # Run E-M..
                    self._results[K] = self.expectation_maximization(y, *state, **kwds)

                except ValueError:
                    logger.exception(f"Failed to initialize mixture at K = {K}")
                    K_skip.append(K)
                    continue

                else:
                    # Update predictions.
                    I, I_var, _ = self._predict_message_length(Kp, N, D, **kwds)

                    idx = np.argmin(self._state_I)

                    best_K, best_I = (self._state_K[idx], self._state_I[idx])
                    logger.info(f"Best so far is K = {best_K} with I = {best_I:.0f} nats")

                    min_I = np.min(self._state_I)
                    eif = np.abs(min_I - np.min(I))/min_I
                    logger.info(f"Expected improvement fraction: {eif}")

                    # Check predictions are within tolerance and that we have
                    # no better predictions to make.
                    if np.abs(prev_I - min_I) < self.threshold \
                    and eif <= expected_improvement_fraction:
                        converged = True
                        break

                    prev_I = min_I
                    break

            if converged: 
               break

        else:
            logger.warning("Bayesian optimization did not converge.")

        # Select the best mixture.
        result = self._results[self._state_K[np.argmin(self._state_I)]]

        meta = dict(K_init=K_init,
                    expected_improvement_fraction=expected_improvement_fraction)

        return (*result, meta)


    def _search_greedy_kmeans(self, y, K_max=None, **kwargs):
        r"""
        Perform a greedy search to estimate the optimal number of Gaussian
        components :math:`K`, and the optimal parameters of that mixture,
        using the K-means++ algorithm to initialize each mixture.

        This algorithm has no stopping criteria in :math:`K`: it will continue
        until `K_max` is reached, and then return the mixture with the minimum
        message length.

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is the
            number of dimensions per observation.

        :param K_max: [optional]
            The maximum number of components to trial. If not given, then this
            will default to :math:`N`, the number of data points.
        """

        N, D = y.shape
        K_max = N if K_max is None else K_max

        kwds = {**self._em_kwds, **kwargs}

        results = {}

        Ks = 1 + np.arange(K_max)
        for K in Ks:
        
            try:
                *state, R = self.initialize(y, K, **kwds)

                # Run E-M.
                results[K] = self.expectation_maximization(y, *state, **kwds)

            except ValueError:
                logger.warn(f"Failed to initialize at K = {K}")
                continue

            else:
                # Make predictions.
                self._predict_message_length(Ks, N, D, **kwds)

        # Select the best mixture.
        result = results[self._state_K[np.argmin(self._state_I)]]

        meta = dict()
        return (*result, meta)



    def _search_kasarapu_allison_2015(self, y, **kwargs):
        r"""
        Find the optimal mixture of Gaussians using the perturbation search
        algorithm described by Kasarapu & Allison (2015).

        :param y:
            A :math:`N\times{}D` array of the observations :math:`y`,
            where :math:`N` is the number of observations, and :math:`D` is the
            number of dimensions per observation.
        """

        N, D = y.shape
        kwds = {**self._em_kwds, **kwargs}
        tqdm_format = lambda f: None if kwds.get("quiet", False) else f

        # Initialize.
        *state, R = self.initialize(y, K=1, **kwds)
        R, ll, I = self.expectation(y, *state, **kwds)

        ml = lambda I: np.sum(np.hstack(I.values()))

        iterations, prev_ml = (0, ml(I))

        while True:

            K = state[-1].size
            best_perturbations = defaultdict(lambda: [np.inf])

            # Exhaustively split all components.
            for k in tqdm(range(K), desc=tqdm_format(f"Splitting K={K}")):
                p = operations.split_component(y, *state, R, k, **kwds)

                # Keep best split component.
                I = ml(p[-1])
                if I < best_perturbations["split"][0]:
                    best_perturbations["split"] = [I, k] + list(p)

            if K > 1:
                # Exhaustively delete all components.
                for k in tqdm(range(K), desc=tqdm_format(f"Deleting K={K}")):
                    p = operations.delete_component(y, *state, R, k, **kwds)

                    # Keep best delete component.
                    I = ml(p[-1])
                    if I < best_perturbations["delete"][0]:
                        best_perturbations["delete"] = [I, k] + list(p)

                # Exhaustively merge all components.
                for k in tqdm(range(K), desc=tqdm_format(f"Merging K={K}")):
                    p = operations.merge_component(y, *state, R, k, **kwds)

                    # Keep best merged component.
                    I = ml(p[-1])
                    if I < best_perturbations["merge"][0]:
                        best_perturbations["merge"] = [I, k] + list(p)

            # Get best perturbation.
            bop, bp = min(best_perturbations.items(), key=lambda x: x[1][0])
            logger.debug(f"Best operation is {bop} on index {bp[1]}")

            if bp[0] < prev_ml:
                # Set the new state as the best perturbation.
                prev_ml, _, state, R, ll, I = bp
                iterations += 1

            else:
                # Done. All perturbations are worse.
                break

        meta = dict()
        return (state, R, ll, I, meta)


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
        handler = kwargs["visualization_handler"]
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

        return (I, I_var, I_lower)


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


def _initialize_with_kmeans_pp(y, K, random_state=None, **kwargs):
    r"""
    Initialize a Gaussian mixture model using the K-means++ algorithm.

    :param y:
        The data values, :math:`y`, which are expected to have :math:`N` 
        samples each with :math:`D` dimensions. Expected shape of :math:`y` 
        is :math:`(N, D)`.

    :param K:
        The number of Gaussian components in the mixture.
    
    :param random_state: [optional]
        The state to provide to the random number generator.

    :returns:
        A four-length tuple containing:

        (1) an estimate of the means of the components

        (2) an estimate of the covariance matrices of the components

        (3) an estimate of the relative mixings of the components

        (4) the responsibility matrix for each data point to each component.
    """

    random_state = check_random_state(random_state)
    squared_norms = row_norms(y, squared=True)
    means = cluster.k_means_._k_init(y, K,
                                     random_state=random_state,
                                     x_squared_norms=squared_norms)

    labels = np.argmin(scipy.spatial.distance.cdist(means, y), axis=0)

    N, D = y.shape
    responsibilities = np.zeros((K, N))
    responsibilities[labels, np.arange(N)] = 1.0

    covs = em._estimate_covariance_matrix_full(y, means, responsibilities, 
        covariance_regularization=kwargs.get("covariance_regularization", 0))

    weights = responsibilities.sum(axis=1)/N

    return (means, covs, weights, responsibilities)


def _initialize(y, covariance_type, covariance_regularization, **kwargs):
    r"""
    Return initial estimates of the parameters.

    :param y:
        The data values, :math:`y`, which are expected to have :math:`N` 
        samples each with :math:`D` dimensions. Expected shape of :math:`y` 
        is :math:`(N, D)`.

    :param covariance_type: [optional]
        The structure of the covariance matrix for individual components.
        The available options are: 

        - "full": for a full-rank covariance matrix with non-zero off-diagonal
                  terms,
        - "diag": for a diagonal covariance matrix.

    :param covariance_regularization: [optional]
        Regularization strength to add to the diagonal of covariance matrices
        (default: `0`).
    
    :returns:
        A three-length tuple containing:

        (1) an estimate of the means of the components

        (2) an estimate of the covariance matrices of the components

        (3) an estimate of the relative mixings of the components
    """

    # If you *really* know what you're doing, then you can give your own.
    if kwargs.get("__initialize", None) is not None:
        return kwargs.pop("__initialize")

    weights = np.ones((1, 1))
    N, D = y.shape
    means = np.mean(y, axis=0).reshape((1, -1))

    responsibilities = np.ones((1, N))

    covs = em._estimate_covariance_matrix(
        y, means, responsibilities, covariance_type, covariance_regularization)

    return (means, covs, weights)
    