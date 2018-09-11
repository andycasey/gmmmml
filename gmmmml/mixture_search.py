
"""
An estimator for modelling data from a mixture of Gaussians, 
using an objective function based on minimum message length.
"""

__all__ = [
    "GaussianMixture", 
    "responsibility_matrix",
    "split_component", "merge_component", "delete_component", 
] 
import logging
import numpy as np
import scipy
import scipy.misc
import scipy.stats
import scipy.optimize as op
import tqdm
import warnings
import os
from sklearn import cluster
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms

from collections import defaultdict

from . import mml

logger = logging.getLogger(__name__)


def _group_over(x, y, function):

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    x_unique = np.sort(np.unique(x))
    y_unique = np.nan * np.ones_like(x_unique)

    for i, xi in enumerate(x_unique):
        match = (x == xi)
        y_unique[i] = function(y[match])

    return (x_unique, y_unique)



def _approximate_ldc_per_mixture_component(k, *params):
    a, b, c = params
    return a/(k - b) + c



def _total_parameters(K, D):
    r"""
    Return the total number of model parameters :math:`Q`, if a full 
    covariance matrix structure is assumed.

    .. math:

        Q = \frac{K}{2}\left[D(D+3) + 2\right] - 1


    :param K:
        The number of Gaussian mixtures.

    :param D:
        The dimensionality of the data.

    :returns:
        The total number of model parameters, :math:`Q`.
    """
    return (0.5 * D * (D + 3) * K) + K - 1






def _approximate_bound_sum_log_determinate_covariances(target_K, cov):

    K, D, _ = cov.shape

    target_K = np.atleast_1d(target_K)
    current_logdet = np.log(np.linalg.det(cov))
    min_logdet, max_logdet = np.min(current_logdet), np.max(current_logdet)

    bounds = np.zeros((target_K.size, 2))
    for i, k in enumerate(target_K):
        bounds[i] = [np.sum(k * min_logdet), np.sum(k * max_logdet)]

    #if target_K[0] > 8:
    #    raise a
    bounds[:, 1] = 125.9 * target_K
    return bounds


def _mixture_message_length_parts(K, N, D):

    Q = _total_parameters(K, D)

    I_mixtures = K * np.log(2) * (1 - D/2.0) + scipy.special.gammaln(K) \
        + 0.25 * (2.0 * (K - 1) + K * D * (D + 3)) * np.log(N)
    I_parameters = 0.5 * np.log(Q * np.pi) - 0.5 * Q * np.log(2 * np.pi)

    return I_mixtures + I_parameters



def _mixture_message_length(K, N, D, log_likelihood, slogdetcov, weights=None, 
    yerr=0.001):
    """
    Estimate the message length of a gaussian mixture model.

    :param K:
        The number of mixtures. This can be an array.

    :param N:
        The number of data points.
    
    :param D:
        The dimensionality of the data.

    :param log_likelihood:
        The estimated sum of the log likelihood of the future mixtures.

    :param slogdetcov:
        The estimated sum of the log of the determinant of the covariance 
        matrices of the future mixtures.

    :param weights: [optional]
        The estimated weights of future mixtures. If `None` is given then the
        upper bound for a losslessly-encoded multinomial distribution of values
        will be used to calculate the message length:

        .. math:

            \sum_{k=1}^{K}\log{w_k} \approx -K\log{K}

    :param yerr: [optional]
        The homoscedastic noise in y for each data point.
    """

    K = np.atleast_1d(K)
    slogdetcov = np.atleast_1d(slogdetcov)
    log_likelihood = np.atleast_1d(log_likelihood)
    
    if K.size != slogdetcov.size:
        raise ValueError("the size of K and slogdetcov are different")

    if K.size != log_likelihood.size:
        raise ValueError("the size of K and log_likelihood are different")

    # Calculate the different contributions of the message length so we can
    # predict them.
    if weights is not None:
        w_size = np.array([len(w) for w in weights])
        if not np.all(w_size == K):
            raise ValueError("the size of the weights does not match K")
        slogw = np.array([np.sum(np.log(w)) for w in weights])
    else:
        slogw = -K * np.log(K)

    Q = _total_parameters(K, D)

    I_mixtures = K * np.log(2) * (1 - D/2.0) + scipy.special.gammaln(K) \
        + 0.25 * (2.0 * (K - 1) + K * D * (D + 3)) * np.log(N)
    I_parameters = 0.5 * np.log(Q * np.pi) - 0.5 * Q * np.log(2 * np.pi)
    
    I_data = -log_likelihood 
    # TODO: this requires a thinko
    #- D * N * np.log(yerr)
    I_slogdetcovs = -0.5 * (D + 2) * slogdetcov
    I_weights = (0.25 * D * (D + 3) - 0.5) * slogw

    I_parts = dict(
        I_mixtures=I_mixtures, I_parameters=I_parameters, 
        I_data=I_data, I_slogdetcovs=I_slogdetcovs, 
        I_weights=I_weights)

    I = np.sum([I_mixtures, I_parameters, I_data, I_slogdetcovs, I_weights],
        axis=0) # [nats]

    return I_parts





class GaussianMixture(object):

    r"""
    Model data from (potentially) many multivariate Gaussian distributions, 
    using minimum message length (MML) as the objective function.

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
    """

    parameter_names = ("means", "covariances", "weights")

    def __init__(self, covariance_type="full", covariance_regularization=0, 
        threshold=1e-5, max_em_iterations=10000, visualization_handler=None,
        predict=25, **kwargs):

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

        mu, cov, weight, responsibility = _initialize_with_kmeans_pp(y, K, **kwargs)

        # Do visualization stuff.

        return (mu, cov, weight, responsibility)


    def expectation(self, y, mu, cov, weight, **kwargs):

        kwds = {**self._em_kwds, **kwargs}

        R, ll, message_length = _expectation(y, mu, cov, weight, **kwds)

        # Record state for predictions.
        self._record_state_for_predictions(cov, weight, ll, message_length)

        # Do visualization stuff.
        handler = kwargs.get("visualization_handler", None)
        if handler is not None:
            handler.emit("actual_I_weights", 
                         dict(K=weight.size, I=message_length["I_weights"]))

            handler.emit("actual_I_slogdetcovs", 
                         dict(K=weight.size, I=message_length["I_slogdetcovs"]))

            handler.emit("actual_I_data", 
                         dict(K=weight.size, I=-np.sum(ll)))

            handler.emit("actual_I",
                dict(K=weight.size, I=np.sum(np.hstack(message_length.values()))),
                snapshot=kwds.get("__snapshot_every_em_step", False))


        return (R, ll, message_length)


    def maximization(self, y, means, covs, weights, responsibilities, **kwargs):

        kwds = {**self._em_kwds, **kwargs}
        means, covs, weights = _maximization(y, means, covs, weights, 
                                             responsibilities, **kwds)

        # TODO: Make predictions?

        # Do visualization stuff.
        handler = kwargs.get("visualization_handler", None)
        if handler is not None:
            handler.emit("maximization", dict(means=means,
                                              covs=covs,
                                              weights=weights))

        return (means, covs, weights)


    def _search_log_jumper(self, y, **kwargs):

        y = np.atleast_1d(y)

        N, D = y.shape
        kwds = {**self._em_kwds, **kwargs}

        handler = kwds.get("visualization_handler", None)

        # Initial guesses.
        K_inits = np.logspace(0, np.log10(N), kwds.get("K_init", 10)).astype(int)

        for K in K_inits[1:]:

            # Assign everything to the closest thing.
            means, covs, weights, responsibilities = self.initialize(y, K, **kwds)

            # Run one E-M step.
            try:                       
                responsibilities, ll, message_length \
                    = self.expectation(y, means, covs, weights, **kwds)

            except ValueError:
                logger.exception("Failed to calculate E-step")
                # Break to adjust the delta_K
                break

            means, covs, weights = self.maximization(y, means, covs, weights,
                                                     responsibilities, **kwds)

        # Predict message lengths.
        K_predict = np.arange(1, K)

        s = 1

        K_skip = []

        converged = False
        while not converged:

            print("Re-predicting")

            I, I_var, I_lower = self._predict_message_length(K_predict, N, D, **kwds)

            idx = np.argsort(I)

            for K in K_predict[idx]:
                if K in K_skip or K in self._state_K:
                    continue

                kbest = np.array(self._state_K)[np.argmin(self._state_I)]
                print(f"Running {K}: best so far is K = {kbest}")

                print(np.min(I - s * np.sqrt(I_var)), np.min(self._state_I), np.min(I - s * np.sqrt(I_var)) > np.min(self._state_I))

                # Stop when there is no better prediction (after accounting for 3\sqrt(var))
                # and we have trialled +/- 1 of the K_best.
                if np.min(I - s * np.sqrt(I_var)) > np.min(self._state_I):
                    print("Checking")
                    K_best = np.array(self._state_K)[np.argmin(self._state_I)]
                    print(K_best - 1 in self._state_K, K_best + 1 in self._state_K)
                    if K_best - 1 in self._state_K and K_best + 1 in self._state_K:
                        print("WE ARE DONE")
                        converged = True
                        break

                try:
                    means, covs, weights, responsibilities = self.initialize(y, K, **kwds)

                    # Run one E-M step.
                    responsibilities, ll, message_length \
                        = self.expectation(y, means, covs, weights, **kwds)

                except ValueError:
                    logger.exception("Failed to calculate E-step")
                    # Break to adjust the delta_K
                    K_skip.append(K)
                    continue


                prev_I = np.inf
                for i in range(self.max_em_iterations):

                    means, covs, weights = self.maximization(y, means, covs, weights,
                                                             responsibilities, **kwds)

                    try:
                        responsibilities, ll, message_length \
                            = self.expectation(y, means, covs, weights, **kwds)

                    except ValueError:
                        break


                    I = np.sum(np.hstack(message_length.values()))
                    if (prev_I - I) < self.threshold:
                        break

                break

            else:
                break


        raise a
            


    def kmeans_search(self, y, K_max=None, **kwargs):

        y = np.atleast_1d(y)

        N, D = y.shape
        K_max = N if K_max is None else K_max

        kwds = {**self._em_kwds, **kwargs}
        handler = kwds.get("visualization_handler", None)


        for K in range(1, K_max):
        
            # Assign everything to the closest thing.
            means, covs, weights, responsibilities = self.initialize(y, K, **kwargs)

            prev_I = np.inf
            for i in range(self.max_em_iterations):

                means, covs, weights = self.maximization(y, means, covs, weights,
                                                         responsibilities, **kwargs)

                try:
                    responsibilities, ll, message_length \
                        = self.expectation(y, means, covs, weights, **kwargs)

                except ValueError:
                    logger.exception("Failed to calculate E-step")
                    break

                I = np.sum(np.hstack(message_length.values()))
                if (prev_I - I) < self.threshold:
                    break

                prev_I = I

            else:
                logger.warn(f"Convergence not reached after {i+1} iterations.")

            if handler is not None:
                K_predict = np.arange(1, weights.size + 25)
                self._predict_message_length(K_predict, N, D, **kwds)
                



        return None





    def search(self, y, **kwargs):

        kwds = dict(
            threshold=self._threshold, 
            max_em_iterations=self._max_em_iterations,
            covariance_type=self.covariance_type, 
            covariance_regularization=self._covariance_regularization,
            visualization_handler=None)
        kwds.update(kwargs)

        # Initialize the mixture.
        mu, cov, weight = _initialize(y, **kwds)
        R, ll, message_length = _expectation(y, mu, cov, weight, **kwds)

        # Record things for predictive purposes.
        self._record_state_for_predictions(cov, weight, ll)



        while True:
            K = weight.size
            best_perturbations = defaultdict(lambda: [np.inf])

            for k in range(K):
                perturbation = split_component(y, mu, cov, weight, R, k, **kwds)
    
                p_cov, p_weight, p_ll = (perturbation[1], perturbation[2], perturbation[-1])
                self._record_state_for_predictions(p_cov, p_weight, p_ll)

                if perturbation[-1] < best_perturbations["split"][-1]:
                    best_perturbations["split"] = [k] + list(perturbation)

            bop, bp = min(best_perturbations.items(), key=lambda x: x[1][-1])
            b_k, b_mu, b_cov, b_weight, b_R, b_meta, b_ml = bp

            
            # Check to see if we are cooked.
            if b_ml >= message_length: break
            # Not cooked!

            mu, cov, weight, R, meta = (b_mu, b_cov, b_weight, b_R, b_meta)

            message_length = b_ml
            ll = b_meta["log_likelihood"]

            # Record things for predictive purposes.
            self._record_state_for_predictions(cov, weight, ll)

            # Predict future mixtures.
            if visualization_handler is not None:

                target_K = weight.size + np.arange(1, 10)


                self._predict_message_length(target_K, cov, weight, y.shape[0], ll, I, **kwds)

                #visualization_handler.emit("predict", dict(model=self))



        raise a




    def _predict_message_length(self, K, N, D, **kwargs):
        """
        Predict the message lengths of past or future mixtures.

        :param K:
            An array-like object of K-th mixtures to predict message lengths
            for.
        """

        K = np.atleast_1d(K)

        # Constant terms.
        I_other = mml.information_of_mixture_constants(K, N, D)

        # Sum of the log of the weights.
        I_sum_log_weights, I_sum_log_weights_var, \
        I_sum_log_weights_lower, I_sum_log_weights_upper \
            = mml.predict_information_of_sum_log_weights(
                K, N, D, data=(self._state_K, self._state_slog_weights))

        # Sum of the log of the determinant of the covariance matrices.
        I_sum_log_det_covs, I_sum_log_det_covs_var, \
        I_sum_log_det_covs_lower, I_sum_log_det_covs_upper \
            = mml.predict_information_of_sum_log_det_covs(
                K, D, data=(self._state_K, self._state_det_covs))

        # Negative log-likelihood.
        I_data, I_data_var, I_data_lower \
            = mml.predict_negative_sum_log_likelihood(
                K, N, D, data=(self._state_K, -np.array(self._state_slog_likelihoods),
                         self._state_weights, self._state_det_covs))

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
                K=K, I=I_sum_log_weights, I_var=I_sum_log_weights_var,
                I_lower=I_sum_log_weights_lower, I_upper=I_sum_log_weights_upper))

            handler.emit("predict_I_slogdetcovs", dict(
                K=K, I=I_sum_log_det_covs, I_var=I_sum_log_det_covs_var,
                I_lower=I_sum_log_det_covs_lower,
                I_upper=I_sum_log_det_covs_upper))

            handler.emit("predict_I_data", dict(
                K=K, I=I_data, I_var=I_data_var, I_lower=I_data_lower))

            # Since this is the final prediction, create a snapshot image.
            handler.emit("predict_I", 
                         dict(K=K, I=I, I_var=I_var, I_lower=I_lower),
                         snapshot=True)


        return (I, I_var, I_lower)


    def _record_state_for_predictions(self, cov, weight, log_likelihood,
        message_lengths):
        r"""
        Record 'best' trialled states (for a given K) in order to make some
        predictions about future mixtures.
        """

        # Check that the state *should* be saved.
        I = np.sum(np.hstack(message_lengths.values()))

        if not np.all(np.isfinite(np.hstack([cov.flatten(), weight,
            log_likelihood, I]))):
            print("Ignoring state")
            return None

        self._state_K.append(weight.size)

        # Record determinates of covariance matrices.
        self._state_det_covs.append(np.linalg.det(cov))

        # Record sum of the log of the weights.
        self._state_weights.append(weight)
        self._state_slog_weights.append(np.sum(np.log(weight)))

        # Record log likelihood
        self._state_slog_likelihoods.append(np.sum(log_likelihood))
        self._state_I.append(I)



    

def responsibility_matrix(y, mu, cov, weight, covariance_type, 
    full_output=False, **kwargs):
    r"""
    Return the responsibility matrix,

    .. math::

        r_{ij} = \frac{w_{j}f\left(x_i;\theta_j\right)}{\sum_{k=1}^{K}{w_k}f\left(x_i;\theta_k\right)}


    where :math:`r_{ij}` denotes the conditional probability of a datum
    :math:`x_i` belonging to the :math:`j`-th component. The effective 
    membership associated with each component is then given by

    .. math::

        n_j = \sum_{i=1}^{N}r_{ij}
        \textrm{and}
        \sum_{j=1}^{M}n_{j} = N


    where something.
    
    :param y:
        The data values, :math:`y`.

    :param mu:
        The mean values of the :math:`K` multivariate normal distributions.

    :param cov:
        The covariance matrices of the :math:`K` multivariate normal
        distributions. The shape of this array will depend on the 
        ``covariance_type``.

    :param weight:
        The current estimates of the relative mixing weight.

    :param full_output: [optional]
        If ``True``, return the responsibility matrix, and the log likelihood,
        which is evaluated for free (default: ``False``).

    :returns:
        The responsibility matrix. If ``full_output=True``, then the
        log likelihood (per observation) will also be returned.
    """

    precision_cholesky = _compute_precision_cholesky(cov, covariance_type)
    weighted_log_prob = np.log(weight) \
                      + _estimate_log_gaussian_prob(y, mu, precision_cholesky, covariance_type)
        

    log_likelihood = scipy.misc.logsumexp(weighted_log_prob, axis=1)
    with np.errstate(under="ignore"):
        log_responsibility = weighted_log_prob - log_likelihood[:, np.newaxis]

    responsibility = np.exp(log_responsibility).T


    return (responsibility, log_likelihood) if full_output else responsibility



def _parameters_per_mixture(D, covariance_type):
    r"""
    Return the number of parameters per Gaussian component, given the number 
    of observed dimensions and the covariance type.

    :param D:
        The number of dimensions per data point.

    :param covariance_type:
        The structure of the covariance matrix for individual components.
        The available options are: `full` for a free covariance matrix, or
        `diag` for a diagonal covariance matrix.

    :returns:
        The number of parameters required to fully specify the multivariate
        mean and covariance matrix of a :math:`D`-dimensional Gaussian.
    """

    if covariance_type == "full":
        return int(D + D*(D + 1)/2.0)
    elif covariance_type == "diag":
        return 2 * D
    else:
        raise ValueError("unknown covariance type '{}'".format(covariance_type))


def _initialize_with_kmeans_pp(y, K, random_state=None, **kwargs):

    random_state = check_random_state(random_state)
    squared_norms = row_norms(y, squared=True)
    mu = cluster.k_means_._k_init(y, K, x_squared_norms=squared_norms,
        random_state=random_state)

    labels = np.argmin(scipy.spatial.distance.cdist(mu, y), axis=0)

    # generate repsonsibilities.
    N, D = y.shape
    responsibility = np.zeros((K, N))
    responsibility[labels, np.arange(N)] = 1.0

    # estimate covariance matrices.
    cov = _estimate_covariance_matrix_full(y, responsibility, mu)

    # If this is K = 1, then use this as the bound limit for sumlogdetcov
    weight = responsibility.sum(axis=1)/N

    return (mu, cov, weight, responsibility)



def _initialize(y, covariance_type, covariance_regularization, **kwargs):
    r"""
    Return initial estimates of the parameters.

    :param y:
        The data values, :math:`y`.

    :param covariance_type:
        The structure of the covariance matrix for individual components.
        The available options are: `full` for a free covariance matrix, or
        `diag` for a diagonal covariance matrix.

    :param covariance_regularization:
        Regularization strength to add to the diagonal of covariance matrices.


    :returns:
        A three-length tuple containing the initial (multivariate) mean,
        the covariance matrix, and the relative weight.
    """

    # If you *really* know what you're doing, then you can give your own.
    if kwargs.get("__initialize", None) is not None:
        return kwargs.pop("__initialize")

    weight = np.ones((1, 1))
    N, D = y.shape
    mean = np.mean(y, axis=0).reshape((1, -1))

    cov = _estimate_covariance_matrix(y, np.ones((1, N)), mean,
        covariance_type, covariance_regularization)

    visualization_handler = kwargs.get("visualization_handler", None)
    if visualization_handler is not None:
        I_other = _mixture_message_length_parts(weight.size, N, D)
        visualization_handler.emit(
            "model", 
            dict(mean=mean, cov=cov, weight=weight, I_other=I_other),
            snapshot=True)


    return (mean, cov, weight)
    

def _expectation(y, mu, cov, weight, **kwargs):
    r"""
    Perform the expectation step of the expectation-maximization algorithm.

    :param y:
        The data values, :math:`y`.

    :param mu:
        The current best estimates of the (multivariate) means of the :math:`K`
        components.

    :param cov:
        The current best estimates of the covariance matrices of the :math:`K`
        components.

    :param weight:
        The current best estimates of the relative weight of all :math:`K`
        components.

    :param N_component_pars:
        The number of parameters required to specify the mean and covariance
        matrix of a single Gaussian component.

    :returns:
        A three-length tuple containing the responsibility matrix,
        the  log likelihood, and the change in message length.
    """

    responsibility, log_likelihood = responsibility_matrix(
        y, mu, cov, weight, full_output=True, **kwargs)

    nll = -np.sum(log_likelihood)

    #I = _message_length(y, mu, cov, weight, responsibility, nll, **kwargs)
    K = weight.size
    N, D = y.shape
    slogdetcov = np.sum(np.linalg.slogdet(cov)[1])
    I = _mixture_message_length(K, N, D, -nll, slogdetcov, 
        weights=[weight])

    return (responsibility, log_likelihood, I)




def _compute_precision_cholesky(covariances, covariance_type):
    r"""
    Compute the Cholesky decomposition of the precision of the covariance
    matrices provided.

    :param covariances:
        An array of covariance matrices.

    :param covariance_type:
        The structure of the covariance matrix for individual components.
        The available options are: `full` for a free covariance matrix, or
        `diag` for a diagonal covariance matrix.
    """

    singular_matrix_error = "Failed to do Cholesky decomposition"

    if covariance_type in "full":
        M, D, _ = covariances.shape

        cholesky_precision = np.empty((M, D, D))
        for m, covariance in enumerate(covariances):
            try:
                cholesky_cov = scipy.linalg.cholesky(covariance, lower=True) 
            except scipy.linalg.LinAlgError:
                raise ValueError(singular_matrix_error)


            cholesky_precision[m] = scipy.linalg.solve_triangular(
                cholesky_cov, np.eye(D), lower=True).T

    elif covariance_type in "diag":
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(singular_matrix_error)

        cholesky_precision = covariances**(-0.5)

    else:
        raise NotImplementedError("nope")

    return cholesky_precision



def _estimate_covariance_matrix_full(y, responsibility, mean, 
    covariance_regularization=0):

    N, D = y.shape
    M, N = responsibility.shape

    membership = np.sum(responsibility, axis=1)

    I = np.eye(D)
    cov = np.empty((M, D, D))
    for m, (mu, rm, nm) in enumerate(zip(mean, responsibility, membership)):

        diff = y - mu
        denominator = nm - 1 if nm > 1 else nm

        cov[m] = np.dot(rm * diff.T, diff) / denominator \
               + covariance_regularization * I

    return cov


def _estimate_covariance_matrix(y, responsibility, mean, covariance_type,
    covariance_regularization):

    available = {
        "full": _estimate_covariance_matrix_full,
        "diag": _estimate_covariance_matrix_diag
    }

    try:
        function = available[covariance_type]

    except KeyError:
        raise ValueError("unknown covariance type")

    return function(y, responsibility, mean, covariance_regularization)

def _estimate_covariance_matrix_diag(y, responsibility, mean, 
    covariance_regularization=0):

    N, D = y.shape
    M, N = responsibility.shape

    denominator = np.sum(responsibility, axis=1)
    denominator[denominator > 1] = denominator[denominator > 1] - 1

    membership = np.sum(responsibility, axis=1)

    I = np.eye(D)
    cov = np.empty((M, D))
    for m, (mu, rm, nm) in enumerate(zip(mean, responsibility, membership)):

        diff = y - mu
        denominator = nm - 1 if nm > 1 else nm

        cov[m] = np.dot(rm, diff**2) / denominator + covariance_regularization

    return cov

    


def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):
    """Compute the log-det of the cholesky decomposition of matrices.
    Parameters
    ----------
    matrix_chol : array-like,
        Cholesky decompositions of the matrices.
        'full' : shape of (n_components, n_features, n_features)
        'tied' : shape of (n_features, n_features)
        'diag' : shape of (n_components, n_features)
        'spherical' : shape of (n_components,)
    covariance_type : {'full', 'tied', 'diag', 'spherical'}
    n_features : int
        Number of features.
    Returns
    -------
    log_det_precision_chol : array-like, shape (n_components,)
        The determinant of the precision matrix for each component.
    """
    if covariance_type == 'full':
        n_components, _, _ = matrix_chol.shape
        log_det_chol = (np.sum(np.log(
            matrix_chol.reshape(
                n_components, -1)[:, ::n_features + 1]), 1))

    elif covariance_type == 'tied':
        log_det_chol = (np.sum(np.log(np.diag(matrix_chol))))

    elif covariance_type == 'diag':
        log_det_chol = (np.sum(np.log(matrix_chol), axis=1))

    else:
        log_det_chol = n_features * (np.log(matrix_chol))

    return log_det_chol


def _estimate_log_gaussian_prob(X, means, precision_cholesky, covariance_type):
    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(
        precision_cholesky, covariance_type, n_features)

    if covariance_type in 'full':
        log_prob = np.empty((n_samples, n_components))
        for k, (mu, prec_chol) in enumerate(zip(means, precision_cholesky)):
            y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(y), axis=1)

    elif covariance_type in 'diag':
        precisions = precision_cholesky**2
        log_prob = (np.sum((means ** 2 * precisions), 1) - 2.0 * np.dot(X, (means * precisions).T) + np.dot(X**2, precisions.T))

    #return (-0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det, log_prob)
    return -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det



def _maximization(y, mu, cov, weight, responsibility, parent_responsibility=1,
    **kwargs):
    r"""
    Perform the maximization step of the expectation-maximization algorithm
    on all components.

    :param y:
        The data values, :math:`y`.

    :param mu:
        The current estimates of the Gaussian mean values.

    :param cov:
        The current estimates of the Gaussian covariance matrices.

    :param weight:
        The current best estimates of the relative weight of all :math:`K`
        components.

    :param responsibility:
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component.
    
    :param parent_responsibility: [optional]
        An array of length :math:`N` giving the parent component 
        responsibilities (default: ``1``). Only useful if the maximization
        step is to be performed on sub-mixtures with parent responsibilities.

    :returns:
        A three length tuple containing the updated multivariate mean values,
        the updated covariance matrices, and the updated mixture weights. 
    """

    M = weight.size 
    N, D = y.shape
    
    # Update the weights.
    effective_membership = np.sum(responsibility, axis=1)
    new_weight = (effective_membership + 0.5)/(N + M/2.0)

    w_responsibility = parent_responsibility * responsibility
    w_effective_membership = np.sum(w_responsibility, axis=1)

    new_mu = np.zeros_like(mu)
    new_cov = np.zeros_like(cov)
    for m in range(M):
        new_mu[m] = np.sum(w_responsibility[m] * y.T, axis=1) \
                  / w_effective_membership[m]

    new_cov = _estimate_covariance_matrix(y, responsibility, new_mu,
        kwargs["covariance_type"], kwargs["covariance_regularization"])

    state = (new_mu, new_cov, new_weight)

    assert np.all(np.isfinite(new_mu))
    assert np.all(np.isfinite(new_cov))
    assert np.all(np.isfinite(new_weight))
    
    return state 




def _expectation_maximization(y, mu, cov, weight, responsibility=None, **kwargs):
    r"""
    Run the expectation-maximization algorithm on the current set of
    multivariate Gaussian mixtures.

    :param y:
        A :math:`N\times{}D` array of the observations :math:`y`,
        where :math:`N` is the number of observations, and :math:`D` is the
        number of dimensions per observation.

    :param mu:
        The current estimates of the Gaussian mean values.

    :param cov:
        The current estimates of the Gaussian covariance matrices.

    :param weight:
        The current estimates of the relative mixing weight.

    :param responsibility: [optional]
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component. If ``None`` is given
        then the responsibility matrix will be calculated in the first
        expectation step.

    :param covariance_type: [optional]
        The structure of the covariance matrix for individual components.
        The available options are: `free` for a free covariance matrix,
        `diag` for a diagonal covariance matrix, `tied` for a common covariance
        matrix for all components, `tied_diag` for a common diagonal
        covariance matrix for all components (default: ``free``).

    :param threshold: [optional]
        The relative improvement in log likelihood required before stopping
        an expectation-maximization step (default: ``1e-5``).

    :param max_em_iterations: [optional]
        The maximum number of iterations to run per expectation-maximization
        loop (default: ``10000``).

    :returns:
        A six length tuple containing: the updated multivariate mean values,
        the updated covariance matrices, the updated mixture weights, the
        updated responsibility matrix, a metadata dictionary, and the change
        in message length.
    """   

    M = weight.size
    N, D = y.shape
    
    # Calculate log-likelihood and initial expectation step.
    _init_responsibility, ll, dl = _expectation(y, mu, cov, weight, **kwargs)

    if responsibility is None:
        responsibility = _init_responsibility

    iterations = 1
    ll_dl = [(ll.sum(), dl)]

    while True:

        # Perform the maximization step.
        mu, cov, weight \
            = _maximization(y, mu, cov, weight, responsibility, **kwargs)

        # Run the expectation step.
        responsibility, ll, dl \
            = _expectation(y, mu, cov, weight, **kwargs)

        # Check for convergence.
        lls = np.sum(ll)
        prev_ll, prev_dl = ll_dl[-1]
        relative_delta_message_length = np.abs((lls - prev_ll)/prev_ll)
        ll_dl.append([lls, dl])
        iterations += 1

        assert np.isfinite(relative_delta_message_length)

        if relative_delta_message_length <= kwargs["threshold"] \
        or iterations >= kwargs["max_em_iterations"]:
            break

    print("RAN {} E-M steps".format(iterations))

    meta = dict(warnflag=iterations >= kwargs["max_em_iterations"], log_likelihood=ll)
    if meta["warnflag"]:
        logger.warn("Maximum number of E-M iterations reached ({}) {}".format(
            kwargs["max_em_iterations"], kwargs.get("_warn_context", "")))

    return (mu, cov, weight, responsibility, meta, dl)


