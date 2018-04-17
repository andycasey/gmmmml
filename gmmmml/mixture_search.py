
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
import warnings
import os
from sklearn import cluster
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms

from collections import defaultdict


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


def _bound_sum_log_weights(K, N):
    r"""
    Return the analytical bounds of the function:

    .. math:

        \sum_{k=1}^{K}\log{w_k}

    Where :math:`K` is the number of mixtures, and :math:`w` is a multinomial
    distribution. The bounded function for when :math:`w` are uniformly
    distributed is:

    .. math:

        \sum_{k=1}^{K}\log{w_k} \lteq -K\log{K}

    and in the other extreme case, all the weight would be locked up in one
    mixture, with the remaining :math:`w` values encapsulating the minimum
    (physically realistic) weight of one data point. In that extreme case,
    the bound becomes:

    .. math:

        \sum_{k=1}^{K}\log{w_k} \gteq log{(N + 1 - K)} - K\log{N}

    :param K:
        The number of target Gaussian mixtures.

    :param N:
        The number of data points.

    :returns:
        The lower and upper bound on :math:`\sum_{k=1}^{K}\log{w_k}`.
    """

    upper = -K * np.log(K)
    lower = np.log(N + 1 - K) - K * np.log(N)

    return (lower, upper)


'''
def _approximate_log_likelihood(N, D, K, logdetcovs, weights, actuals):
    """
    Calculate a first-order approximation of the log-likelihood for the
    :math:`K`-th mixture.

    :param K:
        The number of target Gaussian mixtures.

    :param N:
        The number of data points.

    :param D:
        The dimensionality of the data points.

    :param logdetcovs:
        The natural logarithm of the determinant of the covariance matrices of
        the :math:`K`-th mixtures. If :math:`K` is like a list, then the
        length of `logdetcovs` must match the size of `K`.

    :param weights:
        The weights of the :math:`K`-th mixtures. 

    """

    K = np.atleast_1d(K)
    log_likelihoods = np.zeros(K.size)

    assert len(K) == len(logdetcovs)
    assert len(K) == len(weights)

    # chi_r is, approximately, the estimated reduced chi-squared value for
    # data that are correctly fit by a component in the mixture model.
    est_red_chisq = 0

    for i, (k, ldcs, ws) in enumerate(zip(K, logdetcovs, weights)):

        log_likelihoods[i] = N * np.sum(ws * np.log(ws)) \
                           + N * np.sum(ws * ldcs) \
                           - 0.5 * N * D * (np.log(2 * np.pi) + est_red_chisq)




    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(K, -log_likelihoods, label="prediction", c='r')
    ax.scatter(K, -np.array(actuals))

    ax.set_ylabel('-log(L)')

    raise a


    for i, (k, logdetcov, weight) in enumerate(zip(K, logdetcovs, weights)):
        # most things will be chi-sq ~ 5 away (per dim) from most K means
        # TODO: my intuition is that this should be D * 5, but that doesn't
        # seem to come out through experiment.
        log_prob = np.ones((N, k)) * 3
        # just assign each object to one mixture, where we would expect the 
        # chi-sq value to be approximately 1 per dimension D
        for j, w in enumerate(weight):
            Nk = int(np.round(w * N))
            # TODO: I feel like this should be 1 * D,.... as per above.
            log_prob.T[j, np.arange(Nk)] = 1

        #log_prob[:, 0] = D

        weighted_log_prob = np.log(weight) + logdetcov \
            - 0.5 * (D * np.log(2 * np.pi) + log_prob)

        log_likelihoods[i] = np.sum(
            scipy.misc.logsumexp(weighted_log_prob, axis=1))

    raise a

    if not np.all(np.isfinite(log_likelihoods)):
        logger.warn("Non-finite predictions of the log-likelihood!")
    return log_likelihoods



def _log_likelihood_bound(K, N, D, logdetcov_bound, aggregate_function,
    chisq_scalar=1):

    scale = 1e6
    chisq_scalar = D
    K = np.atleast_1d(K)
    log_likelihoods = np.zeros((K.size, 2))
    
    for i, (k, logdetcov) \
    in enumerate(zip(K.astype(int), logdetcov_bound)):

        log_prob_uniform = np.ones((N, k)) * scale * chisq_scalar

        # Consider the first case where weights are uniformly distributed.
        uniform_weights = np.ones(k, dtype=float)/N
        Sk = 0
        for j, w in enumerate(uniform_weights):
            Nk = int(np.round(w * N))
            log_prob_uniform.T[j, np.arange(Sk, Sk + Nk)] = chisq_scalar
            Sk += Nk

        wlp_lu = np.log(uniform_weights) + logdetcov \
            - 0.5 * (D * np.log(2 * np.pi) + log_prob_uniform)
        ll_lu = np.sum(scipy.misc.logsumexp(wlp_lu, axis=1))
        
        log_prob_non_uniform = np.ones_like(log_prob_uniform) * scale * chisq_scalar

        non_uniform_weights = np.ones(k, dtype=float)
        non_uniform_weights[1:] = 1.0/N
        non_uniform_weights[0] = 1.0 - np.sum(non_uniform_weights[1:])

        Sk = 0
        for j, w in enumerate(non_uniform_weights):
            Nk = int(np.round(w * N))
            log_prob_non_uniform.T[j, np.arange(Sk, Sk + Nk)] = chisq_scalar
            Sk += Nk

        wlp_ln = np.log(non_uniform_weights) + logdetcov \
            - 0.5 * (D * np.log(2 * np.pi) + log_prob_uniform)
        
        ll_ln = np.sum(scipy.misc.logsumexp(wlp_ln, axis=1))
        
        log_likelihoods[i, :] = [ll_lu, ll_ln]

    # I think that it will always be the case that the uniform weight dist.
    # provides a lower bound, but just in case we calculate both.
    return aggregate_function(log_likelihoods, axis=1)

'''



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
    
    I_data = -log_likelihood - D * N * np.log(yerr)
    I_slogdetcovs = -0.5 * (D + 2) * slogdetcov
    I_weights = (0.25 * D * (D + 3) - 0.5) * slogw

    I_parts = dict(
        I_mixtures=I_mixtures, I_parameters=I_parameters, 
        I_data=I_data, I_slogdetcovs=I_slogdetcovs, 
        I_weights=I_weights)

    I = np.sum([I_mixtures, I_parameters, I_data, I_slogdetcovs, I_weights],
        axis=0) # [nats]

    return (I, I_parts)





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

    parameter_names = ("mean", "covariance", "weight")

    def __init__(self, covariance_type="full", covariance_regularization=0, 
        threshold=1e-5, max_em_iterations=10000, **kwargs):

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

        self._threshold = threshold
        self._max_em_iterations = max_em_iterations
        self._covariance_type = covariance_type
        self._covariance_regularization = covariance_regularization

        # Lists to record states for predictive purposes.
        self._state_K = []
        self._state_det_covs = []
        self._state_weights = []
        self._state_slog_weights = []
        self._state_slog_likelihoods = []

        self._state_predictions_K = []
        self._state_predictions_slog_det_covs = []
        self._state_predictions_slog_likelihoods = []
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


    def kmeans_search(self, y, K_max=None, **kwargs):

        kwds = dict(
            threshold=self._threshold, 
            max_em_iterations=self._max_em_iterations,
            covariance_type=self.covariance_type, 
            covariance_regularization=self._covariance_regularization,
            visualization_handler=None)
        kwds.update(kwargs)
        
        y = np.atleast_1d(y)

        N, D = y.shape
        K_max = N if K_max is None else K_max

        for K in range(1, K_max):

            #model = cluster.KMeans(n_clusters=K)
            #model.fit(y)
            #mu = model.cluster_centers_
            #labels = model.labels_

            # Assign everything to the closest thing.            
            mu = _initialize_with_kmeans_pp(y, K)
            labels = np.argmin(scipy.spatial.distance.cdist(mu, y), axis=0)

            # generate repsonsibilities.
            responsibility = np.zeros((K, N))
            responsibility[labels, np.arange(N)] = 1.0

            # estimate covariance matrices.
            cov = _estimate_covariance_matrix_full(y, responsibility, mu)

            # If this is K = 1, then use this as the bound limit for sumlogdetcov
            weight = responsibility.sum(axis=1)/N

            # Do one E-M step.
            try:
                R, ll, message_length = _expectation(y, mu, cov, weight, **kwds)

            except ValueError:
                logger.exception("Failed to calculate E-step")
                continue

            self._record_state_for_predictions(cov, weight, ll)

            mu, cov, weight = _maximization(
                y, mu, cov, weight, responsibility, **kwds)
            
            if kwds["visualization_handler"] is not None:

                K_predict = np.arange(1, weight.size + 25)
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
        K = np.arange(1, 11)
        self._predict_slogweights(K)
        self._predict_slogdetcovs(K)

        raise a





    def _predict_message_length(self, K, N, D, **kwargs):
        """
        Predict the message lengths of past or future mixtures.

        :param K:
            An array-like object of K-th mixtures to predict message lengths
            for.
        """

        K = np.atleast_1d(K)
        
        # Predict the sum of the log of the weights.
        p_slw, p_slw_err, slw_lower, slw_upper = self._predict_slogweights(K, N)
        
        # Predict sum of the log of the determinant of the covariance matrices.
        p_sldc, p_sldc_pos_err, p_sldc_neg_err = self._predict_slogdetcovs(K)

        # Predict log-likelihoods.
        p_nll = self._predict_negative_log_likelihood(K, N, D)

        # Calculate the other contributions to the message length.
        _z = np.zeros_like(K)
        _, I_parts = _mixture_message_length(K, N, D, _z, _z)
        I_other = I_parts["I_mixtures"] + I_parts["I_parameters"]

        I_slw_c = (0.25 * D * (D + 3) - 0.5)
        I_sldc_c = -0.5 * (D + 2)
            
        log_det_covs = np.log(np.hstack(self._state_det_covs))            
        lower_ldc = K * np.max(log_det_covs)
        upper_ldc = K * np.min(log_det_covs)

        nll_strict_bound, nll_relaxed_bound \
            = self._predict_lower_bounds_on_negative_log_likelihood(K, N, D)


        yerr = 0.001
        I_strict_lower_bound = I_other \
                             + I_slw_c * slw_lower \
                             + I_sldc_c * lower_ldc \
                             + nll_strict_bound \
                             - D * N * np.log(yerr)

        I_relaxed_lower_bound = I_other \
                              + I_slw_c * slw_lower \
                              + I_sldc_c * upper_ldc \
                              + nll_relaxed_bound \
                              - D * N * np.log(yerr)



        # Visualize predictions.
        visualization_handler = kwargs.get("visualization_handler", None)
        if visualization_handler is not None:

            # Show the other contributions to the message length.
            #visualization_handler.emit("predict_I_other",
            #    dict(K=K, I_other=I_other))

            # Show the information contributions from the weights.
            visualization_handler.emit("I_slw_bounds", dict(K=K, 
                lower=I_other + I_slw_c * slw_lower, 
                upper=I_other + I_slw_c * slw_upper))

            visualization_handler.emit("predict_I_slw",
                dict(K=K, p_slw=I_other + I_slw_c * p_slw, 
                    p_slw_err=I_other + I_slw_c * p_slw_err))


            # Show the information contributions from the sum of the log of the
            # determinant of the covariance matrices.
            visualization_handler.emit("I_sldc_bounds", dict(K=K,
                upper=I_sldc_c * upper_ldc, lower=I_sldc_c * lower_ldc))

            # Only show predictions for the sum of the log of the determinant
            # of the covariance matrices if it is valid.
            # TODO: Should we do this using Gaussian KDE instead?
            if p_sldc is not None \
            and np.all(np.isfinite(np.hstack([p_sldc, p_sldc_pos_err]))) \
            and np.sum((p_sldc + p_sldc_pos_err) <= (K * np.max(log_det_covs))) >= (K.size - 2):
                visualization_handler.emit("predict_I_slogdetcov",
                    dict(K=K, p_slogdetcovs=I_sldc_c * p_sldc,
                        p_slogdetcovs_pos_err=I_sldc_c * p_sldc_pos_err,
                        p_slogdetcovs_neg_err=I_sldc_c * p_sldc_neg_err))

                I_prediction = I_other \
                     + I_slw_c * p_slw \
                     + I_sldc_c * p_sldc \
                     + p_nll \
                     - D * N * np.log(yerr)

                index = np.argmin(I_prediction)
                print("Prediction for best K = {}".format(K[index]))

            else:
                I_prediction = np.nan * np.ones(K.size, dtype=float)

            # Predict the negative log likelihood of the mixtures.
            visualization_handler.emit("predict_nll", dict(K=K, p_nll=p_nll))

            visualization_handler.emit("predict_nll_bounds", dict(K=K,
                nll_strict_bound=nll_strict_bound, nll_relaxed_bound=nll_relaxed_bound))


            # Predict the total message length.
            visualization_handler.emit("predict_I_bounds", dict(K=K,
                I_strict_bound=I_strict_lower_bound,
                I_relaxed_bound=I_relaxed_lower_bound,
                p_I=I_prediction))




    def _record_state_for_predictions(self, cov, weight, log_likelihood):
        r"""
        Record 'best' trialled states (for a given K) in order to make some
        predictions about future mixtures.
        """

        self._state_K.append(weight.size)

        # Record determinates of covariance matrices.
        determinates = np.linalg.det(cov)
        self._state_det_covs.append(determinates)

        # Record sum of the log of the weights.
        self._state_weights.append(weight)
        self._state_slog_weights.append(np.sum(np.log(weight)))

        # Record log likelihood
        self._state_slog_likelihoods.append(np.sum(log_likelihood))



    def  _predict_slogweights(self, target_K, N):

        slw_lower, slw_upper = _bound_sum_log_weights(target_K, N)

        x_unique, y_unique = _group_over(
            self._state_K, self._state_slog_weights, np.min)

        if x_unique.size > 1:            
            function = lambda x, scale, constant: -x * scale * np.log(x) + constant
            p_opt, p_cov = op.curve_fit(function, x_unique, y_unique)

            pred = function(target_K, *p_opt)

            if x_unique.size > 5:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(2)
                ax = axes[0]
                ax.plot(x_unique, y_unique)
                ax.plot(target_K, slw_lower)
                ax.plot(target_K, slw_upper)

                def normit(values):
                    N = len(values)
                    return (values - slw_lower[:N])/(slw_upper[:N] - slw_lower[:N])

                ax = axes[1]
                ax.plot(x_unique, normit(y_unique), c='b')
                ax.plot(target_K, normit(slw_lower), c='r')
                ax.plot(target_K, normit(slw_upper), c='g')

                raise a

            if np.all(np.isfinite(p_cov)):
                draws = np.random.multivariate_normal(p_opt, p_cov, size=100)
                pred_err = np.percentile(
                    [function(target_K, *draw) for draw in draws], [16, 84], axis=0) \
                    - pred

            else:
                pred_err = np.nan * np.ones((2, target_K.size))

        else:
            pred = slw_upper
            pred_err = np.nan * np.ones((2, pred.size))

        return (pred, pred_err, slw_lower, slw_upper)


    def _predict_negative_log_likelihood(self, K, N, D):

        # Fractionally, how close are we to uniform mixtures?
        K = np.atleast_1d(K)
        Kp = np.array(self._state_K)
        upper = lambda K: N * np.log(K)
        lower = lambda K: N * np.log(N) - (N - K + 1) * np.log(N - K + 1)

        nswlw = - N * np.array([np.sum(w * np.log(w)) for w in self._state_weights])

        fractions = (nswlw - lower(Kp))/(upper(Kp) - lower(Kp))

        # TODO: Revisit this.
        # Just take the most recent fraction.
        # TODO: Should this default be zero of 1.0? What is more conservative?
        f = fractions[-1] if np.isfinite(fractions[-1]) else 1.0

        print("fraction {:.1f}".format(f))

        # What is the minimum log determinant of a covariance matrix observed to
        # date?

        # Draw some values from log|C|
        # TODO: Draw from 

        nll = 0.5 * N * D * np.log(2 * np.pi) \
            + lower(K) + f * (upper(K) - lower(K))


        if len(self._state_det_covs[-1]) > 1:
            kernel = scipy.stats.gaussian_kde(np.log(self._state_det_covs[-1]))
            # TODO: This assumes uniform weights but we know that might not be
            # true.
            for i, k in enumerate(K):
                w = 1.0/k
                nll -= N * np.sum(w * kernel(k))

        else:
            nll -= N * np.log(self._state_det_covs[0][0])

        # Compare to NLLs already observed?
        obs_nll = -np.array(self._state_slog_likelihoods)
        pre_nll = nll[:len(obs_nll)]

        chisqs = (obs_nll - pre_nll)/(0.5 * N * D)

        # chisqs/K w.r.t K will follow a 1/K**2 relation
        # y = a*k**-2

        x = 1.0/(1. + np.array(self._state_K))
        y = chisqs

        with warnings.catch_warnings():
            warnings.filterwarnings("error")

            try:
                coeff = np.polyfit(x, y, 2, w=x)

            except np.RankWarning:
                pred_chisqs = np.max(y)

            else:
                pred_chisqs = np.clip(np.polyval(coeff, 1.0/(1 + K)), 0, np.max(y))

        # predict chisq diff.
        nll += 0.5 * N * D * pred_chisqs


        if obs_nll.size >= 32:

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.scatter(x, chisqs)

            ax.plot(x, np.polyval(coeff, 1.0/(1 + x)), c='r')
        
            raise a

        #if np.all(nll >= np.max(obs_nll)):
        #    raise a
        return nll





    def _old_predict_lower_bounds_on_negative_log_likelihood(self, K, N, D, 
        obs_logdetcov_function=np.min):

        K = np.atleast_1d(K)

        # The two bounds for -N\sum^{K}w_{k}\log{w_{k}} are:
        #   (1): -N\sum^{K}w_{k}\log{w_{k}} = N\log{K}
        #   (2): -N\sum^{K}w_{k}\log{w_{k}} = N\log{N} - (N - K + 1)\log{N - K + 1}

        # Assuming N > K, (1) > (2) so (2) gives us the lower bound.

        nll = 0.5 * N * D * np.log(2 * np.pi) \
            + N * np.log(N) - (N - K + 1) * np.log(N - K + 1) 

        logdetcovs = np.log(np.hstack(self._state_det_covs))
        nll_bound_obs = nll - N * obs_logdetcov_function(logdetcovs)

        # Get better bounds for observed mixtures.
        maxlogdetcov = np.max(logdetcovs)
        nll_bound_theory = nll - N * maxlogdetcov
                
        for i, k in enumerate(K):
            if k not in self._state_K:
                continue

            index = self._state_K.index(k)

            nll_bound_theory[i] += N * maxlogdetcov - N * np.array()



        return (nll_bound_theory, nll_bound_obs)


    def _predict_lower_bounds_on_negative_log_likelihood(self, K, N, D):

        # For K values already trialled, use what came of it.
        K = np.atleast_1d(K)
        relaxed_bound = np.ones(K.size, dtype=float) * 0.5 * N * D * np.log(2 * np.pi) 

        logdetcovs = np.log(np.hstack(self._state_det_covs))

        min_logdetcov = np.min(logdetcovs)
        strict_bound = relaxed_bound - N * np.max(logdetcovs)

        for i, k in enumerate(K):

            try:
                index = self._state_K.index(k)

            except ValueError:
                relaxed_bound[i] -= N * min_logdetcov

            else:
                wts = self._state_weights[index]
                dcs = self._state_det_covs[index]

                relaxed_bound[i] -= N * np.sum(wts * (np.log(dcs) + np.log(wts)))

        return (strict_bound, relaxed_bound)


    def _lower_bounds_on_negative_log_likelihood_for_past_mixtures(self, N, D):

        sumwwldc = N * np.array([np.sum(w * (np.log(dc) + np.log(w))) \
            for w, dc in zip(self._state_weights, self._state_det_covs)])

        return 0.5 * N * D * np.log(2 * np.pi) - sumwwldc




    def _old_predict_log_likelihoods(self, target_K):

        x = np.array(self._state_K)
        y = np.array(self._state_slog_likelihoods)

        x_unique, y_unique = _group_over(x, y, np.max)
        normalization = y_unique[0]
        
        x_fit = x_unique
        y_fit = y_unique/normalization

        f_likelihood_improvement = lambda x, *p: p[0] * np.exp(x * p[1]) + p[2]


        def ln_prior(theta):
            if theta[0] > 1 or theta[0] < 0 or theta[1] > 0:
                return 0
            return -0.5 * (theta[0] - 0.5)**2 / 0.10**2

        def ln_like(theta, x, y):
            return -0.5  * np.sum((y - f_likelihood_improvement(x, *theta))**2)

        def ln_prob(theta, x, y):
            lp = ln_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + ln_like(theta, x, y)

        p0 = self._state_meta.get("_predict_log_likelihoods_p0", [0.5, -0.10, 0.5])

        p_opt, f, d = op.fmin_l_bfgs_b(lambda *args: -ln_prob(*args),
            fprime=None, x0=p0, approx_grad=True, args=(x_fit, y_fit))
        
        self._state_meta["_predict_log_likelihoods_p0"] = p0

        """
        raise a

        function = lambda x, *p: p[0] / np.exp(p[1] * x) + p[2]
        f = lambda x, *p: normalization * function(x, *p)

        p0 = np.ones(3)

        try:
            p_opt, p_cov = op.curve_fit(function, x_fit, y_fit, 
                p0=p0, maxfev=10000)
        except (RuntimeError, TypeError):
            return (None, None)

        pred = f(target_K, *p_opt)

        if np.all(np.isfinite(p_cov)):
            draws = np.random.multivariate_normal(p_opt, p_cov, size=100)
            pred_err = np.percentile(
                [f(target_K, *draw) for draw in draws], [16, 84], axis=0) \
                - pred

        else:
            pred_err = np.nan * np.ones((2, target_K.size))
        """
        pred = normalization * f_likelihood_improvement(target_K, *p_opt)
        pred_err = np.zeros_like(pred)

        return (pred, pred_err)



    def _predict_slogdetcovs(self, target_K, size=100):
        """
        Predict the sum of the log of the determinate of the covariance matrices 
        for future target mixtures.

        :param target_K:
            The K-th mixtures to predict the sum of the log of the determinant
            of the covariance matrices.

        :param size: [optional]
            The number of Monte Carlo draws to make when calculating the error
            on the sum of the log of the determinant of the covariance matrices.
        """

        K = np.array(self._state_K)
        sldc = np.array([np.sum(np.log(dc)) for dc in self._state_det_covs])

        # unique things.
        Ku = np.sort(np.unique(K))
        sldcu = np.array([np.median(sldc[K==k]) for k in K])

        if Ku.size <= 3:
            return (None, None, None)

        x, y = (Ku, sldcu/Ku)

        p0 = self._state_meta.get("_predict_slogdetcovs_p0", [y[0], 0.5, 0])
        try:
            p_opt, p_cov = op.curve_fit(_approximate_ldc_per_mixture_component,
                x, y, p0=p0, maxfev=100000, sigma=x.astype(float)**-2)

        except RuntimeError:
            return (None, None, None)

        # Save optimized point for next iteration.
        self._state_meta["_predict_slogdetcovs_p0"] = p_opt

        p_sldcu = np.array([
            target_K * _approximate_ldc_per_mixture_component(target_K, *p_draw)\
            for p_draw in np.random.multivariate_normal(p_opt, p_cov, size=size)])

        p50, p16, p84 = np.percentile(p_sldcu, [50, 16, 84], axis=0)

        u_pos, u_neg = (p84 - p50, p16 - p50)

        return (p50, u_pos, u_neg)


    

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
    weighted_log_prob = np.log(weight) + \
        _estimate_log_gaussian_prob(y, mu, precision_cholesky, covariance_type)

    log_likelihood = scipy.misc.logsumexp(weighted_log_prob, axis=1)
    with np.errstate(under="ignore"):
        log_responsibility = weighted_log_prob - log_likelihood[:, np.newaxis]

    
    responsibility = np.exp(log_responsibility).T
    
    if kwargs.get("dofail", False):
        raise a

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


def _initialize_with_kmeans_pp(y, K, random_state=None):

    random_state = check_random_state(random_state)
    squared_norms = row_norms(y, squared=True)
    return cluster.k_means_._k_init(y, K, x_squared_norms=squared_norms,
        random_state=random_state)




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
        visualization_handler.emit("model", dict(mean=mean, cov=cov, weight=weight,
            I_other=I_other))


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
    I, I_parts = _mixture_message_length(K, N, D, -nll, slogdetcov, 
        weights=[weight])

    visualization_handler = kwargs.get("visualization_handler", None)
    if visualization_handler is not None:
        visualization_handler.emit("expectation", dict(
            K=weight.size, message_length=I, responsibility=responsibility,
            log_likelihood=log_likelihood))

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
    
    visualization_handler = kwargs.get("visualization_handler", None)
    if visualization_handler is not None:
        I_other = _mixture_message_length_parts(new_weight.size, N, D)
        visualization_handler.emit("model", dict(mean=new_mu, cov=new_cov, 
            weight=new_weight, I_other=I_other))

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


