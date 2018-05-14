
"""
Functions to calculate information of quantities.
"""

import logging
import numpy as np
import warnings
from collections import OrderedDict
from scipy import optimize as op, interpolate
from scipy.special import gammaln
from scipy.stats import gaussian_kde

from .utils import aggregate

def number_of_gmm_parameters(K, D):
    r"""
    Return the total number of model parameters :math:`Q`, if a full 
    covariance matrix structure is assumed, for a gaussian mixture model of
    :math:`K` mixtures in :math:`D` dimensions.

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


def _gmm_parameter_message_length(K, N, D):
    """
    Retrun the message length required to encode some of the easier parameters
    of the mixture.

    # TODO: Update docs.

    :param K:
        The number of Gaussian components in the mixture.

    :param N:
        The number of data points.

    :param D:
        The dimensionality of the data.
    """

    # TODO: update docs.
    #print("update docs for _gmm_parameter_message_length")

    Q = number_of_gmm_parameters(K, D)

    # Calculate information required to encode the mixture parameters,
    # regardless of the covariance matrices and the weights, etc.
    I_mixtures = K * np.log(2) * (1 - D/2.0) + gammaln(K) \
               + 0.25 * (2.0 * (K - 1) + K * D * (D + 3)) * np.log(N)
    I_parameters = 0.5 * np.log(Q * np.pi) - 0.5 * Q * np.log(2 * np.pi)

    return (I_mixtures, I_parameters)


def mixture_message_length(K, N, D, cov, weight, log_likelihood, yerr=0.001,
    **kwargs):
    r"""
    Return the message length of a Gaussian mixture model. 

    :param K:
        The number of Gaussian components in the mixture. 

    :param N:
        The number of data points.

    :param D:
        The dimensionality of the data.

    :param cov:
        The covariance matrices of the components in the mixture.

    :param weight:
        The relative weights of the components in the mixture.

    :param log_likelihood:
        The sum of the log likelihood for the data, given the model.

    :param yerr: [optional]
        The errors in each dimension for each of the data points. If an array
        is given then it must have the same size as `(N, D)`.

    :returns:
        A two-length tuple containing the total message length (in units of
        nats), and a dictionary containing the message length of different
        constitutents (all in units of nats).
    """

    K, N, D = np.array([K, N, D], dtype=int)

    yerr = np.atleast_1d(yerr)
    if yerr.size > 1 and yerr.size != (N * D):
        raise ValueError("yerr size does not match what is expected")

    # Allow for the sum of the log of the determinant of the covariance 
    # matrices to be directly given, to avoid duplicate computing
    if cov is None:
        sum_log_det_cov = kwargs["__slogdetcov"]
    else:
        _, log_det_cov = np.linalg.slogdet(cov)
        assert np.all(_ > 0), \
               "Unstable covariance matrices: negative log determinants"
        sum_log_det_cov = np.sum(log_det_cov)

    # Same for the weight.
    if weight is None:
        sum_log_weights = kwargs["__slogw"]
    else:
        sum_log_weights = np.sum(np.log(weight))

    I_mixtures, I_parameters = _gmm_parameter_message_length(K, N, D)

    I_yerr = -np.log(yerr) if yerr.size > 1 else - N * D * np.log(yerr)


    I_data = -log_likelihood + I_yerr
    I_slogdetcovs = -0.5 * (D + 2) * sum_log_det_cov
    I_weights = (0.25 * D * (D + 3) - 0.5) * sum_log_weights

    I_parts = dict(
        I_mixtures=I_mixtures, I_parameters=I_parameters, I_data=I_data,
        I_slogdetcovs=I_slogdetcovs, I_weights=I_weights)

    I = I_mixtures + I_parameters + I_data + I_slogdetcovs + I_weights #[nats]

    return (I, I_parts)


def bounds_of_sum_log_weights(K, N):
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


def predict_sum_log_weights(K, N, previous_states=None):
    """
    Predict the sum of the log of the weights for target (future) mixtures of
    Gaussians.

    The sum of the log of the weights of :math:`K` mixtures is bounded by the
    condition when all :math:`w` are uniformly distributed:

    .. math:

        \sum_{k=1}^{K}\log{w_k} \lteq -K\log{K}


    and in the other extreme when all the weight is locked up in one mixture,
    with the remaining :math:`w` values encapsulating the minimum (physically
    realistic) weight of one data point. In that extreme case the bound is:

    .. math:

        \sum_{k=1}^{K}\log{w_k} \gteq log{(N + 1 - K)} - K\log{N}


    :param K:
        The number of target Gaussian mixtures.

    :param N:
        The number of data points.

    :param previous_states: [optional]
        A two-length tuple containing the previous :math:`K` trials, and the
        sum of the log of the weights :math:`w` for each :math:`K'` mixture.
        If provided, these previous states are used to make predictions for
        future mixtures.

    :returns:
        A six-length tuple containing:

        (1) the prediction of the sum of the log weights for the :math:`K`
            target mixtures;

        (2) the error on the prediction of the sum of the log of the weights
            for the :math:`K` target mixtures;

        (3) the theoretical lower bound on the sum of the log of the weights
            for the :math:`K` target mixtures;

        (4) the theoretical upper bound on the sum of the log of the weights
            for the :math:`K` target mixtures;

        (5) the determined fraction between mixture uniformity (:math:`f = 1`)
            and non-uniformity (:math:`f = 0`);

        (6) the error on the fraction between mixture uniformity and 
            non-uniformity.
    """

    K = np.atleast_1d(K)
    N = int(N)

    fraction, fraction_err = (np.nan, np.nan)
    if previous_states is not None:
        # We will fit some fractional value between these bounds.
        
        previous_K, previous_weights = previous_states
        previous_slogw = np.array([np.sum(np.log(w)) for w in previous_weights])
        
        if previous_K.size != previous_slogw.size:
            raise ValueError("number of previous K does not match number of "
                             "previous sum of the log of weights")

        lower_states, upper_states = bounds_of_sum_log_weights(previous_K, N)

        # Normalize.
        normalised = (previous_slogw - lower_states) \
                   / (upper_states - lower_states)

        # Use nanmedian/nanstd because information for K = 1 is always zero.
        fraction = np.nanmedian(normalised)
        fraction_err = np.nanstd(normalised)

    if not np.all(np.isfinite([fraction, fraction_err])):
        fraction = 0.5
        fraction_err = 0.16 # such that 3\sigma \approx 100% of range.
    
    target_lower, target_upper = bounds_of_sum_log_weights(K, N)

    target_ptp = target_upper - target_lower
    target_prediction = target_lower + fraction * target_ptp

    # Bound the prediction errors by the theoretical bounds.
    target_prediction_err = fraction_err * target_ptp
    target_prediction_pos_err = - target_prediction + np.min([target_upper,
        target_prediction + target_prediction_err], axis=0)

    target_prediction_neg_err = - target_prediction + np.max([target_lower,
        target_prediction - target_prediction_err], axis=0)

    target_prediction_err = \
        (target_prediction_pos_err, target_prediction_neg_err)

    return (target_prediction, target_prediction_err, target_lower,
        target_upper, fraction, fraction_err)


def bounds_of_sum_log_det_covs(K, N, log_max_det, log_min_det, **kwargs):
    r"""
    Predict the theoretical bounds on the sum of the log of the determinant of
    the covariance matrices for future (target) mixtures with :math:`K`
    components.

    :param K:
        The number of target Gaussian mixtures.

    :param N:
        The number of data points.

    :param log_max_det:
        The logarithm of the determinant of the covariance matrix for the
        :math:`K = 1` mixture, which represents the largest amount of 
        (co-)variance that needs to be captured by the model.

    :param log_min_det:
        An estimate of the logarithm of the minimum determinant of a covariance
        matrix for some mixture, as determined by the smallest pair-wise
        distance between two points.

        # TODO improve docs here.

    """

    # This is the limiting case when the log_max_det must be ~uniformly
    # distributed over K mixtures. This represents the *lower* bound on the
    # information required to encode it (or the upper bound on the sum of the
    # log of the determinant of the covariance matrices).
    upper_bound = K * log_max_det - K * np.log(K)

    # This is the limiting case when every K-th mixture has a very small
    # covariance matrix. 
    lower_bound = K * log_min_det

    return (lower_bound, upper_bound)


def predict_sum_log_det_covs(K, N, previous_states, draws=100, **kwargs):
    r"""
    Predict the sum of the log of the determinant of covariance matrices for
    future (target) mixtures with :math:`K` components.

    :param K:
        The number of target Gaussian mixtures.

    :param N:
        The number of data points.

    :param previous_states:
        A two-length tuple containing the previous :math:`K` trials, and the
        determinant of the covariance matrices for each of those :math:`K`
        trials.

    :param draws: [optional]
        The number of error draws to make when predicting the sum of the log
        of the determinant of covariance matrices of future mixtures.

    :returns:
        A three-length tuple containing:

        (1) the predicted sum of the log of the determinant of covariance
            matrices for future (target) mixtures with :math:`K` components;

        (2) the error (positive, negative) on the predicted sum of the log of
            the determinant of covariance matrices for future (target) 
            mixtures with :math:`K` components;

        (3) a dictionary containing recommended entries to use to update the
            state of the metadata for the gaussian mixture model, in order
            to improve the optimization of future predictions.
    """

    K = np.atleast_1d(K)

    __failed_value = np.nan * np.ones(len(K))
    __default_value = (__failed_value, (__failed_value, __failed_value), dict())

    previous_K, previous_det_cov = previous_states
    
    # TODO: Assume the most recent previous_det_cov is representative of what
    #       we want to target. This won't be the case if we are jumping around
    #       between very different mixtures.
    
    if len(previous_det_cov[-1]) < 2:
        return __default_value

    # Let's predict the upper bound on this quantity.
    # TODO: move elsewhere if it works.

    # The upper bound is when the maximal sample variance is distributed 
    # among the different mixtures.
    log_max_det = np.log(previous_det_cov[0][0])
    
    # Case 1: Uniform spread in variance.
    #         This provides the true lower limit for the information content,
    #         because I = -\frac{(D + 2)}{2}\sum\log{|C_k|}
    upper_sum_log_det_cov = K * log_max_det - K * np.log(K)

    # Case 2: Non-uniform spread in variance, where the variance follows the
    #         weights.
    lower_sum_log_det_cov = K * log_max_det - K * np.log(N) + np.log(N - K + 1)


    raise a

    p_sldc = np.zeros(K.size)

    return (p_sldc, (lower_sum_log_det_cov, upper_sum_log_det_cov), {})







def _deprecated_predict_sum_log_det_covs(K, previous_states, draws=100, **kwargs):
    r"""
    Predict the sum of the log of the determinant of covariance matrices for
    future (target) mixtures with :math:`K` components.

    :param K:
        The number of target Gaussian mixtures.

    :param previous_states:
        A two-length tuple containing the previous :math:`K` trials, and the
        determinant of the covariance matrices for each of those :math:`K`
        trials.

    :param draws: [optional]
        The number of error draws to make when predicting the sum of the log
        of the determinant of covariance matrices of future mixtures.

    :returns:
        A three-length tuple containing:

        (1) the predicted sum of the log of the determinant of covariance
            matrices for future (target) mixtures with :math:`K` components;

        (2) the error (positive, negative) on the predicted sum of the log of
            the determinant of covariance matrices for future (target) 
            mixtures with :math:`K` components;

        (3) a dictionary containing recommended entries to use to update the
            state of the metadata for the gaussian mixture model, in order
            to improve the optimization of future predictions.
    """

    __failed_value = np.nan * np.ones(len(K))
    __default_value = (__failed_value, (__failed_value, __failed_value), dict())

    previous_K, previous_det_cov = previous_states
    previous_sldc = np.array([np.sum(np.log(dc)) for dc in previous_det_cov])

    previous_K_u, previous_sldc_u \
        = aggregate(previous_K, previous_sldc, np.median)

    if previous_K_u.size <= 3:
        return __default_value

    x, y = (previous_K_u, previous_sldc_u/previous_K_u)

    p0 = kwargs.get("_predict_sum_log_det_covs_p0", [y[0], 0.5, 0])
    
    __approximate_ldc_per_mixture = lambda k, *p: p[0]/(k - p[1]) + p[2]

    try:
        p_opt, p_cov = op.curve_fit(__approximate_ldc_per_mixture, x, y,
                                    p0=p0, maxfev=100000, 
                                    sigma=x.astype(float)**-2)

    except RuntimeError:
        logging.exception(
            "Failed to predict sum of the log of the determinant of the "\
            "covariance matrices for future mixtures:")
        return __default_value

    update_state = dict(_predict_sum_log_det_covs_p0=p_opt)

    if np.all(np.isfinite(p_cov)):        
        p_sldc = np.array([K * __approximate_ldc_per_mixture(K, *p_draw) \
            for p_draw in np.random.multivariate_normal(p_opt, p_cov, size=draws)])

        prediction, p16, p84 = np.percentile(p_sldc, [50, 16, 84], axis=0)
        prediction_err = (p84 - prediction, p16 - prediction)

    else:
        prediction = K * __approximate_ldc_per_mixture(K, *p_opt)
        prediction_err = (np.zeros_like(prediction), np.zeros_like(prediction))

    return (prediction, prediction_err, update_state)


def predict_negative_log_likelihood(K, N, D, predicted_uniformity_fraction,
    previous_states, **kwargs):
    r"""
    Predict the sum of the negative log likelihood for future (target)
    mixtures with :math:`K` components.

    :param K:
        The number of target Gaussian mixtures.

    :param N:
        The number of data points.

    :param D:
        The dimensionality of the data points.

    :param predicted_uniformity_fraction:
        The predicted uniformity fraction of the true mixture, based on the
        weights of previously computed mixtures.

    :param previous_states:
        A four-length tuple containing:

        (1) previous trialled :math:`K` values;

        (2) previous computed weights for each of the :math:`K` mixtures

        (2) the determinates of covariance matrices for previously-trialled
            mixtures

        (3) the sum of the log likelihoods for previously-trialled mixtures
    """


    
    _state_K, _state_weights, _state_det_covs, _state_sum_log_likelihoods \
        = previous_states

    obs_nll = -np.array(_state_sum_log_likelihoods)
    P = obs_nll.size
    K = np.atleast_1d(K)
    


    """
    Approximate the function:

    .. math:
        
        -N\sum_{k=1}^{K} w_k\log{w_k}

    Using the predicted uniformity fraction, taking the two extrema:

    (1) The mixture is uniform and all weights are equal such that.

    .. math:
        
        -N\sum_{k=1}^{K} w_k\log{w_k} = N\log{K}

    (2) There are K - 1 mixtures that each describe two data points, and the
        remaining mixture describes the rest of the data. In this extreme:

    .. math:
        
        -N\sum_{k=1}^{K} w_k\log{w_k} = (2K - 2 - N)\log{(N - 2K + 2)} + 2(1 - K)\log{2} + N\log{N}

    """


    if P > 3:

        # Fit and predict the concentration.
        y = np.array([np.sum(w * np.log(dc)) \
            for w, dc in zip(_state_weights, _state_det_covs)])

        x = K[:y.size].astype(float)
        objective_function = lambda x, *p: p[0] * np.exp(-x/p[1]) + p[2]

        p_opt, p_cov = op.curve_fit(objective_function, x, y, p0=[y[0], 10, 0])

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.plot(K, objective_function(K, *p_opt), c='r')

        concentration = lambda K: objective_function(K, *p_opt)

    else:
        # Just get the last concentration.
        concentration = lambda K: \
            np.sum(_state_weights[-1] * np.log(_state_det_covs[-1]))

    upper = lambda K: N * np.log(K)
    lower = lambda K: N * np.log(N) + 2*(1 - K)*np.log(2) \
                    + (2*K - 2 - N) * np.log(N - 2*K + 2)

    # Calculate the NLL without any contribution by chi-squared values (yet).
    pre_nll = 0.5 * N * D * np.log(2 * np.pi) + lower(K) \
            + predicted_uniformity_fraction * (upper(K) - lower(K)) \
            + concentration(K)

    if P > 3:
        # Predict the improvement with mean reduced \chi^2 value.
        y = (obs_nll - pre_nll[:P]) / (0.5 * N * D)
        x = K[:y.size].astype(float)

        objective_function = lambda x, *p: p[0] * np.exp(-x/p[1]) + p[2]
        p0 = [y[0], 10, 0]

        p_mrc_opt, p_mrc_cov = op.curve_fit(objective_function, x, y, p0=p0)

        mean_reduced_chisq = lambda K: objective_function(K, *p_mrc_opt)

    else:
        mean_reduced_chisq = lambda K: (obs_nll[-1] - pre_nll[P]) / (0.5 * N * D)

    pre_nll += 0.5 * N * D * mean_reduced_chisq(K)


    """
    # Calculate the lower bound on the negative log-likelihood.
    concentration = N * np.sum(_state_weights[-1] * np.log(_state_det_covs[-1]))
    nll_practical_lower_bound = lower(K) + predicted_uniformity_fraction * (upper(K) - lower(K)) \
        + 0.5 * N * D * np.log(2*np.pi) - concentration

    nll_theoretical_lower_bound = lower(K) + 0.5 * N * D * np.log(2*np.pi) \
                                - N * np.sum(_state_weights[0] * np.log(_state_det_covs[0]))
    """

    #nll_practical_lower_bound = lower(K) \
    #    + predicted_uniformity_fraction * (upper(K) - lower(K)) \
    #    + lower_concentration + 0.5 * N * D * (np.log(2*np.pi) + 1)

    # Make a prediction for the actual log-likelihood. 
    #predict_offset = 0.5 * N * D * (np.log(2 * np.pi) + mean_chisq)
    
    #nll = mN_sum_wlogw + predict_concentration + predict_offset



    if len(obs_nll) > 30:

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(K[:pre_nll.size], pre_nll, c='r')
        ax.plot(K[:obs_nll.size], obs_nll, c='k')

        fig, ax = plt.subplots()
        ax.plot(K[:diff_cac.size], diff_cac)

        raise a
        #moo6 = N * np.sum(weights * np.log(weights)) \
        #     + N * np.sum(weights * log_det) \
        #     - 0.5 * N * D * (np.log(2 * np.pi) + 1)

        approx_nll = N * np.array([np.sum(w * np.log(w * dc)) for w, dc in zip(_state_weights, _state_det_covs)]) \
                   + 0.5 * N * D * (np.log(2 * np.pi) + 0)

        diff_cac = (approx_nll - obs_nll)/(0.5 * N * D)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        ax.plot(K[:approx_nll.size], approx_nll)
        ax.plot(K[:obs_nll.size], obs_nll)

        fig, ax = plt.subplots()
        ax.plot(K[:obs_nll.size], diff_cac)

        concentration = np.array([np.sum(w * np.log(w*dc)) for w, dc in zip(_state_weights, _state_det_covs)])

        fig, ax = plt.subplots()
        ax.plot(K[:concentration.size], concentration)

        raise a
        M = len(_state_weights)
        predict_nll_parts = 0.5 * N * D * np.log(2 * np.pi) + lower(K[:M]) + predicted_uniformity_fraction * (upper(K[:M]) - lower(K[:M])) \
                          + np.array([np.sum(w * np.log(dc)) for w, dc in zip(_state_weights, _state_det_covs)])

        fig, ax = plt.subplots()
        ax.plot(K[:M], predict_nll_parts)
        ax.plot(K[:obs_nll.size], obs_nll, c='k')

        fig, ax = plt.subplots()
        diff_cac = (obs_nll - predict_nll_parts)/(0.5 * N * D)
        ax.plot(K[:M], diff_cac)

        fig, ax = plt.subplots()
        ax.plot(K[:M], np.array([np.sum(w * np.log(dc)) for w, dc in zip(_state_weights, _state_det_covs)]))

        raise a


    """

    if len(obs_nll) > 20:

        # Fit this.
        x, y = 1.0/(K[:P] + 1), diff_cac
        Q = min(4, len(obs_nll) - 1)
        p0 = np.hstack([1, np.zeros(Q)])

        function = lambda x, *params: np.polyval(params, x)

        p_opt, p_cov = op.curve_fit(function, x[2:], y[2:], p0=p0, sigma=x[2:])

        pre_nll[:P] += 0.5 * N * D * diff_cac
        pre_nll[P:] += 0.5 * N * D * function(1.0/(K[P:] + 1), *p_opt)

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        concentration = N * np.array([np.sum(w * np.log(dc)) for w, dc in zip(_state_weights, _state_det_covs)])

        ax.plot(K[:concentration.size], concentration)


        
        #approx_nll = -N * np.sum(weight * (np.log(weight) + np.linalg.det(cov))) \
        #           + 0.5 * N * D * (np.log(2 * np.pi) + 0)

        approx_nll = -N * np.array([np.sum(w * (np.log(w) + np.log(dc))) for w, dc in zip(_state_weights, _state_det_covs)]) \
                   + 0.5 * N * D * (np.log(2 * np.pi) + 0)

        fig, ax = plt.subplots()

        ax.plot(K[:approx_nll.size], approx_nll)
        ax.plot(K[:obs_nll.size], obs_nll)

        raise a

    """

    """
    if K.max() > 70:

        function = lambda x, *params: np.polyval(params, x)

        x = np.array(1.0/(1 + K[:len(diff_cac)]), dtype=float)

        import scipy.optimize as op
        p0 = [0, 1, 0]
        p_opt, p_cov = op.curve_fit(function, x, diff_cac, p0=p0)


        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.scatter(K[:len(diff_cac)], diff_cac)

        ax.plot(K[:len(diff_cac)], function(x, *p_opt), c='r')


        raise a
    """

    """
    _deprecated_previous_states = [
        _state_K,
        _state_det_covs,
        _state_sum_log_likelihoods
    ]
    _deprecated_nll, _deprecated_nll_theorical_lower_bound = \
    _deprecated_predict_negative_log_likelihood(K, N, D, predicted_uniformity_fraction,
        previous_states=_deprecated_previous_states)
    """

    nll_theoretical_lower_bound = np.zeros_like(K)
    nll_practical_lower_bound = np.zeros_like(K)
    return (pre_nll, nll_theoretical_lower_bound, nll_practical_lower_bound)





def _deprecated_predict_negative_log_likelihood(K, N, D, uniformity_fraction,
    previous_states, **kwargs):
    r"""
    Predict the negative log likelihood for future (target) mixtures with
    :math:`K` components.

    :param K:
        The number of target Gaussian mixtures.

    :param N:
        The number of data points.

    :param D:
        The dimensionality of the data points.

    :param previous_states:
        A four-length tuple containing:

        (1) previous trialled :math:`K` values;

        (2) the determinates of covariance matrices for previously-trialled
            mixtures;

        (3) the sum of the log likelihoods for previously-trialled mixtures.
    """

    print("_deprecated_predict_negative_log_likelihood")

    K = np.atleast_1d(K)

    _state_K, _state_det_covs, _state_sum_log_likelihoods = previous_states
    
    # Use the predicted uniformity fraction.
    upper = lambda K: N * np.log(K)
    lower = lambda K: N * np.log(N) - (N - K + 1) * np.log(N - K + 1)

    # Calculate the first part of the negative log-likelihood.
    nll = 0.5 * N * D * np.log(2 * np.pi) \
        + lower(K) + uniformity_fraction * (upper(K) - lower(K))

    # Draw some determinants..
    print("EXTRACT THE STATES")
    if len(_state_det_covs[-1]) > 1:
        # Use a Gaussian KDE to estimate the determinants of covariance
        # matrices for future mixtures.
        kernel = gaussian_kde(np.log(_state_det_covs[-1]))

        # TODO: This assumes uniform 
        print("STOP ASSUMING UNIFORMITY")
        for i, k in enumerate(K):
            w = 1.0/k
            
            print("ANDY CHECK THIS")
            # check this.
            nll -= N * np.sum(w * kernel(k))

    else:
        nll -= N * np.log(_state_det_covs[0][0])

    # The current calculation of the negative log-likelihood is based on the
    # ideal case where the mean chi-squared value is the limiting case of zero
    # (e.g., a perfect fit).

    # In practice this prediction is going to be wrong for earlier K than the
    # true K value, because the chi-squared value is going to be non-zero.

    # Let's compare our predicted negative log-likelihood values to that which
    # was determined from previous trials, in order to see how quickly we are
    # improving, and how we can adjust our prediction to take this improving
    # fit into account.

    # Compare to what is already observed.
    obs_nll = -np.array(_state_sum_log_likelihoods)
    pre_nll = nll[:len(obs_nll)] # assumes one trial per K.
    print("stop assuming one trial per K")

    # The difference between our observed negative log likelihoods and the
    # predicted negative log-likelihoods is the mean chi-squared value for
    # each observation (weighted over each of the mixtures).
    chisqs = (obs_nll - pre_nll)/(0.5 * N * D)

    x = 1.0/(1.0 + _state_K)
    y = chisqs

    with warnings.catch_warnings():
        warnings.filterwarnings("error")

        try:
            coeff = np.polyfit(x, y, 2, w=x.astype(float))

        except np.RankWarning:
            pred_chisqs = np.max(y)

        else:
            pred_chisqs = np.clip(np.polyval(coeff, 1.0/(1.0 + K)), 0, np.max(y))

    # Predict difference based on chi-sq improvement through increasing number
    # of mixtures.
    nll_lower_bound = nll.copy()
    nll += 0.5 * N * D * pred_chisqs

    #nll_lower_bound = N * np.log(N) + (K - 1 - N) * np.log(N - K + 1) \
    #                + N * K * D * np.log(min_mean_pairwise_distance/2.0) \
    #                + 0.5 * N * D * (1 + np.log(2 * np.pi))

    return (nll, nll_lower_bound)



def predict_message_length(K, N, D, previous_states, yerr=0.001, 
    min_mean_pairwise_distance=np.nan, state_meta=None, **kwargs):
    """
    Predict the message length of past or future mixtures.

    :param K:
        An array-like object of the :math:`K`-th mixtures to predict the
        message elngths of.

    :param N:
        The number of data points.

    :param D:
        The dimensionality of the data points.
    """

    # TODO: update docs

    state_meta = state_meta or {}

    _state_K, _state_weights, _state_det_covs, _state_sum_log_likelihoods = previous_states

    _state_K = np.array(_state_K)
    

    K = np.atleast_1d(K)

    # Predict the sum of the log of the weights.
    p_slogw, p_slogw_err, t_slogw_lower, t_slogw_upper, uniformity_fraction, \
        uniformity_fraction_err = predict_sum_log_weights(K, N, 
            previous_states=(_state_K, _state_weights))

    # Predict the sum of the log of the determinant of the covariance
    # matrices.
    log_max_det = np.log(_state_det_covs[0][0])
    log_min_det = np.log(np.min(np.hstack(_state_det_covs)))
    #log_min_det = np.log(min_mean_pairwise_distance)

    t_slogdetcov_lower, t_slogdetcov_upper = bounds_of_sum_log_det_covs(
        K, N, log_max_det, log_min_det)


    p_slogdetcov, p_slogdetcov_err, update_meta = _deprecated_predict_sum_log_det_covs(
        K, N=N, previous_states=(_state_K, _state_det_covs), **state_meta)

    p_nll, _, t_nll_lower = predict_negative_log_likelihood(
        K, N, D, uniformity_fraction, previous_states=(
            _state_K, 
            _state_weights,
            _state_det_covs,
            _state_sum_log_likelihoods
        ))

    # From these quantities, calculate the predicted parts of the message
    # lengths for the future mixtures.
    I_mixtures, I_parameters = _gmm_parameter_message_length(K, N, D)
    I_other = I_mixtures + I_parameters

    # Let's group things together that are bound by theory, or have analytic
    # expressions.
    slw_scalar = 0.25 * D * (D + 3) - 0.5
    p_I_analytic = I_other + slw_scalar * p_slogw

    p_slogw_pos_err, p_slogw_neg_err = p_slogw_err
    p_I_analytic_pos_err = slw_scalar * p_slogw_pos_err
    p_I_analytic_neg_err = slw_scalar * p_slogw_neg_err

    t_I_analytic_lower = I_other + slw_scalar * t_slogw_lower 
    t_I_analytic_upper = I_other + slw_scalar * t_slogw_upper

    # Now we will make predictions for the sum of the log of the determinant
    # of the covariance matrices.
    sldc_scalar = -0.5 * (D + 2)
    p_I_slogdetcov = sldc_scalar * p_slogdetcov
    p_slogdetcov_pos_err, p_slogdetcov_neg_err = p_slogdetcov_err
    p_I_slogdetcov_pos_err = sldc_scalar * p_slogdetcov_pos_err
    p_I_slogdetcov_neg_err = sldc_scalar * p_slogdetcov_neg_err
    t_I_slogdetcov_upper = sldc_scalar * t_slogdetcov_lower
    t_I_slogdetcov_lower = sldc_scalar * t_slogdetcov_upper

    # The predictions for the negative log-likelihood are already in units of
    # nats, so nothing needed there. But we do need to incorporate the errors
    # on y.

    # TODO: better way to encode yerr?
    p_I_data = p_nll - D * N * np.log(yerr)
    t_I_data_lower = t_nll_lower - D * N * np.log(yerr)
    p_I = p_I_analytic + p_I_slogdetcov + p_I_data

    t_I_lower = t_I_analytic_lower + t_I_slogdetcov_lower + t_I_data_lower

    #assert np.all(t_I_analytic_lower <= t_I_analytic_upper)
    #if np.isfinite(t_I_slogdetcov_upper).all():
    #    assert np.all(t_I_slogdetcov_upper >= t_I_slogdetcov_lower)

    predictions = OrderedDict([
        ("K", K),
        ("p_I_analytic", p_I_analytic),
        ("p_I_analytic_pos_err", p_I_analytic_pos_err),
        ("p_I_analytic_neg_err", p_I_analytic_neg_err),
        ("t_I_analytic_lower", t_I_analytic_lower),
        ("t_I_analytic_upper", t_I_analytic_upper),
        ("t_I_slogdetcov_lower", t_I_slogdetcov_lower),
        ("t_I_slogdetcov_upper", t_I_slogdetcov_upper),
        ("p_I_slogdetcov", p_I_slogdetcov),
        ("p_I_slogdetcov_pos_err", p_I_slogdetcov_pos_err),
        ("p_I_slogdetcov_neg_err", p_I_slogdetcov_neg_err),
        ("p_nll", p_nll),
        ("t_nll_lower", t_nll_lower),
        ("t_I_data", t_I_data_lower),
        ("p_I_data", p_I_data),
        ("p_I", p_I),
        ("t_I_lower", t_I_lower)
    ])


    return (predictions, update_meta)
