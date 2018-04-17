
"""
Functions to calculate information of quantities.
"""

import numpy as np
from scipy.special import gammaln

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


def mixture_message_length(N, D, K, cov, weight, log_likelihood, yerr=0.001,
    **kwargs):
    r"""
    Return the message length of a Gaussian mixture model. 

    :param N:
        The number of data points.

    :param D:
        The dimensionality of the data.

    :param K:
        The number of Gaussian components in the mixture. 

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


    I_yerr = -np.log(yerr) if yerr.size > 1 else - N * D * np.log(yerr)

    Q = number_of_gmm_parameters(K, D)

    # Calculate information required to encode the mixture parameters,
    # regardless of the covariance matrices and the weights, etc.
    I_mixtures = K * np.log(2) * (1 - D/2.0) + gammaln(K) \
               + 0.25 * (2.0 * (K - 1) + K * D * (D + 3)) * np.log(N)
    I_parameters = 0.5 * np.log(Q * np.pi) - 0.5 * Q * np.log(2 * np.pi)

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
        A four-length tuple containing:

        (1) the prediction of the sum of the log weights for the :math:`K`
            target mixtures;

        (2) the error on the prediction of the sum of the log of the weights
            for the :math:`K` target mixtures;

        (3) the theoretical lower bound on the sum of the log of the weights
            for the :math:`K` target mixtures;

        (4) the theoretical upper bound on the sum of the log of the weights
            for the :math:`K` target mixtures.

    """

    K = np.atleast_1d(K)
    N = int(N)

    fraction, fraction_err = (np.nan, np.nan)
    if previous_states is not None:
        # We will fit some fractional value between these bounds.
        
        previous_K, previous_slogw = previous_states
        previous_K = np.atleast_1d(previous_K)
        previous_slogw = np.atleast_1d(previous_slogw)

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
    target_prediction_pos_err = np.min([target_upper,
        target_prediction + target_prediction_err], axis=0)
    target_prediction_neg_err = np.max([target_lower,
        target_prediction - target_prediction_err], axis=0)

    target_prediction_err = \
        (target_prediction_pos_err, target_prediction_neg_err)

    return \
        (target_prediction, target_prediction_err, target_lower, target_upper)



def predict_sum_log_det_covs(K, previous_states, draws=100, **kwargs):
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

    print("Andy: update method for sum_log_det_covs to use a gaussian KDE")

    previous_K, previous_det_cov = previous_states
    previous_K = np.array(previous_K)
    previous_sldc = np.array([np.sum(np.log(dc)) for dc in previous_det_cov])

    previous_K_u, previous_sldc_u \
        = aggregate(previous_K, previous_sldc, np.median)

    if previous_K_u.size <= 3:
        return (None, None, None)

    x, y = (previous_K_u, previous_sldc_u/previous_K_u)

    p0 = kwargs.get("_predict_sum_log_det_covs_p0", [y[0], 0.5, 0])
    
    __approximate_ldc_per_mixture = lambda k, *p: p[0]/(k - p[1]) + p[2]

    try:
        p_opt, p_cov = op.curve_fit(__approximate_ldc_per_mixture, x, y,
                                    p0=p0, maxfev=100000, 
                                    sigma=x.astype(float)**-2)

    except RuntimeError:
        return (None, None, None)

    update_state = dict(_predict_sum_log_det_covs_p0=p_opt)

    p_sldc = np.array([K * __approximate_ldc_per_mixture(K, *p_draw) \
        for p_draw in np.random.multivariate_normal(p_opt, p_cov, size=draws)])

    prediction, p16, p84 = np.percentile(p_sldc, [50, 16, 84], axis=0)
    prediction_err = (p84 - p50, p16 - p50)

    return (prediction, prediction_err, update_state)