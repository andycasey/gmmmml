
"""
Functions to calculate information of quantities.
"""

import numpy as np


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

    else:
        fraction = 0.5
        fraction_err = 0.16 # such that 3\sigma \approx 100% of range.
    
    target_lower, target_upper = bounds_of_sum_log_weights(K, N)

    target_ptp = target_upper - target_lower
    target_prediction = target_lower + fraction * target_ptp
    target_prediction_err = fraction_err * target_ptp

    return \
        (target_prediction, target_prediction_err, target_lower, target_upper)
