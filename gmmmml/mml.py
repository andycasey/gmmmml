
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
