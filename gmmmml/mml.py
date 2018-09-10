import numpy as np
import scipy.optimize as op

# TODO: move to utils?

def _group_over(x, y, function):

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    x_unique = np.sort(np.unique(x))
    y_unique = np.nan * np.ones_like(x_unique)

    for i, xi in enumerate(x_unique):
        match = (x == xi)
        y_unique[i] = function(y[match])

    return (x_unique, y_unique)





def _bounds_of_sum_log_weights(K, N):
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
    (physically realistic) weight of two data points. In that extreme case,
    the bound becomes:

    .. math:




    :param K:
        The number of target Gaussian mixtures.

    :param N:
        The number of data points.

    :returns:
        The lower and upper bound on :math:`\sum_{k=1}^{K}\log{w_k}`.
    """

    upper = -K * np.log(K)
    lower = (K - 1) * np.log(2) - K * np.log(N) + np.log(N - 2 * K + 2)

    return (lower, upper)



def information_bounds_of_sum_log_weights(K, N, D):
    """
    Return the lower and upper bounds on the information of the sum of the log
    of the weights. Specifically, the lower bound is given when all weight is
    locked up in one mixture

    .. math:

        I_w \gteq C\log{(N + 1 - K)} - K\log{N}

    and the upper bound is given when the weights are uniformly distributed

    .. math:

        I_w \lteq -CK\log{K}
    
    where the constant :math:`C` in both cases is

    .. math:

        C = \frac{D(D + 3)}{4} - \frac{1}{2}

    The information bounds are given in units of nats.

    :param K:
        The number of Gaussian components in the mixture.

    :param N:
        The number of data points.

    :param D:
        The dimensionality of the data.

    :returns:
        A two-length tuple with the lower and upper information bounds on the
        sum of the log of the mixing weights, in units of nats.
    """

    return information_of_sum_log_weights(_bounds_of_sum_log_weights(K, N), D)


def information_of_sum_log_weights(sum_log_weights, D):
    r"""
    Return the information content on the sum of the log of the weights.

    # TODO:
    """

    constant = (0.25 * D * (D + 3) - 0.5)

    return constant * np.array(sum_log_weights)




def predict_information_of_sum_log_weights(K, N, D, data=None):
    r"""
    Predict the information content of the sum of the log of the weights of 
    components in a Gaussian mixture.

    :param K:
        The number of components in the target Gaussian mixture.

    :param N:
        The number of data points.

    :param D:
        The dimensionality of the data.

    :param data: [optional]
        If given, this should be a two-length tuple containing (1) the number of
        components in previously-trialled Gaussian mixtures, and (2) the sum of
        the log of the weights for that trialled Gaussian mixture.

    """

    I_lower, I_upper = information_bounds_of_sum_log_weights(K, N, D)

    _default_f, _default_f_var = (0.5, 0.5**2)

    if data is not None:
        k, sum_log_weights = _group_over(data[0], data[1], np.min)
        y = information_of_sum_log_weights(sum_log_weights, D)

        lower, upper = information_bounds_of_sum_log_weights(k, N, D)

        v = (y - lower)/(upper - lower)
        f, f_var = (np.nanmean(v), np.nanvar(v))        

    else:
        f, f_var = (_default_f, _default_f_var)

    f = f if np.isfinite(f) else _default_f
    f_var = f_var if np.isfinite(f_var) and f_var > 0 else _default_f_var

    # Now make predictions.
    lower, upper = information_bounds_of_sum_log_weights(K, N, D)
    I = lower + f * (upper - lower)
    I_var = (np.sqrt(f_var) * (upper - lower))**2

    print(f, f_var, _default_f, _default_f_var)

    return (I, I_var, I_lower, I_upper)

