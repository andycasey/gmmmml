import numpy as np
import scipy.optimize as op

import george
from george import (kernels, modeling)

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
    
        \sum_{k=1}^{K}\log{w_k} \gteq (K - 1)\log{2} - K\log{N} + \log{(N - 2K + 2)}

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

        I_w \gteq C\left[(K - 1)\log{2} - K\log{N} + \log{(N - 2K + 2)}\right]

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
    Return the information content on the sum of the log of the weights:

    .. math:

        I_w = C\sum_{k=1}^{K}\log{w_k}

    where 

    .. math:

        C = \frac{D(D+3)}{4} - \frac{1}{2}

    :param sum_log_weights:
        The sum of the log of the weights for a Gaussian mixture.

    :param D:
        The dimensionality of the data.

    :returns:
        The information content of the sum of the log of the weights, in units
        of nats.
    """
    return (0.25 * D * (D + 3) - 0.5) * np.array(sum_log_weights)




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

    :returns:
        A four-length tuple containing:

            (1) the predicted information content of the sum of the log of the
                weights :math:`I_{w}` for each :math:`K` mixture;

            (2) the variance on the predicted information content on the sum of
                the log of the weights :math:`I_{w}` for each :math:`K` mixture;

            (3) the lower information bound on the sum of the log of the weights
                :math:`I_{w}` for each :math:`K` mixture;

            (4) the upper information bound on the sum of the log of the weights
                :math:`I_w` for each :math:`K` mixture.

        All predictions are given in units of nats.
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

    return (I, I_var, I_lower, I_upper)


def information_of_sum_log_det_covs(sum_log_det_covs, D):
    r"""
    Return the information content on the sum of the log of the determinant of
    the covariance matrices:

    .. math:

        I_c = C\sum_{k=1}^{K}\log\det{C_k}

    where 

    .. math:

        C = -\frac{1}{2}(D + 2)
    
    :param sum_log_det_covs:
        The sum of the log of the determinant for a given Gaussian mixture.

    :param D:
        The dimensionality of the data.

    :returns:

    """
    return -0.5 * (D + 2) * sum_log_det_covs


def predict_information_of_sum_log_det_covs(K, D, data):
    r"""
    Predict the information content of the sum of the log of the determinant of
    the covariance matrices for a Gaussian mixture model containing :math:`K`
    components. Specifically, the information is given by:

    .. math:

        I_C = -\frac{1}{2}(D + 2)\sum_{k=1}^{K}\log\det{C_k}

    :param K:
        The number of components in the target Gaussian mixture.

    :param D:
        The dimensionality of the data.

    :param data:
        A two-length tuple containing (1) the number of components in previously
        trialled Gaussian mixtures, and the sum of the log of the determinant of
        their covariance matrices.

    :returns:
        A two-length tuple containing:

            (1) the predicted information content of the sum of the log of the
                determinant of the covariance matrices for each :math:`K`
                mixture;

            (2) the variance on the predicted information content on the sum of
                the log of the determinant of the covariance matrices.
    """

    if data is None:
        raise NotImplementedError("cannot predict this theoretically")

    x, y = _group_over(data[0], data[1], np.mean)
    _, yerr = _group_over(data[0], data[1], np.std)

    kernel = np.var(y) * kernels.LinearKernel(order=1, log_gamma2=1) \
           + np.var(y) * kernels.ExpSquaredKernel(1)
        
    gp = george.GP(kernel=kernel, 
                   white_noise=np.log(np.std(y)), fit_white_noise=True)
                   
    def nll(p):
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(y, quiet=True)
        return -ll if np.isfinite(ll) else 1e25

    def grad_nll(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y, quiet=True)


    gp.compute(x.astype(float), yerr=np.ones_like(x))

    #print("Initial \log{{L}} = {:.2f}".format(gp.log_likelihood(y)))
    #print("Initial \grad\log{{L}} = {}".format(gp.grad_log_likelihood(y)))

    p0 = gp.get_parameter_vector()

    gp_result = op.minimize(nll, p0, jac=grad_nll, method="L-BFGS-B", tol=1e-8)

    #print("Final \log{{L}} = {:.2f}".format(gp.log_likelihood(y)))
    #print("Final \grad\log{{L}} = {}".format(gp.grad_log_likelihood(y)))

    pred, pred_var = gp.predict(y, K, return_var=True)

    I = information_of_sum_log_det_covs(pred, D)
    I_var = information_of_sum_log_det_covs(np.sqrt(pred_var), D)**2

    return (I, I_var)

    
    