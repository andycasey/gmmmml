import numpy as np
import scipy.optimize as op

import george
from george import (kernels, modeling)
from scipy.special import (gammaln, logsumexp)

# TODO: move to utils?
from . import utils


# TODO: improve these docs. they are not consistent and the math is borked.

def gaussian_mixture_message_length(K, N, D, log_likelihood, slogdetcov, weights):
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

        .. math::

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
    w_size = np.array([len(w) for w in weights])
    if not np.all(w_size == K):
        raise ValueError("the size of the weights does not match K")
    slogw = np.array([np.sum(np.log(w)) for w in weights])

    Q = gmm_number_of_parameters(K, D)

    I_mixtures = K * np.log(2) * (1 - D/2.0) + gammaln(K) \
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


    return I_parts


def gmm_component_contributions_to_message_length(responsibilities, 
                                                  log_likelihoods, 
                                                  covs, weights):
    """
    Return the component-wise contributions to the message length.
    """
    K, N = responsibilities.shape
    K, D, _ = covs.shape

    # TODO: Refactor with gaussian_mixture_message_length
    I_data = responsibilities @ -log_likelihoods

    Q = gmm_number_of_parameters(K, D)

    I_mixtures = K * np.log(2) * (1 - D/2.0) + gammaln(K) \
        + 0.25 * (2.0 * (K - 1) + K * D * (D + 3)) * np.log(N)
    I_parameters = 0.5 * np.log(Q * np.pi) - 0.5 * Q * np.log(2 * np.pi)
    

    I_slogdetcovs = -0.5 * (D + 2) * np.linalg.slogdet(covs)[1]
    I_weights = (0.25 * D * (D + 3) - 0.5) * np.log(weights)

    I_components = (I_mixtures + I_parameters)/K \
                 + I_data + I_slogdetcovs + I_weights

    return I_components


def gmm_number_of_parameters(K, D):
    r"""
    Return the total number of model parameters :math:`Q`, if a full 
    covariance matrix structure is assumed.

    .. math::

        Q = \frac{K}{2}\left[D(D+3) + 2\right] - 1


    :param K:
        The number of Gaussian mixtures.

    :param D:
        The dimensionality of the data.

    :returns:
        The total number of model parameters, :math:`Q`.
    """
    return (0.5 * D * (D + 3) * K) + K - 1



def information_of_mixture_constants(K, N, D):
    r"""
    Return the message length, or information, to encode the additional parameters
    for a Gaussian mixture model. Specifically this returns the sum of the
    information to encode the number of parameters and the constant terms
    related to the mixture itself:

    .. math::

        I_{mixture} = (1 - \frac{D}{2})K\log{2} + \Gamma\left(\log{K}\right)
                    + \frac{1}{4}(2(K - 1) + KD(D+3))\log{N}

    and

    .. math::

        I_{parameters} = \frac{1}{2}\log{Q\pi} - \frac{Q}{2}\log{2\pi}

    where :math:`Q` is the total number of model parameters

    .. math::

        Q = \frac{KD(D + 3)}{2} + K - 1

    :param K:
        The number of components in the target Gaussian mixture.

    :param N:
        The number of data points.

    :param D:
        The dimensionality of the data.
    """

    Q = gmm_number_of_parameters(K, D)

    I_mixtures = K * np.log(2) * (1 - D/2.0) + gammaln(K) \
        + 0.25 * (2.0 * (K - 1) + K * D * (D + 3)) * np.log(N)
    I_parameters = 0.5 * np.log(Q * np.pi) - 0.5 * Q * np.log(2 * np.pi)

    return (I_mixtures + I_parameters)


def _bounds_of_sum_log_weights(K, N):
    r"""
    Return the analytical bounds of the function:

    .. math::

        \sum_{k=1}^{K}\log{w_k}

    Where :math:`K` is the number of mixtures, and :math:`w` is a multinomial
    distribution. The bounded function for when :math:`w` are uniformly
    distributed is:

    .. math::

        \sum_{k=1}^{K}\log{w_k} \lteq -K\log{K}

    and in the other extreme case, all the weight would be locked up in one
    mixture, with the remaining :math:`w` values encapsulating the minimum
    (physically realistic) weight of two data points. In that extreme case,
    the bound becomes:

    .. math::
    
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
    r"""
    Return the lower and upper bounds on the information of the sum of the log
    of the weights. Specifically, the lower bound is given when all weight is
    locked up in one mixture

        .. math::

            I_w > C\left[(K - 1)\log{2} - K\log{N} + \log{(N - 2K + 2)}\right]

    and the upper bound is given when the weights are uniformly distributed

        .. math::

            I_w < -CK\log{K}
        
    where the constant :math:`C` in both cases is

        .. math::

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

    .. math::

        I_w = \left(\frac{D(D+3)}{4} - \frac{1}{2}\right)\sum_{k=1}^{K}\log{w_k}


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
        #  TODO a dictionary with required keys

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
        I_w = information_of_sum_log_weights(data["sum_log_weights"], D)

        # Chose best I_w based on the best I in each mixture.
        k, y = utils._best_mixture_parameter_values(data["K"], data["I"], I_w)

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

    .. math::

        I_C = -\frac{1}{2}(D + 2)\sum_{k=1}^{K}\log|{C_k}|

    where :math:`D` is the dimensionality of the data.
    
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

    .. math::

        I_C = -\frac{1}{2}(D + 2)\sum_{k=1}^{K}\log|{C_k}|

    :param K:
        The number of components in the target Gaussian mixture.

    :param D:
        The dimensionality of the data.

    :param data:
        A two-length tuple containing (1) the number of components in previously
        trialled Gaussian mixtures, and the determinant of the
        covariance matrices in each mixture.

    :returns:
        A three-length tuple containing:

            (1) the predicted information content of the sum of the log of the
                determinant of the covariance matrices for each :math:`K`
                mixture;

            (2) the variance on the predicted information content on the sum of
                the log of the determinant of the covariance matrices

            (3) the lower bound on the information content on the sum of the
                log of the determinant of the covariance matrices. This bound
                is not a theoretical bound: it is a lower bound based on the
                smallest determinant of the covariance matrices.
    """

    if data is None:
        raise NotImplementedError("cannot predict this theoretically")

    I_sldc = information_of_sum_log_det_covs(
        np.array([np.sum(np.log(dc)) for dc in data["det_covs"]]), D)

    x, y = utils._best_mixture_parameter_values(data["K"], data["I"], I_sldc)


    yerr = np.ones_like(y)

    var_y = np.var(y)
    var_y = var_y if (np.isfinite(var_y) and var_y > 0) else 1

    kernel = var_y * kernels.Matern32Kernel(1)#ExpSquaredKernel(1)

    white_noise = np.log(np.sqrt(var_y))

    gp = george.GP(kernel=kernel, 
                   mean=np.mean(y), fit_mean=True,
                   white_noise=white_noise, fit_white_noise=True)
                  
    gp.compute(x.astype(float), yerr=yerr)

    def nlp(p):
        gp.set_parameter_vector(p)
        lp = gp.log_likelihood(y, quiet=True) + gp.log_prior()
        return -lp if np.isfinite(lp) else 1e25

    def grad_nlp(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y, quiet=True)

    p0 = gp.get_parameter_vector()

    results = op.minimize(nlp, p0, method="L-BFGS-B")

    gp.set_parameter_vector(results.x)

    I, I_var = gp.predict(y, K, return_var=True)

    # Calculate the lower bound based on the data we have.
    max_log_det_cov = np.max(np.log(np.hstack(data["det_covs"])))
    min_log_det_cov = np.min(np.log(np.hstack(data["det_covs"])))


    I_lower = information_of_sum_log_det_covs(K * min_log_det_cov, D)
    I_upper = information_of_sum_log_det_covs(K * max_log_det_cov, D)

    # Calculate upper bound based on the two-point autocorrelation function

    return (I, I_var, I_lower, I_upper)


def predict_negative_sum_log_likelihood(K, N, D, data):
    r"""
    Predict the negative sum of the log likelihood for a Gaussian mixture model
    with :math:`K` components.

    :param K:
        The number of components in the target Gaussian mixture.

    :param data:
        A two-length tuple containing (1) the number of components in previously
        trialled Gaussian mixtures, and (2) the negative sum of the 
        log-likelihood of those Gaussian mixtures.

    :returns:
        A three-length tuple containing:

            (1) the predicted negative sum of the log likelihood for each
                :math:`K` mixture;

            (2) the variance in the predicted negative sum of the log likelihood

            (3) the lower bound on the negative sum of the log likelihood for
                each :math:`K` mixture.

    """

    x, y = utils._best_mixture_parameter_values(data["K"], data["I"],
        data["negative_log_likelihood"])

    yerr = np.ones_like(y)

    var_y = np.var(y)
    var_y = var_y if (var_y > 0 and np.isfinite(var_y)) else 1

    class MeanModel(modeling.Model):
        parameter_names = ("c", "a")

        def get_value(self, k):
            k = k.flatten()
            return self.c/k + self.a

    kernel = var_y * kernels.ExpSquaredKernel(1)#(order=1, log_gamma2=0)

    gp = george.GP(kernel=kernel,
                   mean=MeanModel(c=1, a=np.mean(y)), fit_mean=True,
                   white_noise=np.log(np.sqrt(var_y)), fit_white_noise=True)


    def nll(p):
        gp.set_parameter_vector(p)
        ll = gp.log_likelihood(y, quiet=True) + gp.log_prior()
        return -ll if np.isfinite(ll) else 1e25

    def grad_nll(p):
        gp.set_parameter_vector(p)
        return -gp.grad_log_likelihood(y, quiet=True)

    gp.compute(x.astype(float), yerr=yerr)

    p0 = gp.get_parameter_vector()

    results = op.minimize(nll, p0, method="L-BFGS-B")

    gp.set_parameter_vector(results.x)

    pred_nll, pred_nll_var = gp.predict(y, K, return_var=True)

    # Lower bound.
    lower = predict_lower_bound_on_negative_log_likelihood(K, N, D, data)

    return (pred_nll, pred_nll_var, lower)


def predict_lower_bound_on_negative_log_likelihood(K, N, D, data):

    # weights.
    # alternative: 
    """
    1): -log(K)

    2): (K - 1) * (2/N) * log(2/N) + (1 - 2(K - 1)/N)log(1 - 2(K-1)/N)
    """

    w_a = 2.0/N
    w_b = 1. - (K - 1) * w_a

    I_lower = -N * ((K - 1) * w_a * np.log(w_a) + w_b * np.log(w_b))

    # constants
    I_lower += 0.5 * N * D * np.log(2 * np.pi)

    # the chi squared distribution has D degrees of freedom AND I DONT KNOW WHY
    # TODO
    I_lower += 0.5 * N * D

    # concentration.


    # concentration
    #kt, nll, weights, det_covs = data
    kt = data["K"]
    weights = data["weights"]
    det_covs = data["det_covs"]

    mean_concentration = -0.5 * np.array(
        [np.sum(w*np.log(dc))/k for k, w, dc in zip(kt, weights, det_covs)])

    I_lower += 0.5 * N * K * np.max(mean_concentration)

    return I_lower

