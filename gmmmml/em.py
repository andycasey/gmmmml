
"""
Expectation-Maximization.
"""

import logging
import numpy as np
import scipy.linalg
from scipy.special import logsumexp
from tqdm import tqdm

from .mml import (gaussian_mixture_message_length,
                  gmm_component_contributions_to_message_length)

logger = logging.getLogger(__name__)


def expectation(y, means, covs, weights, **kwargs):
    r"""
    Perform the expectation step of the expectation-maximization algorithm.

    :param y:
        The data values, :math:`y`, which are expected to have :math:`N` samples
        each with :math:`D` dimensions. Expected shape of ``y`` is 
        :math:`(N, D)`.

    :param means:
        The current estimate of the multivariate means of the :math:`K`
        components. The expected shape of ``means`` is :math:`(K, D)`.

    :param covs:
        The current estimate of the covariance matrices of the :math:`K`
        components. The expected shape of ``covs`` is :math:`(K, D, D)`.

    :param weights:
        The current estimate of the relative weights :math:`w` of all :math:`K`
        components. The sum of weights must equal 1. The expected shape of
        ``weights`` is :math:`(K, )`.

    :returns:
        A three-length tuple containing the responsibility matrix,
        the log-likelihood, and a dictionary containing the message lengths of
        various parts of the mixture.
    """

    kwds = kwargs.copy()
    kwds.pop("full_output", False)

    R, ll = responsibilities(y, means, covs, weights, 
                             full_output=True, **kwds)

    K, N, D = (weights.size, *y.shape)

    I = gaussian_mixture_message_length(K, N, D, np.sum(ll), 
                                        np.sum(np.linalg.slogdet(covs)[1]),
                                        [weights])

    return (R, ll, I)


def _component_expectations(y, means, covs, weights, **kwargs):

    R, ll = responsibilities(y, means, covs, weights, 
                             full_output=True, **kwargs)


    I_components = gmm_component_contributions_to_message_length(R, ll, covs, weights)

    return (R, ll, I_components)
    

def maximization(y, means, covs, weights, responsibilities, 
                 parent_responsibilities=1, **kwargs):
    r"""
    Perform the maximization step of the expectation-maximization algorithm
    on all components.

    :param y:
        The data values, :math:`y`, which are expected to have :math:`N` samples
        each with :math:`D` dimensions. Expected shape of ``y`` is 
        :math:`(N, D)`.

    :param means:
        The current estimate of the multivariate means of the :math:`K`
        components. The expected shape of ``means`` is :math:`(K, D)`.

    :param covs:
        The current estimate of the covariance matrices of the :math:`K`
        components. The expected shape of ``covs`` is :math:`(K, D, D)`.

    :param weights:
        The current estimate of the relative weights :math:`w` of all :math:`K`
        components. The sum of weights must equal 1. The expected shape of
        ``weights`` is :math:`(K, )`.

    :param responsibilities:
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component. The expected shape of
        `responsibilities` is :math:`(N, K)`.
    
    :param parent_responsibilities: [optional]
        An array of length :math:`N` giving the parent component 
        responsibilities (default: ``1``). Only useful if the maximization
        step is to be performed on sub-mixtures with parent responsibilities.

    :returns:
        A three length tuple containing the new estimate on the component means,
        the component covariance matrices, and the relative mixture weights.
    """

    K, N, D = (weights.size, *y.shape)
    
    effective_membership = np.sum(responsibilities, axis=1)
    weights_ = (effective_membership + 0.5)/(N + K/2.0)

    weighted_responsibilities = parent_responsibilities * responsibilities
    w_effective_membership = np.sum(weighted_responsibilities, axis=1)

    means_, covs_ = (np.zeros_like(means), np.zeros_like(covs))
    for k in range(K):
        means_[k] = np.sum(weighted_responsibilities[k] * y.T, axis=1) \
                  / w_effective_membership[k]

    covs_ = _estimate_covariance_matrix(y, means_, responsibilities, **kwargs)

    return (means_, covs_, weights_)



def expectation_maximization(y, means, covs, weights, covariance_type="full", 
                             covariance_regularization=0, threshold=1e-2, 
                             max_em_iterations=100, quiet=False, **kwargs):
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

    e_step = kwargs.pop("__expectation_function", expectation)
    m_step = kwargs.pop("__maximization_function", maximization)
    callback_step = kwargs.pop("__callback_function", None)

    R = kwargs.pop("responsibilities", None)

    kwds = dict(covariance_type=covariance_type,
                covariance_regularization=covariance_regularization)
    kwds.update(kwargs)

    state = (means, covs, weights)
    responsibilities, ll, I = e_step(y, *state, **kwds)

    # Overwrite responsibilities if explicitly given.
    if R is not None: responsibilities = R

    prev_I = np.sum(np.hstack(I.values()))

    tqdm_kwds = dict(disable=True) if quiet \
                                   else dict(desc=f"E-M @ K={weights.size}")

    for iteration in tqdm(range(max_em_iterations), **tqdm_kwds):
        
        state = m_step(y, *state, responsibilities, **kwds)
        responsibilities, ll, I = e_step(y, *state, **kwds)

        current_I = np.sum(np.hstack(I.values()))
        diff = np.abs(prev_I - current_I)
        if diff <= threshold or not np.isfinite(diff):
            break

        prev_I = current_I

    else:
        logger.warning(
            f"Convergence not reached ({diff:.1e} > {threshold:.1e}) "\
            f"after {iteration + 1} iterations")

    if callback_step is not None:
        callback_step(state, R, ll, I)
    return (state, responsibilities, ll, I)


def responsibilities(y, means, covs, weights, covariance_type="full",
                     full_output=False, **kwargs):
    r"""
    Return the responsibility matrix,

    .. math::

        r_{ij} = \frac{w_{j}f\left(y_i;\theta_j\right)}{\sum_{k=1}^{K}{w_k}f\left(y_i;\theta_k\right)}


    where :math:`r_{ij}` denotes the conditional probability of a datum
    :math:`x_i` belonging to the :math:`j`-th component. The effective 
    membership associated with each component is then given by

    .. math::

        n_j = \sum_{i=1}^{N}r_{ij}
        \quad \textrm{and} \quad
        \sum_{k=1}^{K}n_{k} = N

    
    :param means:
        The current estimate of the multivariate means of the :math:`K`
        components. The expected shape of ``means`` is :math:`(K, D)`.

    :param covs:
        The current estimate of the covariance matrices of the :math:`K`
        components. The expected shape of ``covs`` is :math:`(K, D, D)`.

    :param weights:
        The current estimate of the relative weights :math:`w` of all :math:`K`
        components. The sum of weights must equal 1. The expected shape of
        ``weights`` is :math:`(K, )`.

    :param covariance_type: [optional]
        The structure of the covariance matrices. Available types include:

            - "full": full covariance matrix with non-zero off-diagonal terms

            - "diag": diagonal covariance matrix.

    :param full_output: [optional]
        If ``True``, return the responsibility matrix, and the log likelihood,
        which is evaluated for free (default: ``False``).

    :returns:
        The responsibility matrix. If ``full_output`` is ``True``, then the
        log-likelihood (per observation) will also be returned.
    """

    precision_cholesky = _compute_precision_cholesky(covs, covariance_type)
    lp = _gaussian_log_prob(y, means, weights, precision_cholesky,
                            covariance_type)
        
    ll = logsumexp(lp, axis=1)
    with np.errstate(under="ignore"):
        log_R = lp - ll[:, np.newaxis]

    R = np.exp(log_R).T

    return (R, ll) if full_output else R


def _compute_precision_cholesky(covariances, covariance_type):
    r"""
    Compute the Cholesky decomposition of the precision of the covariance
    matrices provided.

    :param covariances:
        An array of covariance matrices. Given :math:`K` covariance matrices
        that have :math:`D` dimensions, the expected shape of ``covariances``
        depends on ``covariance_type``:

            - "full": :math:`(K, D, D)`

            - "diag": :math:`(K, D)`

    :param covariance_type:
        The structure of the covariance matrix for individual components.
        The available options are:

        -   "full" for a free covariance matrix, or
        -   "diag" for a diagonal covariance matrix.

    :returns:
        The Cholesky decomposition of the precision of the covariance matrices.
    """

    singular_matrix_error = "Failed to do Cholesky decomposition"

    if covariance_type in "full":
        K, D, _ = covariances.shape

        I = np.eye(D)

        cholesky_precision = np.empty((K, D, D))
        for k, covariance in enumerate(covariances):
            try:
                cholesky_cov = scipy.linalg.cholesky(covariance, lower=True) 

            except scipy.linalg.LinAlgError:
                raise ValueError(singular_matrix_error)

            cholesky_precision[k] = scipy.linalg.solve_triangular(
                                    cholesky_cov, I, lower=True).T

    elif covariance_type in "diag":
        if np.any(np.less_equal(covariances, 0.0)):
            raise ValueError(singular_matrix_error)
        cholesky_precision = covariances**(-0.5)

    else:
        raise NotImplementedError(f"unknown covariance type '{covariance_type}'")

    return cholesky_precision


def _estimate_covariance_matrix(y, means, responsibilities, covariance_type,
                                covariance_regularization=0, **kwargs):
    r"""
    Estimate the covariance matrix given the data, the responsibility matrix,
    and an estimate of the means of the Gaussian components.

    :param y:
        The data values, :math:`y`, which are expected to have :math:`N` samples
        each with :math:`D` dimensions. Expected shape of ``y`` is 
        :math:`(N, D)`.

    :param means:
        The current estimate of the multivariate means of the :math:`K`
        components. The expected shape of ``means`` is :math:`(K, D)`.

    :param responsibilities:
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component. The expected shape of
        ``responsibilities`` is :math:`(N, K)`.

    :param covariance_type:
        The structure of the covariance matrices. Available types include:

            - "full": full covariance matrix with non-zero off-diagonal terms

            - "diag": diagonal covariance matrix.

    :param covariance_regularization: [optional]
        A regularization term that is added to the diagonal of the covariance
        matrices. Default is 0.

    :returns:
        An estimate of the covariance matrices of the Gaussian components.
    """

    available = {
        "full": _estimate_covariance_matrix_full,
        "diag": _estimate_covariance_matrix_diag
    }

    try:
        function = available[covariance_type]

    except KeyError:
        raise ValueError(f"unknown covariance type '{covariance_type}'")

    return function(y, means, responsibilities, covariance_regularization)


def _estimate_covariance_matrix_full(y, means, responsibilities,
                                     covariance_regularization=0):
    r"""
    Estimate the covariance matrix given the data, the responsibility matrix,
    and an estimate of the means of the Gaussian components. The covariance
    matrix is assumed to be full rank.

    :param y:
        The data values, :math:`y`, which are expected to have :math:`N` samples
        each with :math:`D` dimensions. Expected shape of ``y`` is 
        :math:`(N, D)`.

    :param means:
        The current estimate of the multivariate means of the :math:`K`
        components. The expected shape of ``means`` is :math:`(K, D)`.

    :param responsibilities:
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component. The expected shape of
        ``responsibilities`` is :math:`(N, K)`.

    :param covariance_regularization: [optional]
        A regularization term that is added to the diagonal of the covariance
        matrices. Default is 0.

    :returns:
        An estimate of the covariance matrices of the Gaussian components.
    """

    N, D, = y.shape
    K, N = responsibilities.shape

    membership = np.sum(responsibilities, axis=1)

    I = np.eye(D)
    covs = np.empty((K, D, D))
    for k, (mean, R, M) in enumerate(zip(means, responsibilities, membership)):
        diff = y - mean
        denominator = M - 1 if M > 1 else M

        covs[k] = np.dot(R * diff.T, diff) / denominator \
                + covariance_regularization * I

    return covs


def _estimate_covariance_matrix_diag(y, means, responsibilities,
                                     covariance_regularization=0):
    r"""
    Estimate the covariance matrix given the data, the responsibility matrix,
    and an estimate of the means of the Gaussian components. The covariance
    matrix is assumed to have zero elements in off-diagonal entries.

    :param y:
        The data values, :math:`y`, which are expected to have :math:`N` samples
        each with :math:`D` dimensions. Expected shape of ``y`` is 
        :math:`(N, D)`.

    :param means:
        The current estimate of the multivariate means of the :math:`K`
        components. The expected shape of ``means`` is :math:`(K, D)`.

    :param responsibilities:
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component. The expected shape of
        ``responsibilities`` is :math:`(N, K)`.

    :param covariance_type:
        The structure of the covariance matrices. Available types include:

            - "full": full covariance matrix with non-zero off-diagonal terms

            - "diag": diagonal covariance matrix.

    :param covariance_regularization: [optional]
        A regularization term that is added to the diagonal of the covariance
        matrices. Default is 0.

    :returns:
        An estimate of the covariance matrices of the Gaussian components.
    """

    N, D = y.shape
    K, N = responsibilities.shape

    denominator = np.sum(responsibilities, axis=1)
    denominator[denominator > 1] = denominator[denominator > 1] - 1
    memberships = np.sum(responsibilities, axis=1)

    covs = np.empty((K, D))
    for k, (mean, R, M) in enumerate(zip(means, responsibilities, memberships)):
        diff = y - mean
        denominator = M - 1 if M > 1 else M
        covs[k] = np.dot(R, diff**2) / denominator + covariance_regularization

    return covs

    

def _log_det_cholesky(cholesky_decomposition, covariance_type, D):
    r"""
    Compute the log-determinant of the Cholesky decomposition of matrices.

    :param cholesky_decomposition:
        The Cholesky decomposition of the covariance matrices.

    :param covariance_type:
        The structure of the covariance matrices. Available types include:

            - "full": full covariance matrix with non-zero off-diagonal terms

            - "diag": diagonal covariance matrix.

    :param D:
        The dimensionality of the covariance matrices.

    :returns:
        The log-determinant of the precision matrix for each covariance matrix.
    """

    if covariance_type == "full":
        K, _, _ = cholesky_decomposition.shape
        log_det_chol = (np.sum(np.log(
            cholesky_decomposition.reshape(
                K, -1)[:, ::D + 1]), 1))

    elif covariance_type == "tied":
        log_det_chol = (np.sum(np.log(np.diag(cholesky_decomposition))))

    elif covariance_type == "diag":
        log_det_chol = (np.sum(np.log(cholesky_decomposition), axis=1))

    else:
        log_det_chol = D * (np.log(cholesky_decomposition))

    return log_det_chol


def _gaussian_log_prob(y, means, weights, precision_cholesky, covariance_type):
    r"""
    Estimate the weighted log-probability of a Gaussian mixture, given the data.

    :param y:
        The data values, :math:`y`, which are expected to have :math:`N` samples
        each with :math:`D` dimensions. Expected shape of ``y`` is 
        :math:`(N, D)`.

    :param means:
        The current estimate of the multivariate means of the :math:`K`
        components. The expected shape of ``means`` is :math:`(K, D)`.

    :param weights:
        The current estimate of the relative weights :math:`w` of all :math:`K`
        components. The sum of weights must equal 1. The expected shape of
        ``weights`` is :math:`(K, )`.

    :param precision_cholesky:
        The Cholesky decomposition of the precision of the covariance
        matrices of the mixture.

    :param covariance_type:
        The structure of the covariance matrices. Available types include:

            - "full": full covariance matrix with non-zero off-diagonal terms

            - "diag": diagonal covariance matrix.

    :returns:
        The weighted log-probability of the data, given the model.
    """

    N, D = y.shape
    K, D = means.shape

    # Remember: det(precision_chol) is half of det(precision)
    log_det = _log_det_cholesky(precision_cholesky, covariance_type, D)

    if covariance_type in 'full':
        log_prob = np.empty((N, K))
        for k, (mean, prec_chol) in enumerate(zip(means, precision_cholesky)):
            diff = np.dot(y, prec_chol) - np.dot(mean, prec_chol)
            log_prob[:, k] = np.sum(np.square(diff), axis=1)

    elif covariance_type in 'diag':
        precisions = precision_cholesky**2
        log_prob = (np.sum((means**2 * precisions), 1) \
                 - 2.0 * np.dot(y, (means * precisions).T) \
                 + np.dot(y**2, precisions.T))

    ll = -0.5 * (D * np.log(2 * np.pi) + log_prob) + log_det
    return np.log(weights) + ll

