
"""
Functions related to expectation-maximization steps.
"""

import numpy as np
import scipy
from sklearn import cluster
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms

from .mml import mixture_message_length


def _initialise_by_kmeans_pp(y, K, covariance_regularization=0, 
    random_state=None):
    """
    Initialise by k-means++ and assign hard responsibilities to the closest
    centroid.

    :param y:
        The data :math:`y`.

    :param K:
    `   The number of Gaussian mixtures to initialise with.
    
    :param random_state: [optional]
        The state to use for the random number generator.

    :param covariance_regularization: [optional]


    :returns:
        A four-length tuple containing:

        (1) the initialised centroids :math:`\mu`;

        (2) the initialsied covariance matrices :math:`C`;

        (3) the initialised weights for each mixture :math:`w`;

        (4) the responsibility matrix.
    """

    if 1 > K:
        raise ValueError("the number of mixtures must be a positive integer")

    K = int(K)
    y = np.atleast_2d(y)
    N, D = y.shape

    random_state = check_random_state(random_state)
    squared_norms = row_norms(y, squared=True)

    mu = cluster.k_means_._k_init(y, K, x_squared_norms=squared_norms,
        random_state=random_state)

    # Assign everything to the closest mixture.
    labels = np.argmin(scipy.spatial.distance.cdist(mu, y), axis=0)

    # Generate responsibility matrix.
    responsibility = np.zeros((K, N))
    responsibility[labels, np.arange(N)] = 1.0

    # Calculate weights.
    weight = np.sum(responsibility, axis=1)/N

    # Estimate covariance matrices.
    cov = _estimate_covariance_matrix_full(y, responsibility, mu, 
        covariance_regularization=covariance_regularization)

    return (mu, cov, weight, responsibility)


def _compute_log_det_cholesky(cholesky_matrices, covariance_type, n_features):
    r"""
    Compute the log-determinant of the Cholesky decomposition of the given
    matrices.
    
    
    :param cholesky_matrices:
        The Cholesky decomposition of the matrices. The expected shape of this
        array depends on the `covariance_type`:

        - 'full': shape of (`n_components`, `n_features`, `n_features`)

        - 'tied': shape of (`n_features`, `n_features`)

        - 'diag': shape of (`n_components`, `n_features`)

        - 'spherical': shape of (`n_components`, )

    :param covariance_type:
        The type of the covariance matrix. Must be one of 'full', 'tied',
        'diag', or 'spherical'.

    :param n_features:
        The number of features.

    :returns:
        The log of the determinant of the precision matrix for each component.
    """

    if covariance_type == "full":
        n_components, _, _ = cholesky_matrices.shape
        log_det_chol = (np.sum(np.log(
            cholesky_matrices.reshape(
                n_components, -1)[:, ::n_features + 1]), 1))

    elif covariance_type == "tied":
        log_det_chol = (np.sum(np.log(np.diag(cholesky_matrices))))

    elif covariance_type == "diag":
        log_det_chol = (np.sum(np.log(cholesky_matrices), axis=1))

    elif covariance_type == "spherical":
        log_det_chol = n_features * (np.log(cholesky_matrices))

    else:
        raise ValueError("unrecognised covariance type")

    return log_det_chol



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

    :returns:
        The Cholesky decomposition of the precision of the given covariance
        matrices.
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
        raise NotImplementedError("covariance type not recognised")

    return cholesky_precision


def _estimate_log_gaussian_prob(y, mu, precision_cholesky, covariance_type):
    r"""
    Estimate the log of the Gaussian probability of the mixture model, for
    each datum given each mixture.

    :param y:
        The data :math;`y`.

    :param mu:
        The mixture centroids :math:`\mu`.

    :param precision_cholesky:
        The Cholesky decomposition of the precision of the covariance matrices
        for the given mixtures.

    :param covariance_type:
        The covariance matrix type: must be either 'full', 'tied', 'diag', or 
        'spherical'.

    :returns:
        The log of the gaussian probability for each datum belonging to each
        component in the mixture.
    """

    N, D = y.shape
    K, _ = mu.shape

    # Get this tattooed: det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(precision_cholesky, covariance_type, D)

    if covariance_type in "full":
        log_prob = np.empty((N, K))
        for k, (mu, prec_chol) in enumerate(zip(mu, precision_cholesky)):
            diff = np.dot(y, prec_chol) - np.dot(mu, prec_chol)
            log_prob[:, k] = np.sum(np.square(diff), axis=1)

    elif covariance_type in "diag":
        precisions = precision_cholesky**2
        log_prob = (np.sum((mu ** 2 * precisions), 1) \
                 - 2.0 * np.dot(y, (mu * precisions).T) \
                 + np.dot(X**2, precisions.T))

    else:
        raise ValueError("unrecognised covariance type")

    return -0.5 * (D * np.log(2 * np.pi) + log_prob) + log_det
    

def _estimate_covariance_matrix_diag(y, responsibility, mu, 
    covariance_regularization=0):
    r"""
    Estmiate the diagonal covariance matrix for the given data and mixtures.
    Where relevant, The Neyman-Scott correction is applied.

    :param y:
        The data :math:`y`.

    :param responsibility:
        The responsibility matrix.

    :param mu:
        The centroids of the mixtures.

    :param covariance_regularization: [optional]
        The regularization to apply to the diagonal of the covariance matrices
        in oder to maintain stability (default = 0).

    :returns:
        The estimated covariance matrices of the :math:`K` mixtures.
    """

    N, D = y.shape
    M, N = responsibility.shape

    denominator = np.sum(responsibility, axis=1)
    denominator[denominator > 1] = denominator[denominator > 1] - 1

    membership = np.sum(responsibility, axis=1)

    I = np.eye(D)
    cov = np.empty((M, D))
    for m, (mu, rm, nm) in enumerate(zip(mu, responsibility, membership)):
        diff = y - mu
        denominator = nm - 1 if nm > 1 else nm
        cov[m] = np.dot(rm, diff**2) / denominator + covariance_regularization

    return cov


def _estimate_covariance_matrix_full(y, responsibility, mu, 
    covariance_regularization=0):
    """
    Estimate the full (off-diagonal) covariance matrix for the given data and
    mixtures. Where relevant, the Neyman-Scott correction is applied.

    :param y:
        The data :math:`y`.

    :param responsibility:
        The responsibility matrix.

    :param mu:
        The centroids of the mixtures.

    :param covariance_regularization: [optional]
        The regularization to apply to the diagonal of the covariance matrices
        in order to maintain stability (default = 0).

    :returns:
        The estimated covariance matrices of the :math:`K` mixtures.
    """

    N, D = y.shape
    M, N = responsibility.shape

    membership = np.sum(responsibility, axis=1)

    I = np.eye(D)
    cov = np.empty((M, D, D))
    for m, (mu, rm, nm) in enumerate(zip(mu, responsibility, membership)):

        diff = y - mu
        denominator = nm - 1 if nm > 1 else nm

        cov[m] = np.dot(rm * diff.T, diff) / denominator \
               + covariance_regularization * I

    return cov


def estimate_covariance_matrix(y, responsibility, mu, covariance_type,
    covariance_regularization=0, **kwargs):
    """
    Estimate the covariance matrix for the given data and mixtures.

    :param y:
        The data :math:`y`.

    :param responsibility:
        The responsibility matrix.

    :param mu:
        The centroids of the mixtures.

    :param covariance_type:
        The type of covariance structure to employ. Available options include
        'diag' or 'full'.

    :param covariance_regularization: [optional]
        The regularization strength to apply to the diagonal of the covariance
        matrices in order to maintain stability (default = 0).

    :returns:
        The estimated covariance matrices of the :math:`K` mixtures.
    """

    available = {
        "full": _estimate_covariance_matrix_full,
        "diag": _estimate_covariance_matrix_diag
    }

    try:
        function = available[covariance_type]

    except KeyError:
        raise ValueError("unknown covariance type")

    return function(y, responsibility, mu, covariance_regularization)



def responsibility_matrix(y, mu, cov, weight, covariance_type, **kwargs):
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
        A two-length tuple containing the responsibility matrix and the log
        likelihood (per observation in each component of the mixture).
    """

    precision_cholesky = _compute_precision_cholesky(cov, covariance_type)
    weighted_log_prob = np.log(weight) + \
        _estimate_log_gaussian_prob(y, mu, precision_cholesky, covariance_type)

    log_likelihood = scipy.misc.logsumexp(weighted_log_prob, axis=1)
    with np.errstate(under="ignore"):
        log_responsibility = weighted_log_prob - log_likelihood[:, np.newaxis]
    
    responsibility = np.exp(log_responsibility).T
    
    return (responsibility, log_likelihood) 


def expectation(y, mu, cov, weight, **kwargs):
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

    :returns:
        A three-length tuple containing the responsibility matrix, the log 
        likelihood, and the calculated message length for this mixture, given
        the data.
    """

    responsibility, log_likelihood = responsibility_matrix(y, mu, cov, weight,
                                                           **kwargs)
    K = weight.size
    N, D = y.shape
    nll = -np.sum(log_likelihood) # the negative log likelihood

    I, I_parts = mixture_message_length(N, D, K, cov, weight, -nll, **kwargs)

    visualization_handler = kwargs.get("visualization_handler", None)
    if visualization_handler is not None:
        visualization_handler.emit("expectation", dict(
            K=weight.size, message_length=I, responsibility=responsibility,
            log_likelihood=log_likelihood))

    return (responsibility, log_likelihood, I)


def maximization(y, mu, cov, weight, responsibility, parent_responsibility=1,
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

    new_cov = _estimate_covariance_matrix(y, responsibility, new_mu, **kwargs)

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
