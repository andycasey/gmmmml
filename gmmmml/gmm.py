
"""
Model data with a mixture of gaussians.
"""

import logging
import numpy as np
import scipy
from sklearn import cluster
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms


logger = logging.getLogger(__name__)


class GaussianMixture(object):

    r"""
    Model data from many multivariate Gaussian distributions, using minimum 
    message length (MML) as the objective function.

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


    def search_greedy_forgetful(self, y, K_max=None, random_state=None,
        **kwargs):
        r"""
        Fit the data using a greedy search algorithm, where we do not retain
        information about previous mixtures in order to initialise the next
        set of mixtures. Instead, each new trial of :math:`K` is initialised
        using the K-means++ algorithm.


        """

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

            # Initialise using k-means++.
            mu, cov, weight, responsibility = _initialise_by_kmeans_pp(y, K,
                random_state=random_state)

            # TODO: Will giving the same random state yield the same result
            #       on every iteration?


            raise a





        raise a



    def _record_state(self, covs, weights, log_likelihood):

        self._state_K.append(weights.size)

        # Record determinant of covariance matrices.
        self._state_det_covs.append(np.linalg.det(covs))

        # Record weights.
        self._state_weights.append(weights)

        # Record log-likelihood.
        self._state_slog_likelihoods.append(np.sum(log_likelihood))

        return None





def _initialise_by_kmeans_pp(y, K, random_state=None):
    """
    Initialise by k-means++ and assign hard responsibilities to the closest
    centroid.

    :param y:
        The data :math:`y`.

    :param K:
    `   The number of Gaussian mixtures to initialise with.
    
    :param random_state: [optional]
        The state to use for the random number generator.

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

    mu = cluster.k_means_._k_init(y, K, x_squared_norms=squared_nroms,
        random_state=random_state)

    # Assign everything to the closest mixture.
    labels = np.argmin(scipy.spatial.distance.cdist(mu, y), axis=0)

    # Generate responsibility matrix.
    responsibility = np.zeros((K, N))
    responsibility[labels, np.arange(N)] = 1.0

    # Calculate weights.
    weight = np.sum(responsibility, axis=1)/N

    # Estimate covariance matrices.
    cov = _estimate_covariance_matrix_full(y, responsibility, mu)

    return (mu, cov, weight, responsibility)



