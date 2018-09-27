
"""
Policies for prediction and search.
"""

import logging
import numpy as np
import scipy
from sklearn import cluster
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms


from .base import Policy
from ..em import _estimate_covariance_matrix_full

logger_name, *_ = __name__.split(".")
logger = logging.getLogger(logger_name)


class BaseInitialisationPolicy(Policy):

    def __init__(self, model, K_init=1, **kwargs):

        self.model = model
        self.K_init = K_init
        return None


    def initialise(self, y):
        raise NotImplementedError("should be implemented by sub-classes")



class DefaultInitialisationPolicy(BaseInitialisationPolicy):

    def initialise(self, y, **kwargs):

        y = np.atleast_2d(y)
        N, D = y.shape

        K_inits = np.logspace(0, np.log10(N), self.K_init, dtype=int)

        for i, K in enumerate(K_inits):

            try:
                *state, R = kmeans_pp(y, K, **kwargs)
                yield (K, self.model.expectation_maximization(y, *state, **kwargs))

            except ValueError:
                logger.warn(f"Failed to initialise at K = {K}")
                break


        return None






def kmeans_pp(y, K, random_state=None, **kwargs):
    r"""
    Initialize a Gaussian mixture model using the K-means++ algorithm.

    :param y:
        The data values, :math:`y`, which are expected to have :math:`N` 
        samples each with :math:`D` dimensions. Expected shape of :math:`y` 
        is :math:`(N, D)`.

    :param K:
        The number of Gaussian components in the mixture.
    
    :param random_state: [optional]
        The state to provide to the random number generator.

    :returns:
        A four-length tuple containing:

        (1) an estimate of the means of the components

        (2) an estimate of the covariance matrices of the components

        (3) an estimate of the relative mixings of the components

        (4) the responsibility matrix for each data point to each component.
    """

    random_state = check_random_state(random_state)
    squared_norms = row_norms(y, squared=True)
    means = cluster.k_means_._k_init(y, K,
                                     random_state=random_state,
                                     x_squared_norms=squared_norms)

    labels = np.argmin(scipy.spatial.distance.cdist(means, y), axis=0)

    N, D = y.shape
    responsibilities = np.zeros((K, N))
    responsibilities[labels, np.arange(N)] = 1.0

    covs = _estimate_covariance_matrix_full(y, means, responsibilities, 
        covariance_regularization=kwargs.get("covariance_regularization", 0))

    weights = responsibilities.sum(axis=1)/N

    return (means, covs, weights, responsibilities)

