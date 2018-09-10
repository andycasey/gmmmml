
import numpy as np
from sklearn import datasets

from sklearn.neighbors import NearestNeighbors

def concentration(y, K_max=None):
    """
    Calculate the maximum concentration as a function of the number of data
    points. Specifically, calculate the minimum of

    .. math:

        w_{k}\log{|C_k|}

    for any subset of the N data points that exist. That is, find the :math:`M` 
    nearest points in order to calculate the minimum sample covariance among 
    :math:`M` points, and do this for increasing :math:`M` values. Then for
    each number of :math:`K` components, calculate the minimum mixture among a
    set o
    """

    y = np.atleast_2d(y)
    N, D = y.shape

    if K_max is None or K_max > N:
        K_max = N

    knn = NearestNeighbors().fit(y)
    distances, indices = knn.kneighbors(y, n_neighbors=2)
    
    # Successively build up the minimum concentration.
    K = np.arange(1, 1 + K_max, dtype=int)
    minimum_concentration = np.zeros(K.size, dtype=float)
    maximum_concentration = np.zeros(K.size, dtype=float)
    minimum_concentration[0] = maximum_concentration[0] \
                             = np.sum(np.log(np.var(y, axis=0)))
    
    minimum_sum_log_dets = np.zeros(K.size)
    minimum_sum_log_dets[0] = minimum_concentration[0]

    for i, k in enumerate(K[1:]):

        # Find the closest and smallest k-1 mixtures
        closest_indices = np.argsort(distances.T[1])

        assigned = np.zeros(N, dtype=bool)
        components, concentration = (0, 0)
        min_sum_log_det = 0

        for index in closest_indices:
            if np.any(assigned[indices[index, :2]]): continue
            concentration += 2.0/N * np.sum(np.log(np.var(y[indices[index, :2]],
                                                          axis=0)))
            min_sum_log_det += np.sum(np.log(np.var(y[indices[index, :2]], axis=0)))

            assigned[indices[index, :2]] = True
            components += 1

            if components == k - 1:
                break

        sum_log_det = np.sum(np.log(np.var(y[~assigned], axis=0)))
        weight = (N - (k-1) * 2)/float(N)
        minimum_concentration[i] = concentration + weight * sum_log_det
        minimum_sum_log_dets[i] = min_sum_log_det + np.sum(np.log(np.var(y[~assigned], axis=0)))
        print(k, minimum_concentration[i])

    maximum_mixture_concentration = K * np.log(np.product(np.var(y, axis=0))/K)

    return (K, minimum_concentration, maximum_mixture_concentration, minimum_sum_log_dets)


def concentration_hard(y, K_max=None):
    """
    Calculate the maximum concentration as a function of the number of data
    points. Specifically, calculate the minimum of

    .. math:

        w_{k}\log{|C_k|}

    for any subset of the N data points that exist. That is, find the :math:`M` 
    nearest points in order to calculate the minimum sample covariance among 
    :math:`M` points, and do this for increasing :math:`M` values. Then for
    each number of :math:`K` components, calculate the minimum mixture among a
    set o
    """

    y = np.atleast_2d(y)
    N, D = y.shape

    if K_max is None or K_max > N:
        K_max = N

    knn = NearestNeighbors().fit(y)
    distances, indices = knn.kneighbors(y, n_neighbors=int(N/2))
    
    estimated_cumulative_log_det = np.cumsum(
        D * np.log((distances[:, 1:]/(2 * D))**2), axis=1)

    estimated_cumulative_concentration = estimated_cumulative_log_det \
                                       * np.arange(2, 1 + distances.shape[1]) \
                                       / N

    # Successively build up the minimum concentration.
    K = np.arange(2, 1 + K_max, dtype=int)
    minimum_mixture_concentration = np.zeros(K.size, dtype=float)
    for i, k in enumerate(K):

        assigned = np.zeros(N, dtype=bool)

        ecc = np.copy(estimated_cumulative_concentration)

        min_index = np.argmin(estimated_cumulative_concentration)
        xi, yi = int(min_index / ecc.shape[1]), min_index % ecc.shape[1]

        raise a
        #for K = 2 we are looking for 


        # Find the closest and smallest k-1 mixtures
        closest_indices = np.argsort(distances.T[1])

        components, concentration = (0, 0)

        for index in closest_indices:
            if np.any(assigned[indices[index, :2]]): continue
            concentration += 2.0/N * np.sum(np.log(np.var(y[indices[index, :2]],
                                                          axis=0)))

            assigned[indices[index, :2]] = True
            components += 1

            if components == k - 1:
                break

        sum_log_det = np.sum(np.log(np.var(y[~assigned], axis=0)))
        weight = (N - (k-1) * 2)/float(N)
        minimum_mixture_concentration[i] = concentration + weight * sum_log_det

        print(k, minimum_mixture_concentration[i])

    return (K, minimum_mixture_concentration)

def aggregate(x, y, function):
    """
    Group over arrays of :math:`x` and :math:`y` and apply an aggregate
    function to all values at each unique :math:`x` entry.

    :param x:
        An array of x values.

    :param y:
        An array of y values, corresponding to the same length as `x`.

    :param function:
        An aggregate function to apply to all `y` values at each unique `x`
        entry.

    :returns:
        An array of unique `x` values and the aggreggated `y` values for each
        unique `x` value.
    """

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    x_unique = np.sort(np.unique(x))
    y_aggregated = np.nan * np.ones_like(x_unique)

    for i, xi in enumerate(x_unique):
        match = (x == xi)
        y_aggregated[i] = function(y[match])

    return (x_unique, y_aggregated)




# TODO: REPEATED CODE MOVE THIS
def _estimate_covariance_matrix_full(y, responsibility, mean, 
    covariance_regularization=0):

    N, D = y.shape
    M, N = responsibility.shape

    membership = np.sum(responsibility, axis=1)

    I = np.eye(D)
    cov = np.empty((M, D, D))
    for m, (mu, rm, nm) in enumerate(zip(mean, responsibility, membership)):

        diff = y - mu
        denominator = nm - 1 if nm > 1 else nm

        cov[m] = np.dot(rm * diff.T, diff) / denominator \
               + covariance_regularization * I

    return cov



def generate_data(N=None, D=None, K=None, cluster_std=1.0, 
    center_box=(-10, 10.0), shuffle=True, random_state=None):

    if K is None:
        K = max(1, abs(int(np.random.normal(0, 100))))

    if N is None:
        N = int(np.random.uniform(K, K**2))

    if D is None:
        D = int(np.random.uniform(1, 10))

    kwds = dict(n_samples=N, n_features=D, centers=K,
        cluster_std=cluster_std, center_box=center_box, shuffle=shuffle,
        random_state=random_state)
    X, y = datasets.make_blobs(**kwds)

    # Estimate true values from the blobs:
    # mu, covs, weights

    responsibility = np.zeros((K, N))
    responsibility[y, np.arange(N)] = 1.0
    membership = responsibility.sum(axis=1)

    mean = np.zeros((K, D))
    for k in range(K):
        mean[k] = np.sum(responsibility[k] * X.T, axis=1) / membership[k]

    cov = _estimate_covariance_matrix_full(X, responsibility, mean)
    
    weight = responsibility.sum(axis=1)/N

    # TODO: REFACTOR

    from .mixture_search import responsibility_matrix, _mixture_message_length

    
    responsibility, log_likelihood = responsibility_matrix(
        X, mean, cov, weight, full_output=True, covariance_type="full")

    nll = -np.sum(log_likelihood)
    #I, I_parts = _message_length(X, mean, cov, weight, responsibility, nll,
    #    full_output=True, covariance_type="full")

    _, slogdetcovs = np.linalg.slogdet(cov)
    I_parts = _mixture_message_length(K, N, D, -nll, np.sum(slogdetcovs),
        weights=[weight])

    I = np.sum(I_parts.values())

    target = dict(mean=mean, cov=cov, weight=weight, I=I, I_parts=I_parts,
        nll=nll)

    return (X, y, target, kwds)