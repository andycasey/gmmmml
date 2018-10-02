
import numpy as np
from scipy.special import beta

from sklearn import datasets
from sklearn.neighbors import NearestNeighbors



def kullback_leibler_for_multivariate_normals(mu_a, cov_a, mu_b, cov_b):
    r"""
    Return the Kullback-Leibler distance from one multivariate normal
    distribution with mean :math:`\mu_a` and covariance :math:`\Sigma_a`,
    to another multivariate normal distribution with mean :math:`\mu_b` and 
    covariance matrix :math:`\Sigma_b`. The two distributions are assumed to 
    have the same number of dimensions, such that the Kullback-Leibler 
    distance is

    .. math::
        D_{\mathrm{KL}}\left(\mathcal{N}_{a}||\mathcal{N}_{b}\right) = 
            \frac{1}{2}\left(\mathrm{Tr}\left(\Sigma_{b}^{-1}\Sigma_{a}\right) + \left(\mu_{b}-\mu_{a}\right)^\top\Sigma_{b}^{-1}\left(\mu_{b} - \mu_{a}\right) - k + \ln{\left(\frac{\det{\Sigma_{b}}}{\det{\Sigma_{a}}}\right)}\right)

    where :math:`k` is the number of dimensions and the resulting distance is 
    given in units of nats.

    .. warning::
        It is important to remember that 
        :math:`D_{\mathrm{KL}}\left(\mathcal{N}_{a}||\mathcal{N}_{b}\right) \neq D_{\mathrm{KL}}\left(\mathcal{N}_{b}||\mathcal{N}_{a}\right)`.

    :param mu_a:
        The mean of the first multivariate normal distribution.

    :param cov_a:
        The covariance matrix of the first multivariate normal distribution.

    :param mu_b:
        The mean of the second multivariate normal distribution.
        
    :param cov_b:
        The covariance matrix of the second multivariate normal distribution.
    
    :returns:
        The Kullback-Leibler distance from distribution :math:`a` to :math:`b`
        in units of nats. Dividing the result by :math:`\log_{e}2` will give
        the distance in units of bits.
    """

    if len(cov_a.shape) == 1:
        cov_a = cov_a * np.eye(cov_a.size)

    if len(cov_b.shape) == 1:
        cov_b = cov_b * np.eye(cov_b.size)

    U, S, V = np.linalg.svd(cov_a)
    Ca_inv = np.dot(np.dot(V.T, np.linalg.inv(np.diag(S))), U.T)

    U, S, V = np.linalg.svd(cov_b)
    Cb_inv = np.dot(np.dot(V.T, np.linalg.inv(np.diag(S))), U.T)

    k = mu_a.size

    offset = mu_b - mu_a
    return 0.5 * np.sum([
          np.trace(np.dot(Ca_inv, cov_b)),
        + np.dot(offset.T, np.dot(Cb_inv, offset)),
        - k,
        + np.log(np.linalg.det(cov_b)/np.linalg.det(cov_a))
    ])



def _best_mixture_parameter_values(K, I, value):
    """
    Return the mixture parameter value that has the lowest cost for each K.
    """

    K = np.atleast_1d(K)
    I = np.atleast_1d(I)

    unique_K = np.unique(K)
    best_value = np.empty(unique_K.size)

    for i, k in enumerate(unique_K):

        idxs = np.where((K == k))[0]
        min_idx = np.argmin(I[idxs])

        best_value[i] = value[idxs[min_idx]]

    return (unique_K, best_value)



def _group_over(x, y, function):

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    x_unique = np.sort(np.unique(x))
    y_unique = np.nan * np.ones_like(x_unique)

    for i, xi in enumerate(x_unique):
        match = (x == xi)
        y_unique[i] = function(y[match])

    return (x_unique, y_unique)


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




def generate_data(N, D, K=None, dirichlet_concentration=1, isotropy=10, psi=0.05,
                  random_seed=None, **kwargs):
    r"""
    Generate data from a mixture of multivariate Gaussian components with full
    rank covariance matrices with non-zero off-diagonal terms.

    :param N:
        The number of samples to draw.

    :param D:
        The dimensionality of the data.

    :param K: [optional]
        The number of components to model. If ``None`` is given then a random
        number will be drawn from a log-uniform (base 10) distribution between
        :math:`K = 1` and :math:`K = \floor{N/2}`.

    :param dirichlet_concentration: [optional]
        A metric representing the uniformity of the relative weights of the
        mixture. This is the concentration parameter for a symmetric Dirichlet
        distribution to draw the weights of the components. The concentration
        parameter must be greater than zero. A concentration parameter much
        less than one indicates most of the data will be drawn from one
        component. A concentration parameter much greater than one indicates
        that the support will be distributed evenly between the components.

    #  TODO
    """

    np.random.seed(random_seed)

    N, D = (int(N), int(D))

    if K is None:
        K = int(np.round(10**np.random.uniform(0, np.log10(N/2.0))))

    if K >= N/2.0:
        raise ValueError(f"K must be less than N/2 ({2*K} > {N})")

    if 0 >= dirichlet_concentration:
        raise ValueError("dirichlet_concentration must be positive")

    # Generate the weights.
    min_weight = 2.0/N
    max_weight = 1 - min_weight * (K - 1)

    alphas = np.ones(K) * dirichlet_concentration
    for iteration in range(kwargs.pop("dirichlet_draws", 100000)):
        weights = np.random.dirichlet(alphas).flatten()

        # Check the weights.
        if np.min(weights) >= min_weight and np.max(weights) <= max_weight:
            break

    else:
        raise ValueError(f"No suitable sampling of mixture weights found! "\
                          "Try adjusting dirichlet_distribution or increasing "\
                          "dirichlet_draws.")
    
    # Generate means.
    #means = np.random.normal(0, 1, size=(K, D))
    means = np.random.uniform(-1, 1, size=(K, D))

    # Generate correlation coefficients.
    rho = lambda N=1: ((1 - np.random.uniform(-1, 1, N)**2)**(isotropy - 4)/2.) \
                    / beta(0.5, (isotropy - 2)/2)

    # Generate covariance matrices.
    phi = np.abs(np.random.normal(0, psi, size=(K, D)))

    I = np.eye(D)
    covariances = np.empty((K, D, D))
    for k, diag in enumerate(phi):

        covariances[k] = I * diag

        for i, j in zip(*np.tril_indices(D, -1)):
            value = rho(1) * diag[i] * diag[j]
            covariances[k, i, j] = covariances[k, j, i] = value

    # Draw samples from each component to generate the data.
    members = np.round(N * weights).astype(int)

    X = np.empty((N, D))
    R = np.zeros(N, dtype=int)

    for i, (m, mean, cov) in enumerate(zip(members, means, covariances)):
        si = int(np.sum(members[:i]))
        X[si:si + m] = np.random.multivariate_normal(mean, cov, size=m)[:N-si]
        R[si:si + m] = i

    # Shuffle the order.
    idx = np.random.choice(np.arange(N), N, replace=False)
    X, R = (X[idx], R[idx])

    meta = dict(K=K, N=N, D=D, dirichlet_concentration=dirichlet_concentration, 
                isotropy=isotropy, psi=psi, random_seed=random_seed,
                responsibilities=R, truths=dict(
                    means=means,
                    weights=weights,
                    covariances=covariances
                ))

    return (X, meta)





def generate_isotropic_data(N=None, D=None, K=None, cluster_std=1.0, 
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

    from .em import responsibilities, _estimate_covariance_matrix_full
    from .mml import gaussian_mixture_message_length

    cov = _estimate_covariance_matrix_full(X, mean, responsibility)
    
    weight = responsibility.sum(axis=1)/N

    # TODO: REFACTOR


    
    responsibility, log_likelihood, lp = responsibilities(
        X, mean, cov, weight, full_output=True, covariance_type="full")

    nll = -np.sum(log_likelihood)
    #I, I_parts = _message_length(X, mean, cov, weight, responsibility, nll,
    #    full_output=True, covariance_type="full")

    _, slogdetcovs = np.linalg.slogdet(cov)
    I_parts = gaussian_mixture_message_length(K, N, D, -nll, np.sum(slogdetcovs),
        [weight])

    I = np.sum(I_parts.values())

    target = dict(mean=mean, cov=cov, weight=weight, I=I, I_parts=I_parts,
        nll=nll)

    return (X, y, target, kwds)