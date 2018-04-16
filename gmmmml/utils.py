
import numpy as np
from sklearn import datasets


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
    I, I_parts = _mixture_message_length(K, N, D, -nll, np.sum(slogdetcovs),
        weights=[weight])

    target = dict(mean=mean, cov=cov, weight=weight, I=I, I_parts=I_parts,
        nll=nll)

    return (X, y, target, kwds)