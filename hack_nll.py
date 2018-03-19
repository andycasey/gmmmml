
import numpy as np
from gmmmml import (mixture_search as mixture, visualize, utils)
import  scipy.optimize as op
import scipy.misc
np.random.seed(11)

# Seed 1100 is a fun one: K = 103, D = 5, N = 4249

def _approximate_log_gaussian_prob(X, weights, precision_cholesky, covariance_type="full"):
    n_samples, n_features = X.shape
    #n_components, _ = means.shape
    n_components = weights.size
    # det(precision_chol) is half of det(precision)
    log_det = mixture._compute_log_det_cholesky(
        precision_cholesky, covariance_type, n_features)

    
    D = 10.0
    scale = np.ptp(X, axis=0)/2.0

    log_prob = np.ones((n_samples, n_components)) * D * np.mean(scale**2)

    N = X.shape[0]
    Sk = 0
    for k, weight in enumerate(weights):
        Nk = int(np.round(weight * N))
        log_prob[Sk:Sk + Nk, k] = D
        Sk += Nk

    #for k, (mu, prec_chol) in enumerate(zip(means, precision_cholesky)):
    #    y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
    #    log_prob[:, k] = np.sum(np.square(y), axis=1)

    moo =  -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det
    raise a


def _approximate_log_likelihood(X, weights, precision_cholesky, covariance_type="full"):
    n_samples, n_features = X.shape
    #n_components, _ = means.shape
    n_components = weights.size
    # det(precision_chol) is half of det(precision)
    log_det = mixture._compute_log_det_cholesky(
        precision_cholesky, covariance_type, n_features)

    
    D = 10.0
    scale = np.ptp(X, axis=0)/2.0

    log_prob = np.ones((n_samples, n_components)) * D * np.mean(scale**2)

    N = X.shape[0]
    Sk = 0
    for k, weight in enumerate(weights):
        Nk = int(np.round(weight * N))
        log_prob[Sk:Sk + Nk, k] = D
        Sk += Nk

    #for k, (mu, prec_chol) in enumerate(zip(means, precision_cholesky)):
    #    y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
    #    log_prob[:, k] = np.sum(np.square(y), axis=1)

    moo =  -0.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det 
    moo = log_det + np.log(weights) - 0.5 * n_features * np.log(2 * np.pi) \
        - 0.5 * log_prob

    scalar = np.exp(-0.5 * D * np.log(2*np.pi))

    det = np.exp(log_det)
    moo2 = scalar * det * weights * np.exp(-0.5 * log_prob)

    # when a log_prob entry indicates it is correct, it is 1
    # if not, it is some huge value

    # np.exp(-0.5) = 0.6065
    # np.exp(-10000) = 0

    # therefore,...

    prob = np.zeros_like(log_prob)
    Sk = 0
    for k, weight in enumerate(weights):
        Nk = int(np.round(weight * N))
        prob[Sk:Sk + Nk, k] = np.exp(-0.5 * D)
        Sk += Nk

    A = det * weights
    B = scalar * prob
    
    moo3 = np.dot(A, B.T)

    # correct: LL = np.sum(np.log(moo3))

    moo4 = det * weights * scalar * np.exp(-0.5 * D) #* (N * weights)

    # correct: LL = np.sum(N * weights * np.log(moo4))

    moo5 = N * np.sum(weights * np.log(det * weights)) - 0.5 * N * D * (np.log(2*np.pi) + 1)

    # correct: LL = moo5

    moo6 = N * np.sum(weights * np.log(weights)) \
         + N * np.sum(weights * log_det) \
         - 0.5 * N * D * (np.log(2 * np.pi) + 1)


    # Compute the log of the sum of expoentials of input elements.



    raise a






for i in range(10):

    j = 0
    while True:
        print(i, j)
        j += 1
        try:
            
            y, labels, target, kwds = utils.generate_data(K=30, D=10, center_box=(-1000, 1000))

        except:
            continue

        else:
          break

    
    weight = target["weight"]
    cov = target["cov"]
    mu = target["mean"]

    covariance_type = "full"
    precision_cholesky = mixture._compute_precision_cholesky(cov, covariance_type)
    bar = mixture._estimate_log_gaussian_prob(y, mu, precision_cholesky, covariance_type)
    weighted_log_prob = np.log(weight) + bar

    log_likelihood = scipy.misc.logsumexp(weighted_log_prob, axis=1)

    foo = _approximate_log_likelihood(y, weight, precision_cholesky)

    raise a

    foo = _approximate_log_gaussian_prob(y, weight, precision_cholesky)

    raise a
    with np.errstate(under="ignore"):
        log_responsibility = weighted_log_prob - log_likelihood[:, np.newaxis]

    
    responsibility = np.exp(log_responsibility).T
    
    if kwargs.get("dofail", False):
        raise a