
import numpy as np
from gmmmml import (gmm as mixture, visualize_new as visualize, utils)
import  scipy.optimize as op

np.random.seed(11)

# Seed 1100 is a fun one: K = 103, D = 5, N = 4249

import logging

"""
for i in range(10):

    j = 0
    while True:
        print(i, j)
        j += 1
        try:
            y, labels, target, kwds = utils.generate_data(K=30, D=10, center_box=(-1000, 1000))

        except:
          logging.exception("failed")
          if j > 3:
            raise 
          continue 

        else:
          break
"""
from sklearn.datasets import make_blobs

y, assignments = make_blobs(
  n_samples=1000, n_features=2, centers=30, random_state=42)
#y = np.vstack([y, np.random.uniform(-10, 10, size=(10, y.shape[1]))])

#n_samples=100, n_features=2, centers=3, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=Non

visualization_handler = visualize.VisualizationHandler(
    y, target=None, figure_path="tmp/")

search_model = mixture.GaussianMixture(
  max_em_iterations=5, covariance_regularization=1e-10)
search_model.search_greedy_forgetful(y, K_max=50, 
    visualization_handler=visualization_handler)





#y = np.loadtxt("toy-data/cluster_example.txt")


#n_samples = 500

# Generate random sample, two components
#np.random.seed(0)
#C = np.array([[0., -0.1], [1.7, .4]])
#X = np.r_[np.dot(np.random.randn(n_samples, 2), C),
#          .7 * np.random.randn(n_samples, 2) + np.array([-6, 3])]
#y = X


#from sklearn import datasets
#iris = datasets.load_iris()
#y = iris.data

"""
Approximating \sum\log{w_k}...

bar = []
for i in range(1, 101):

    foo = np.random.uniform(size=(1000, i))
    foo = foo.T/np.sum(foo, axis=1)
    print(i)

    mean = np.mean(np.log(foo).sum(axis=0))
    std = np.std(np.log(foo).sum(axis=0))

    bar.append([i, mean, std])

    

bar = np.array(bar)

fig, ax = plt.subplots()
ax.scatter(bar.T[0], bar.T[2])
#ax.scatter(bar.T[0], bar.T[1])
#ax.errorbar(bar.T[0], bar.T[1], bar.T[2], fmt=None)

raise a
"""

"""
# Approximating \log(\sum{r}) (the log of the effective memberships...)
bar = []
for i in range(1, 101):

    foo = np.random.uniform(size=(1000, i))
    foo = foo.T/np.sum(foo, axis=1)
    foo *= 900 # The sample size, say.

    mean = np.mean(np.log(foo).sum(axis=0))
    std = np.std(np.log(foo).sum(axis=0))

    bar.append([i, mean, std])


bar = np.array(bar)

fig, axes = plt.subplots(2)
axes[0].scatter(bar.T[0], bar.T[1])
axes[1].scatter(bar.T[0], bar.T[2])

raise a
"""

search_model = mixture.GaussianMixture()
search_model.kmeans_search(y)



model1 = mixture.GaussianMixture()
mu_1, cov_1, weight_1, meta_1 = model1.fit(y, 1)


model2 = mixture.GaussianMixture()
mu_2, cov_2, weight_2, meta_2 = model2.fit(y, 2)



N, D = y.shape
K = 1

Q_K = (0.5 * D * (D + 3) * K) + (K - 1)
Q_K2 = (0.5 * D * (D + 3) * (K + 1)) + (K + 1 - 1)

# Calculate message lengths according to our simplified expression.
import scipy
exp_I1 = (1 - D/2.0) * np.log(2) + 0.5 * Q_K * np.log(N) + 0.5 * np.log(Q_K * np.pi) \
    - 0.5 * np.sum(np.log(weight_1)) - scipy.special.gammaln(K) - N * D * np.log(0.001) \
    - meta_1["log_likelihood"].sum() + (D*(D+3))/4.0 * np.sum(np.log(weight_1)) \
    - (D + 2)/2.0 * np.sum(np.log(np.linalg.det(cov_1)))


# Calculate the deltas in message length, according to  our expression.
actual_delta_I = meta_2["message_length"] - meta_1["message_length"]
expected_delta_I = np.log(2) \
    + np.log(N)/2.0 - np.log(K) - 0.5 * (np.sum(np.log(weight_2)) - np.sum(np.log(weight_1))) \
    - D * np.log(2)/2.0 + D * (D+3)/4.0 * (np.log(N) + np.sum(np.log(weight_2)) - np.sum(np.log(weight_1))) - (D + 2)/2.0 * (np.sum(np.log(np.linalg.det(cov_2))) - np.sum(np.log(np.linalg.det(cov_1)))) \
    + 0.25 * (2 * np.log(Q_K2/Q_K) - (D * (D+3) + 2) * np.log(2*np.pi)) \
    + meta_2["log_likelihood"] - meta_1["log_likelihood"]
expected_delta_I = expected_delta_I/np.log(2)

dk = 1
expected_delta_I2 = dk * ((1 - D/2.) * np.log(2) + 0.25 * (D*(D+3) + 2) * np.log(N/(2*np.pi))) \
                  + 0.5 * (D*(D+3)/2. - 1) * (np.sum(np.log(weight_2)) - np.sum(np.log(weight_1))) \
                  - np.sum([np.log(K + _) for _ in range(dk)]) \
                  - meta_2["log_likelihood"].sum() + meta_1["log_likelihood"].sum() \
                  + 0.5 * np.log(Q_K2/float(Q_K)) \
                  + (D + 2)/2.0 * (np.sum(np.log(np.linalg.det(cov_1))) - np.sum(np.log(np.linalg.det(cov_2))))
expected_delta_I2 = expected_delta_I2/np.log(2)



# OK,. let's see if we can estimate the learning rate \gamma
def _evaluate_gaussian(y, mu, cov):
   N, D = y.shape
   Cinv = np.linalg.inv(cov)
   scale = 1.0/np.sqrt((2*np.pi)**D * np.linalg.det(cov))#
   #Cinv**(-0.5)
   d = y - mu
   return scale * np.exp(-0.5 * np.sum(d.T * np.dot(Cinv, d.T), axis=0))

model = mixture.GaussianMixture()
mu1, cov1, weight1, meta1 = model.fit(y, 1)

x = []
yvals = []
evaluated = []
prediction = []
for k in range(1, 10):
    model = mixture.GaussianMixture()
    mu, cov, weight, meta = model.fit(y, k)
    yvals.append(meta["log_likelihood"].sum())
    evaluated.append(np.sum(weight * np.vstack([_evaluate_gaussian(y, mu[i], cov[i]) for i in range(k)]).T))
    x.append(k)

    if k < 2:
        prediction.append(np.nan)
    else:
        func = mixture._approximate_log_likelihood_improvement(y, mu1, cov1,
            weight1, meta1["log_likelihood"].sum(), *yvals[1:])
        prediction.append(func(k + 1))


x = np.array(x)
yvals = np.array(yvals)
#ax.scatter(x, yvals)
foo = np.diff(yvals) / np.array(evaluated)[:-1]


cost_function = lambda x, *p: p[0] / np.exp(x) #+ p[1]

import scipy.optimize as op

p_opt, p_cov = op.curve_fit(cost_function, x[:-1][:2], foo[:2], p0=np.ones(1))

fig, ax = plt.subplots()
ax.scatter(x[:-1], foo)

ax.plot(x[:-1], cost_function(x[:-1], *p_opt))

model = mixture.GaussianMixture()
mu, cov, weight, meta = model.fit(y, 1)

model2 = mixture.GaussianMixture()
mu2, cov2, weight2, meta2 = model.fit(y, 2)

model3 = mixture.GaussianMixture()
mu3, cov3, weight3, meta3 = model.fit(y, 3)


func = mixture._approximate_log_likelihood_improvement(y, mu, cov, weight,
    meta["log_likelihood"].sum(), *[meta2["log_likelihood"].sum()])

fig, ax = plt.subplots()
ax.scatter(x, yvals)
ax.scatter(x, prediction)
ax.plot(x, [func(xi + 1) for xi in x], c='r')
#ax.plot(x[:-1][1:], [func(xi) for xi in x[:-1][1:]])

raise a

# OK,. let's see if we can estimate the learning rate \gamma
def _evaluate_gaussian(y, mu, cov):
   N, D = y.shape
   Cinv = np.linalg.inv(cov)
   scale = 1.0/np.sqrt((2*np.pi)**D * np.linalg.det(cov))#
   #Cinv**(-0.5)
   d = y - mu
   return scale * np.exp(-0.5 * np.sum(d.T * np.dot(Cinv, d.T), axis=0))

other = np.log(2) \
    + np.log(N)/2.0 - np.log(K) - 0.5 * (np.sum(np.log(weight_2)) - np.sum(np.log(weight_1))) \
    - D * np.log(2)/2.0 + D * (D+3)/4.0 * (np.log(N) + np.sum(np.log(weight_2)) - np.sum(np.log(weight_1))) \
    - (D + 2)/2.0 * (np.sum(np.log(np.linalg.det(cov_2))) - np.sum(np.log(np.linalg.det(cov_1)))) \
    + 0.25 * (2 * np.log(Q_K2/Q_K) - (D * (D+3) + 2) * np.log(2*np.pi))

gamma = K * _evaluate_gaussian(y, mu_1[0], cov_1[0]).sum() * (actual_delta_I - other)

# OK, now use gamma to estimate K = 3

K = 2
Q_K3 = (0.5 * D * (D + 3) * K) + (K - 1)

# Let us assume the determinants of covariance matrices will decrease:
cov_3_est = K / (K + 1) * np.linalg.det(cov_2)
cov_3_est = np.hstack([cov_3_est.min(), cov_3_est])

est_weight_3 = np.array([1/3., 1/3., 1/3.])


I_K3_to_K2 = np.log(2) \
    + np.log(N)/2.0 - np.log(K) - 0.5 * (np.sum(np.log(est_weight_3)) - np.sum(np.log(weight_2))) \
    - D * np.log(2)/2.0 + D * (D+3)/4.0 * (np.log(N) + np.sum(np.log(est_weight_3)) - np.sum(np.log(weight_2))) \
    - (D + 2)/2.0 * (np.sum(np.log(cov_3_est)) - np.sum(np.log(np.linalg.det(cov_2)))) \
    + 0.25 * (2 * np.log(Q_K3/Q_K2) - (D * (D+3) + 2) * np.log(2*np.pi)) \
    + gamma/(K+1) * np.sum(weight_2 * np.vstack([_evaluate_gaussian(y, mu_2[i], cov_2[i]) for i in range(2)]).T)


raise a

delta_I = np.log(2) + 0.5 * np.log(N) - np.log(K) \
        + 0.5 * (D*(D+3)/2.0 * np.log(N) - D * np.log(2)) \
        + 0.5 * (np.sum(np.log(np.linalg.det(cov_2))) - np.sum(np.log(np.linalg.det(cov_1)))) \
        + 0.5 * (np.log(Q_K2/Q_K) - np.log(2*np.pi)/2.0 * (D * (D + 3) + 2))

raise a