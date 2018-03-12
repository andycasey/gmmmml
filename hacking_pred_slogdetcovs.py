
import numpy as np
from gmmmml import (mixture_search as mixture, visualize, utils)
import  scipy.optimize as op

np.random.seed(42)

import pickle

with open("foo.pkl", "rb")  as fp:
    x, y, logcovs =  pickle.load(fp)



covs = np.array([np.exp(ea) for ea  in logcovs])

xu = x
yu = np.array([np.median(lc) for lc in logcovs])
slogdets = np.array([np.sum(each) for each in logcovs])

p0 = [slogdets[0]/x[0], 0.5, 0]

np.random.seed(42)

for N in range(3, 20):
    indices = np.random.choice(np.arange(len(xu)), N, replace=False)
    x_fit = xu[indices]
    y_fit = slogdets[indices]/x_fit


    def f(x, *params):
        a, b, c = params
        #print(params)
        return a/(x - b) + c #c*x + d

    bounds = np.asarray([
        (y_fit.min(), 100 * y_fit.max()),
        (-np.inf, np.inf),
        (-np.inf, 0)
    ], dtype=float).T

    bounds = (-np.inf, np.inf)

    def objective_function(p):
        return np.sum((y_fit - f(x_fit, *p))**2)
    print(N, p0, objective_function(p0))

    p_opt, p_cov = op.curve_fit(f, x_fit, y_fit, p0=p0, maxfev=1000000,
      bounds=bounds, sigma=x_fit.astype(float)**-1)


    print(N, p_opt, objective_function(p_opt))



    fig, axes = plt.subplots(2)
    ax = axes[0]
    ax.scatter(xu, yu)
    ax.scatter(x_fit, y_fit, c='r')
    ax.set_title("N = {}".format(N))

    xi = np.linspace(xu[0], xu[-1], 1000)
    ax.plot(xi, f(xi, *p_opt), c='r')

    ax = axes[1]
    ax.plot(xu, yu - f(xu, *p_opt))


    # OK try use this to predict what we actually want.
    fig, axes =  plt.subplots(2)
    ax = axes[0]
    ax.scatter(xu, slogdets)
    ax.scatter(x_fit, y_fit * x_fit, c='r')
    ax.plot(xu, f(xu, *p_opt) * xu, c='r')
    ax.set_title("N = {}".format(N))

    ax = axes[1]
    ax.scatter(xu, slogdets  - f(xu, *p_opt) * xu)


    p0 = list()
    p0.extend(p_opt)

raise a
# Fit a spline to the residuals

# just fit a spline.
import scipy.interpolate

knots = np.linspace(x_fit[0], x_fit[-1], 10)[1:-1]
tck = scipy.interpolate.splrep(x_fit, y_fit - f(x_fit, *p_opt), t=knots, task=-1,
  k=2)


ax.plot(x, scipy.interpolate.splev(x, tck), c='r')

raise a

fig,  ax = plt.subplots()
ax.scatter(x,  slogdets)
ax.plot(x, scipy.interpolate.splev(x, tck), c='r')






raise a

raise a

#yerr = np.array([np.std(lc) for lc in logcovs])

"""
fig, ax = plt.subplots()
ax.scatter(x, x*yu)
ax.fill_between(x, x*yu + x*yerr, x*yu - x*yerr, alpha=0.5)

"""



def slogdetcov(x, *params):
    a, b, c = params
    return x * (a * np.exp(b * x - c))

target_K = np.arange(1, 111)

p0 = [1, 0, 0]
size = 30

p_opt, p_cov = op.curve_fit(slogdetcov, xu, slogdets, p0=p0, maxfev=10000)

p16, p84 = np.nanpercentile(np.array(
  [slogdetcov(target_K, *p_draw) for p_draw in np.random.multivariate_normal(p_opt, p_cov, size=size)]),
  [16, 84], axis=0)


import matplotlib.pyplot as plt
fig, axes = plt.subplots(2)
ax = axes[0]


ax.scatter(x, slogdets)
ax.scatter([100], [-129])

ax.plot(target_K, slogdetcov(target_K, *p_opt), c='b')
#ax.fill_between(target_K, p16, p84, facecolor="b", alpha=0.5)

ax = axes[1]
ax.scatter(x, slogdets - slogdetcov(x, *p_opt), c='r')

residual = slogdets - slogdetcov(x, *p_opt)
residual = residual.flatten()

fig, ax = plt.subplots()
ax.scatter(x, residual)


raise a
for i in range(10):

    y, labels, target, kwds = utils.generate_data(K=100)

    visualization_handler = visualize.VisualizationHandler(
        y, target=target, figure_path="tmp/")

    search_model = mixture.GaussianMixture()
    search_model.kmeans_search(y, kwds["centers"] + 10, 
        visualization_handler=visualization_handler)


    K = np.array(search_model._state_K)

    ke = int(np.ceil(K.size**0.5))
    fig, axes = plt.subplots(ke, ke, sharex=True)
    axes = np.array(axes).flatten()

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        

    for i, (k, ax) in enumerate(zip(K, axes)):
        ax.hist(np.log(search_model._state_det_covs[i]), facecolor="g", alpha=0.5)
        ax.text(0.05, 0.95, "{:.0f}".format(k), transform=ax.transAxes,
            verticalalignment="top")

    fig.subplots_adjust(wspace=0, hspace=0, top=1.0, bottom=0.0, left=0, right=1)

    
    raise a



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