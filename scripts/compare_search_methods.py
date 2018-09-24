
"""
Compare methods on generated data.
"""

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pickle

from gmmmml import GaussianMixture


# Methods:
# - greedy-kmeans
# - bayes-jumper
# - kasarapu

data_paths = glob("data/?????_data.pkl")

data_paths = sorted(data_paths)

gmm_kwds = dict(threshold=1e-5, max_em_iterations=1000)

comparison_kwds = dict([
    ("kasarapu-allison-2015", dict(covariance_regularization=1e-10)),
    ("bayes-jumper", dict()),
])



times = {}
K_est = {}


for i, data_path in enumerate(data_paths):

    with open(data_path, "rb") as fp:
        X, meta = pickle.load(fp)

    times.setdefault(data_path, dict(meta=meta))
    K_est.setdefault(data_path, dict())


    for j, (strategy, search_kwds) in enumerate(comparison_kwds.items()):
        assert strategy != "meta", "This will stuff things up"

        model = GaussianMixture(**gmm_kwds)
        try:
            model.search(X, strategy=strategy, **search_kwds)

        except:
            result = np.nan
            K_ = np.nan

        else:
            result = model.meta_["t_search"]
            K_ = model.weights_.size

        times[data_path][strategy] = result
        K_est[data_path][strategy] = K_


# Plot times.
strategies = comparison_kwds.keys()

K = dict([(k, times[k]["meta"]["K"]) for k in times.keys()])
t = dict()
for strategy, kwds in comparison_kwds.items():
    t.setdefault(strategy, [])

x = []
y = []
for data_path, k in K.items():
    x.append(k)

    y_row = []
    for strategy in strategies:
        y_row.append(times[data_path][strategy])

    y.append(y_row)

x, y = (np.array(x), np.array(y))

fig, ax = plt.subplots()
for i, strategy in enumerate(strategies):
    ax.scatter(x, y.T[i], label=strategy)

plt.legend()

raise a
