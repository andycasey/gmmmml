
"""
Plot the results from the evaluations on artificial data.
"""

# TODO: Get these from somewhere else?

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import pickle
from collections import OrderedDict

from glob import glob

search_strategies = OrderedDict([
    ("BayesStepper", dict()),
    ("KasarapuAllison2015", dict())
])

results_path_template = "data/*{search_strategy}.output"

times = dict()
result_keys = ["K", "N", "D", "I", "I_t", "time"]
        
for search_strategy in search_strategies.keys():

    results_paths = glob(
        results_path_template.format(search_strategy=search_strategy))
    N = len(results_paths)

    print(f"Collecting {N} result files on {search_strategy}")

    if N < 1:
        continue

    times[search_strategy] = []

    for i, results_path in enumerate(results_paths):

        print(f"At {i}/{N}: {results_path}")

        with open(results_path, "rb") as fp:
            results = pickle.load(fp)

        # Average draws?
        assert len(results) == 1

        for result in results:
            times[search_strategy].append([result[k] for k in result_keys])


    times[search_strategy] = np.array(times[search_strategy])


# Plot as a function of N, D, K

x_labels = ["N", "D", "K", "ND", "NK"]
y_label = "time"

L = len(x_labels)
fig, axes = plt.subplots(1, L, figsize=(4 * L, 4))


max_y = 0
upper_lim = lambda existing: 10**(1 + np.ceil(np.log10(np.max(existing))))

for i, (ax, x_label) in enumerate(zip(axes, x_labels)):

    try:
        x_idx = result_keys.index(x_label)
    
    except ValueError:
        x_idx = np.array([result_keys.index(_) for _ in x_label])

    y_idx = result_keys.index(y_label)

    max_x = 0
    for search_strategy, data in times.items():

        x = data.T[x_idx]
        y = data.T[y_idx]

        if len(x.shape) > 1:
            x = np.product(x, axis=0)

        converged = data.T[result_keys.index("I")] <= data.T[result_keys.index("I_t")]


        idx = np.argsort(x)
        x, y, converged = x[idx], y[idx], converged[idx]

        max_y = max(y.max(), max_y)
        max_x = max(x.max(), max_x)


        scat = ax.scatter(x[converged], y[converged], label=search_strategy)

        ax.scatter(x[~converged], y[~converged], alpha=0.5, c=scat.get_facecolor())

    ax.loglog()
    ax.set_xlim(1e-0, upper_lim([max_x]))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


for ax in axes:
    ax.set_ylim(1e-2, upper_lim([max_y]))

fig.tight_layout()


x_labels = ["K", "N", "D"]


# Fit the cost.
for search_strategy, data in times.items():

    x_idx = np.array([result_keys.index(x_label) for x_label in x_labels])
    y_idx = result_keys.index(y_label)

    log_x = np.log(data.T[x_idx])
    log_y = np.log(data.T[y_idx])

    f = lambda _, *p: p @ log_x

    i, j = log_x.shape
    p_opt, p_cov = op.curve_fit(f, np.ones(j), log_y, p0=np.ones(i))

    # Plot the time relative to K
    order_repr = "".join([f"[{x}^{p:.1f}]" for x, p in zip(x_labels, p_opt)])

    print(f"{search_strategy}: O({order_repr})")

raise a
