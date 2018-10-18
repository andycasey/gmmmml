
"""
Plot the results from the evaluations on artificial data.
"""

# TODO: Get these from somewhere else?

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as op
import pickle
from collections import OrderedDict

from glob import glob

from mpl_utils import mpl_style

matplotlib.style.use(mpl_style)

search_strategies = OrderedDict([
    ("BayesStepper", dict()),
    ("KasarapuAllison2015", dict()),
    ("MessageJumping", dict())
#    ("BayesJumper", dict())
])

results_path_template = "data_2d/*{search_strategy}.output"

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

        #print(f"At {i}/{N}: {results_path}")

        with open(results_path, "rb") as fp:
            results = pickle.load(fp)

        # Average draws?
        assert len(results) == 1

        for result in results:
            times[search_strategy].append([result[k] for k in result_keys])


    times[search_strategy] = np.array(times[search_strategy])


# Plot as a function of N, K

x_labels = ["K"]
y_label = "time"

latex_labels = dict(time=r"$\textrm{time}\,/\,\textrm{seconds}$",
                    K=r"$\textrm{number of true clusters}$ $K$",
                    KasarapuAllison2015=r"$\textrm{Kasarapu \& Allison (2015)}$",
                    BayesStepper=r"$\textrm{Message-Breaking Method}$",
                    MessageJumping=r"$\textrm{Message-Jumping Method}$")

scat_kwds = dict(KasarapuAllison2015=dict(marker="s", s=50),
                 BayesStepper=dict(s=50))

L = len(x_labels)
fig, axes = plt.subplots(1, L, figsize=(5 * L, 5))
axes = np.atleast_1d([axes]).flatten()


max_y = 0
upper_lim = lambda existing: 10**(1 + np.ceil(np.log10(np.max(existing))))

for i, (ax, x_label) in enumerate(zip(axes, x_labels)):


    x_idx = result_keys.index(x_label)

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

        lx, ly = (np.log10(x), np.log10(y))
        ly_err = 1e-2 * np.ones_like(ly)
        
        A = np.vstack((np.ones_like(lx), lx)).T
        C = np.diag(ly_err**2)

        cov = np.linalg.inv(A.T @ np.linalg.solve(C, A))
        mu = cov @ (A.T @ np.linalg.solve(C, ly))

        xi = np.array([x.min(), x.max()])
        yi = 10**np.polyval(mu[::-1], np.log10(xi))

        ax.plot(xi, yi, "-", label=f"$\mathcal{{O}}({x_label}^{{{mu[1]:.1f}}})$")
        _kwds = dict(label=latex_labels.get(search_strategy, search_strategy))
        _kwds.update(scat_kwds.get(search_strategy, dict()))

        scat = ax.scatter(x[converged], y[converged], **_kwds)

        _kwds.pop("label")
        ax.scatter(x[~converged], y[~converged], alpha=0.5, c=scat.get_facecolor(), **_kwds)

        if x_label == "N":
            Np = 10000
            Yp = 10**np.polyval(mu[::-1], np.log10(Np))
            print(f"Time for {search_strategy} on N = {Np} is {Yp:.0f}")

        elif x_label == "K":
            Kp = 1000
            Yp = 10**np.polyval(mu[::-1], np.log10(Kp))
            print(f"Time for {search_strategy} on K = {Kp} is {Yp:.0f}")



        """
        draws = np.random.multivariate_normal(mu, cov, size=100)[:, ::-1]

        yi_draw = np.array([10**np.polyval(draw, np.log10(xi)) for draw in draws])
        yi_lower, yi_upper = np.percentile(yi_draw, [16, 84], axis=0)

        # project error.
        for each in yi_draw:
            ax.plot(xi, each, alpha=0.01, c=scat.get_facecolor()[0])
        """

        
    ax.loglog()

    ax.set_xlim(0.5, upper_lim([max_x]))


    ax.set_xlabel(latex_labels.get(x_label, x_label))
    ax.set_ylabel(latex_labels.get(y_label, y_label))
    ax.legend(frameon=False)


for ax in axes:
    ax.set_ylim(1e-2, upper_lim([max_y]))

fig.tight_layout()

fig.savefig("article/cost.pdf", dpi=300)

raise a


fit_x_labels = ["KN"]


# Fit the cost.
for search_strategy, data in times.items():

    y_idx = result_keys.index(y_label)
    log_y = np.log10(data.T[y_idx])
    log_x = np.zeros((len(fit_x_labels), len(log_y)))

    for i, x_label in enumerate(fit_x_labels):
        try:
            log_x[i] = np.log10(data.T[result_keys.index(x_label)])

        except ValueError:
            # assume product of individuals
            x_idxs = np.array([result_keys.index(ea) for ea in x_label])
            log_x[i] = np.sum(np.log10(data.T[x_idxs]), axis=0)

    f = lambda _, *p: p @ log_x

    i, j = log_x.shape
    p_opt, p_cov = op.curve_fit(f, np.ones(j), log_y, p0=np.ones(i))

    # Plot the time relative to K
    order_repr = "".join([f"[({x})^{p:.1f}]" for x, p in zip(fit_x_labels, p_opt)])

    print(f"{search_strategy}: O({order_repr})")


"""
for ax, x_label in zip(axes, x_labels):
    x = data.T[x_labels.index(x_label)]
"""