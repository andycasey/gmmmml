
"""
Generate test data to compare methods.
"""

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt


from gmmmml import utils


np.random.seed(0)

M = 1 # number of draws per unit of N, K, D
S = 4 # number of steps to do per log-10 increment
generate_D = lambda N: 2 # dimensionality of the data.
generate_K = lambda N, D: min(int(N/10.), 1000)

max_N = 1e4
plot = True


orders = np.atleast_2d(np.logspace(0, np.log10(max_N), 1 + np.log10(max_N))[1:])
steps = np.atleast_2d(np.round(np.logspace(0, 1, S)).astype(int))


Ns = np.unique((orders.T @ steps).flatten())
Ds = np.array([generate_D(n) for n in Ns])

N, D = np.meshgrid(Ns, Ds)

# Uniquify.
N, D = np.unique(np.vstack([N.flatten(), D.flatten()]), axis=1).astype(int)
K = np.array([generate_K(n, d) for n, d in zip(N, D)])

path_format = "data_2d/{K:05d}_{N:06d}_{D:01d}.data"

dirname = os.path.dirname(path_format)
if not os.path.exists(dirname):
    os.makedirs(dirname, exist_ok=True)

common_kwds = dict(dirichlet_concentration=100, isotropy=100, psi=0.1,
                   scale=10)

for i, (n, d, k) in enumerate(zip(N, D, K)):

    print(f"Generating data for N = {n}, D = {d}, K = {k}")

    draws = []
    for m in range(M):
        X, meta = utils.generate_data(N=n, D=d, K=k, **common_kwds)
        draws.append((X, meta))

        if plot and m < 1:

            fig, ax = plt.subplots()
            ax.scatter(X.T[0], X.T[1], s=1)
            ax.set_title(f"N={n}; D={d}; K={k}")


    kwds = dict(N=n, D=d, K=k)
    with open(path_format.format(**kwds), "wb") as fp:
        pickle.dump(draws, fp)

