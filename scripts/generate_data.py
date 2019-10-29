
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

max_N = 1e3
plot = True
generate_D = lambda N: [2]
generate_K = lambda N, D: np.unique(np.logspace(np.log10(N/5), np.log10(N/10), S).astype(int))

orders = np.atleast_2d(np.logspace(0, np.log10(max_N), 1 + np.log10(max_N))[1:])
steps = np.atleast_2d(np.round(np.logspace(0, 1, S)).astype(int))


Ns = np.unique((orders.T @ steps).flatten())

permutations = []
for N in Ns:
    for D in generate_D(N):
        for K in generate_K(N, D):
            permutations.append([N, D, K])

permutations = np.array(permutations).astype(int)

path_format = "data-permutations/{K:05d}_{N:06d}_{D:01d}_{M:01d}.data"

dirname = os.path.dirname(path_format)
if not os.path.exists(dirname):
    os.makedirs(dirname, exist_ok=True)

common_kwds = dict(dirichlet_concentration=100, isotropy=100, psi=0.1,
                   scale=max_N/10)

for i, (n, d, k) in enumerate(permutations):

    print(f"Generating data for N = {n}, D = {d}, K = {k}")

    draws = []
    for m in range(M):
        try:
            X, meta = utils.generate_data(N=n, D=d, K=k, **common_kwds)

        except ValueError:
            print(f"Failed to generate data for N={n}, D={d}, K={k}")
            continue

        assert np.all(np.isfinite(X))
        
        draws.append((X, meta))

        if plot and m < 1:

            fig, ax = plt.subplots()
            ax.scatter(X.T[0], X.T[1], s=1)
            ax.set_title(f"N={n}; D={d}; K={k}")


        kwds = dict(N=n, D=d, K=k, M=m)
        with open(path_format.format(**kwds), "wb") as fp:
            pickle.dump(draws, fp)

