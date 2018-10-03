
"""
Generate test data to compare methods.
"""

import numpy as np
import os
import pickle

from gmmmml import utils


np.random.seed(123)

M = 1 # number of draws per unit of N, K, D

max_N, max_D = 10**4, 30

orders = np.atleast_2d(np.logspace(0, np.log10(max_N), 1 + np.log10(max_N))[1:])
steps = np.atleast_2d(np.round(np.logspace(0, 1, 4)).astype(int))

Nu = np.unique((orders.T @ steps).flatten())

Du = np.logspace(np.log10(2), np.log10(max_D), 6).astype(int)

N, D = np.meshgrid(Nu, Du)

K = np.clip(np.round([np.random.uniform(2, 0.1 * n) for n in N]),
            2, np.inf)

path_format = "data/{K:05d}_{N:06d}_{D:01d}.data"

dirname = os.path.dirname(path_format)
if not os.path.exists(dirname):
    os.makedirs(dirname, exist_ok=True)


common_kwds = dict(dirichlet_concentration=100, isotropy=10, psi=0.01)

N, D, K = (ea.flatten().astype(int) for ea in (N, D, K))

for i, (n, d, k) in enumerate(zip(N, D, K)):

    print(f"Generating data for N = {n}, D = {d}, K = {k}")

    draws = []
    for m in range(M):
        draws.append(utils.generate_data(N=n, D=d, K=k, **common_kwds))

    kwds = dict(N=n, D=d, K=k)
    with open(path_format.format(**kwds), "wb") as fp:
        pickle.dump(draws, fp)



