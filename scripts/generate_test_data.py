
"""
Generate test data to compare methods.
"""

import numpy as np
import os
import pickle

from gmmmml import utils


random_seed = 42

path_format = "data/{K:05d}_data.pkl"

dirname = os.path.dirname(path_format)
if not os.path.exists(dirname):
    os.makedirs(dirname, exist_ok=True)


Ks = np.round(np.logspace(0, 3, 7)).astype(int)

common_kwds = dict(D=5, dirichlet_concentration=100, isotropy=10, 
                   psi=0.001, random_seed=random_seed)



for K in Ks:
    print(f"Generating data with K = {K} and {common_kwds}")

    X, meta = utils.generate_data(K=K, N = 10*K, **common_kwds)

    with open(path_format.format(K=K), "wb") as fp:
        pickle.dump((X, meta), fp, -1)







