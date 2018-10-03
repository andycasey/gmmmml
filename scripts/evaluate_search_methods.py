
"""
Evaluate the performance of search strategies on generated data.
"""

import numpy as np
import os
import pickle
from collections import OrderedDict
from glob import glob
from time import time

from gmmmml import GaussianMixture

gmm_kwds = dict(threshold=1e-5, 
                max_em_iterations=1000,
                covariance_regularization=1e-10)

search_strategies = OrderedDict([
    ("KasarapuAllison2015", dict()),
    ("BayesStepper", dict()),
])

data_paths = sorted(glob("data/*.data"))

overwrite = False
raise_if_not_converged = False

# Initialise the evaluations.
N = len(data_paths)
ml = lambda I: I if isinstance(I, float) else np.sum(np.hstack(I.values()))

results_path_format = "{data_path}.{search_strategy}.output"

for search_strategy, search_kwds in search_strategies.items():

    for i, data_path in enumerate(data_paths):

        results_path = results_path_format.format(data_path=data_path, 
                                          search_strategy=search_strategy)

        if os.path.exists(results_path) and not overwrite:
            print(f"Skipping {i}/{N} ({data_path}) using {search_strategy} "\
                  f"because {results_path} exists and not overwriting")
            continue

        print(f"Running {i}/{N} ({data_path}) using {search_strategy}")

        with open(data_path, "rb") as fp:
            draws = pickle.load(fp)

        results = []
        for j, (X, meta) in enumerate(draws):

            model = GaussianMixture(**gmm_kwds)

            tick = time()

            try:
                model.search(X, search_strategy=search_strategy, **search_kwds)

            except:
                I = np.inf

            else:
                I = ml(model.message_lengths_)

            finally:
                tock = time()

            # Did we converge to the global solution?
            # (We gauge this by evaluating if we found a message length longer
            # than what is required to encode with the 'true' parameters)
            R_t, ll_t, I_t = model.expectation(X, **meta["truths"])
            I_t = ml(I_t)

            converged = (I <= I_t)

            result = dict(K=meta["K"], N=meta["N"], D=meta["D"],
                          converged=(I <= I_t),
                          I=I, I_t=I_t, time=tock - tick)

            results.append(result)

            print(converged, I, I_t, tock - tick)

            assert converged or not raise_if_not_converged

        with open(results_path, "wb") as fp:
            pickle.dump(results, fp)

