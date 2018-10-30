
"""
Evaluate greedy paths compared to a policy where we only split the component
with the largest message length.
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

from gmmmml import utils, GaussianMixture

random_seed = 123
np.random.seed(random_seed)

overwrite = True

output_path = "track_paths_results.pkl"

gmm_kwds = dict(threshold=1e-5, max_em_iterations=1000, covariance_regularization=0)
search_kwds = dict(quiet=True)

M = 1000

D = lambda *_: 2
K = lambda *_: int(np.random.uniform(1, 25))
N = lambda *_: int(np.random.uniform(1000, 10000))

ml = lambda I: I if isinstance(I, (int, float)) else np.sum(np.hstack(I.values()))


data_kwds = dict(center_box=(-10, 10), cluster_std=0.1)

if os.path.exists(output_path) and not overwrite:
    raise IOError(f"output path exists ({output_path}) and told not to overwrite")


results = []

for m in range(M):

    dk = data_kwds.copy()
    dk.update(N=N(), D=D(), K=K())
    print("DATA KWDS", dk)

    X, y, target, meta = utils.generate_isotropic_data(**dk)

    # Run Message-Breaking Search and track the path.

    model = GaussianMixture(**gmm_kwds)
    model.search(X, search_strategy="DebugMessageBreaking", **search_kwds)


    # Run Kasarapu-Allison and track the path.
    ka = GaussianMixture(**gmm_kwds)
    ka.search(X, search_strategy="DebugKasarapuAllison2015", **search_kwds)

    # Track the paths that were followed by Kasarapu & Allison.

    # sorted_index_chosen
    # diff_from_max_component_chosen
    # percentage_diff_from_max_component_chosen

    trace_path = []

    for each in ka.strategy_._message_length_of_components_split:

        I_chosen, *I_all = each
        sorted_index_chosen = len(I_all) - 1 - np.searchsorted(np.sort(I_all), I_chosen)
        diff_from_max_component_chosen = I_chosen - np.max(I_all)
        percentage_diff_from_max_component_chosen = diff_from_max_component_chosen/np.sum(I_all)

        denominator = np.max(I_all) - np.min(I_all)
        percentile_of_split = (I_chosen - np.min(I_all))/denominator
        

        trace_path.append([
            sorted_index_chosen,
            diff_from_max_component_chosen,
            percentage_diff_from_max_component_chosen,
            percentile_of_split
        ])

    results.append([
        trace_path,
        ml(ka.message_lengths_),
        ka.weights_.size,
        ka.meta_["t_search"],
        ml(model.message_lengths_),
        model.weights_.size,
        model.meta_["t_search"]
    ])

    print(results[-1][1:])
    print(results[-1][0])

    print(f"Saving results to {output_path}")
    with open(output_path, "wb") as fp:
        pickle.dump(results, fp, -1)

