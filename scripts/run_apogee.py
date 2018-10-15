
"""
Run the BayesStepper search algorithm on APOGEE data.
"""

import numpy as np
import os
import pickle
from time import time
from astropy.table import Table

from gmmmml import GaussianMixture

gmm_kwds = dict(threshold=1e-3, 
                max_em_iterations=10000,
                covariance_regularization=1e-10)
search_kwds = dict(search_strategy="StrictBayesStepper")

apogee_input_path = "../ct/catalogs/tc-apogee-dr12-regularized-release.fits"

overwrite = False
output_path = "apogee-results-strict.test.pkl"

if os.path.exists(output_path) and not overwrite:
    raise IOError(f"output path '{output_path}' exists and not overwriting")

# Load APOGEE data.
apogee = Table.read(apogee_input_path)
element_label_names = ('AL_H', 'CA_H', 'C_H', 'FE_H', 'K_H', 'MG_H', 'MN_H', 
                       'NA_H', 'NI_H', 'N_H', 'O_H', 'SI_H', 'S_H', 'TI_H', 
                       'V_H')

X = np.array([apogee[ln] for ln in element_label_names]).T

# Run search.
tick = time()
model = GaussianMixture(**gmm_kwds)
model.search(X, **search_kwds)
tock = time()

result = dict(gmm_kwds=gmm_kwds, search_kwds=search_kwds, time_taken=tock - tick,
              means=model.means_, covs=model.covs_, weights=model.weights_,
              responsibilities=model.responsibilities_, 
              log_likelihoods=model.log_likelihoods_,
              message_lengths=model.message_lengths_,
              meta=model.meta_)

with open(output_path, "wb") as fp:
    pickle.dump(result, fp)

print(f"Search {search_kwds} found K = {model.weights_.size}")
print(f"Wrote result to {output_path}")