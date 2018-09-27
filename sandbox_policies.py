
import numpy as np
from gmmmml import (gmm, visualize, utils, strategies)

np.random.seed(101)


import logging

for i in range(10):

    j = 0
    while True:
        print(i, j)
        j += 1
        try:
            y, labels, target, kwds = utils.generate_data(N=10000, K=100, D=10, center_box=(-10, 10))

        except:
          logging.exception("failed")
          if j > 3:
            raise 
          continue 

        else:
          break

    gmm_kwds = dict(threshold=1e-5, expected_improvement_fraction=1e-5,
                    covariance_regularization=1e-10)

    #bayesjumper_model2 = gmm.GaussianMixture()
    #bayesjumper_model2.search(y, search_strategy="bayes-jumper", **gmm_kwds)

    from time import time

    model = gmm.GaussianMixture(**gmm_kwds)
    t_init = time()
    
    model.search(y, search_strategy="BayesJumper")
    t_bj = time() - t_init

    """
    model2 = gmm.GaussianMixture(**gmm_kwds)
    t_init = time()
    model2.search(y, search_strategy="KasarapuAllison2015")
    t_ka = time() - t_init
    """
    raise a