
import numpy as np
from gmmmml import (gmm, visualize, utils)

np.random.seed(101)


import logging

for i in range(10):

    j = 0
    while True:
        print(i, j)
        j += 1
        try:
            y, labels, target, kwds = utils.generate_data(N=1000, K=35, D=9, center_box=(-10, 10))

        except:
          logging.exception("failed")
          if j > 3:
            raise 
          continue 

        else:
          break

    gmm_kwds = dict(threshold=1e-5, expected_improvement_fraction=1e-2,
                    covariance_regularization=1e-2)

    #bayesjumper_model2 = gmm.GaussianMixture()
    #bayesjumper_model2.search(y, search_strategy="bayes-jumper", **gmm_kwds)


    bjh = visualize.VisualizationHandler(y, figure_prefix="splitter/bj")

    bayesjumper_model = gmm.GaussianMixture()
    bayesjumper_model.search(y, 
                             search_strategy="bayes-jumper",
                             visualization_handler=bjh,
                             **gmm_kwds)

    raise a