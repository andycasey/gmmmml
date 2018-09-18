
import numpy as np
from gmmmml import (gmm, visualize, utils, operations)

np.random.seed(101)


import logging

for i in range(10):

    j = 0
    while True:
        print(i, j)
        j += 1
        try:
            y, labels, target, kwds = utils.generate_data(N=10000, K=35, D=3, center_box=(-10, 10))

        except:
          logging.exception("failed")
          if j > 3:
            raise 
          continue 

        else:
          break

    gkh = visualize.VisualizationHandler(y, figure_prefix="jump-tmp/bj")    

    model = gmm.GaussianMixture()
    model.search(y, strategy="bayes-jumper", visualization_handler=gkh)


    raise a

    # Take this to try the split stuff
    kwds = model._em_kwds.copy()
    kwds.update(covariance_regularization=1e-1, threshold=1e-1)


    means, covs, weights = model.means_, model.covs_, model.weights_
    means, covs, weights, _ = model.initialize(y, 15)

    (means, covs, weights), R, ll, I = operations.iteratively_split_components(y, means, covs, weights, K=165,
                                            **kwds)

    foo = operations.iteratively_remove_components(y, means, covs, weights, K=15, **kwds)



    raise a

    break

print("Fin")