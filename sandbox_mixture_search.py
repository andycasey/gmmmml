
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

    j_handler = visualize.VisualizationHandler(y, figure_prefix="tmp/bayes-jumper")
    k_handler = visualize.VisualizationHandler(y, figure_prefix="tmp/greedy-kmeans")

    j_model = gmm.GaussianMixture()
    j_model.search(y, strategy="bayes-jumper", visualization_handler=j_handler)


    k_model = gmm.GaussianMixture()
    k_model.search(y, strategy="greedy-kmeans", visualization_handler=k_handler,
                   K_max=50)

    #j_handler.create_movie()
    #k_handler.create_movie()

    break

print("Fin")