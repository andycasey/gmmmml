
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

    bayesjumper_model2 = gmm.GaussianMixture()
    bayesjumper_model2.search(y, strategy="bayes-jumper")

    # The greedy k-means method requires a stopping criteria (K_max).
    K_max = 50
    gkh = visualize.VisualizationHandler(y, figure_prefix="tmp/greedy-kmeans")    
    greedykmeans_model = gmm.GaussianMixture()
    greedykmeans_model.search(y, 
                              K_max=K_max,
                              strategy="greedy-kmeans",
                              visualization_handler=gkh)


    # The Kasarapu and Allison method really breaks down if you don't use a 
    # little bit of covariance regularization.
    # TODO: Implement visualization handler with Kasarapu & Allison method.
    kasarapu_model = gmm.GaussianMixture()
    kasarapu_model.search(y,
                          strategy="kasarapu-allison-2015",
                          covariance_regularization=1e-6)


    bjh = visualize.VisualizationHandler(y, figure_prefix="tmp/bayes-jumper")

    bayesjumper_model = gmm.GaussianMixture()
    bayesjumper_model.search(y, 
                             strategy="bayes-jumper",
                             visualization_handler=bjh)



    print(
      """
      Summary:

      - Greedy K-means++ model took {0:.0f} seconds (K_max={1}; incl. visualisations)
      - Kasarapu & Allison model took {2:.0f} seconds (with no visualisations)
      - Bayes jumper model took {3:.0f} seconds (incl. visualisations; {4:.0f} seconds without visualisations)
      """.format(
        greedykmeans_model.meta_["t_search"], K_max,
        kasarapu_model.meta_["t_search"],
        bayesjumper_model.meta_["t_search"],
        bayesjumper_model2.meta_["t_search"]
    ))

    break

print("Fin")