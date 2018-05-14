
import numpy as np
from gmmmml import (gmm as mixture, visualize)
from sklearn.datasets import make_blobs

seed = 1100

np.random.seed(seed)


# Generate some data.
y, assignments = make_blobs(
  n_samples=2500, n_features=2, centers=50, center_box=(-15, 15), random_state=seed)

# Create a visualisation handler to plot the progress.
visualization_handler = visualize.VisualizationHandler(y, target=None, 
  figure_path="tmp/")


# Search
search_model = mixture.GaussianMixture(
  max_em_iterations=5, covariance_regularization=1e-10)
search_model.search_greedy_forgetful(y, K_max=150, K_predict=100,
    visualization_handler=visualization_handler)

