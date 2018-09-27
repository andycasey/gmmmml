
"""
Operations to apply to gaussian mixtures.
"""

import logging
import numpy as np
import scipy
from sklearn import cluster
from sklearn.utils import check_random_state
from sklearn.utils.extmath import row_norms


from . import (em, mml, utils)

logger = logging.getLogger(__name__)




def kmeans_pp(y, K, random_state=None, **kwargs):
    r"""
    Initialize a Gaussian mixture model using the K-means++ algorithm.

    :param y:
        The data values, :math:`y`, which are expected to have :math:`N` 
        samples each with :math:`D` dimensions. Expected shape of :math:`y` 
        is :math:`(N, D)`.

    :param K:
        The number of Gaussian components in the mixture.
    
    :param random_state: [optional]
        The state to provide to the random number generator.

    :returns:
        A four-length tuple containing:

        (1) an estimate of the means of the components

        (2) an estimate of the covariance matrices of the components

        (3) an estimate of the relative mixings of the components

        (4) the responsibility matrix for each data point to each component.
    """

    random_state = check_random_state(random_state)
    squared_norms = row_norms(y, squared=True)
    means = cluster.k_means_._k_init(y, K,
                                     random_state=random_state,
                                     x_squared_norms=squared_norms)

    labels = np.argmin(scipy.spatial.distance.cdist(means, y), axis=0)

    N, D = y.shape
    responsibilities = np.zeros((K, N))
    responsibilities[labels, np.arange(N)] = 1.0

    covs = em._estimate_covariance_matrix_full(y, means, responsibilities, 
        covariance_regularization=kwargs.get("covariance_regularization", 0))

    weights = responsibilities.sum(axis=1)/N

    return (means, covs, weights, responsibilities)



def split_component(y, means, covs, weights, responsibilities, index, split=2,
                    **kwargs):
    r"""
    Split a component from the current mixture and determine the new optimal
    state.

    :param y:
        A :math:`N\times{}D` array of the observations :math:`y`,
        where :math:`N` is the number of observations, and :math:`D` is the
        number of dimensions per observation.

    :param means:
        The current estimates of the Gaussian mean values.

    :param covs:
        The current estimates of the Gaussian covariance matrices.

    :param weights:
        The current estimates of the relative mixing weights.

    :param responsibilities:
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component.

    :param index:
        The index of the component to be split.

    :param split: [optional]
        The number of components to split the component into. Default to 2, so
        one component will be split into two.
    """
    logger.debug("Splitting component {} of {}".format(index, weights.size))

    K, N, D = (weights.size, *y.shape)
    
    # Compute the direction of maximum variance of the parent component, and
    # locate two points which are one standard deviation away on either side.
    U, S, V = _svd(covs[index], **kwargs)

    # Pick points along the eigenvector.
    sigma_points = np.atleast_2d(np.linspace(-3, 3, 2 + split)[1:-1])
    projection = np.atleast_2d(V[0] * S[0]**0.5)

    child_means = means[index] + sigma_points.T @ projection

    # Responsibilities are initialized by allocating the data points to the 
    # closest of the means.
    # TODO: I'm sure there's a better way to do this.
    distance = np.vstack([np.sum((y - cmu)**2, axis=1) for cmu in child_means])
    
    child_responsibilities = np.zeros((split, N))
    child_responsibilities[np.argmin(distance, axis=0), np.arange(N)] = 1.0

    # Calculate the child covariance matrices.
    child_covs = em._estimate_covariance_matrix(
        y, child_means, child_responsibilities, **kwargs)

    child_effective_membership = np.sum(child_responsibilities, axis=1)    
    child_weights = child_effective_membership.T/child_effective_membership.sum()

    # We will need these later.
    parent_weights = weights[index]
    parent_responsibilities = responsibilities[index]

    """
    # Run expectation-maximization on the child mixtures.
    child_means, child_cov, child_weight, child_responsibilities, meta, dl = \
        _expectation_maximization(y, child_means, child_cov, child_weight, 
            responsibility=child_responsibilities, 
            parent_responsibility=parent_responsibility,
            covariance_type=covariance_type, **kwargs)
    """

    # After the chld mixture is locally optimized, we need to integrate it
    # with the untouched M - 1 components to result in a M + 1 component
    # mixture M'.

    # An E-M is finally carried out on the combined M + 1 components to
    # estimate the parameters of M' and result in an optimized 
    # (M + 1)-component mixture.

    # Update the component weights.
    # Note that the child A mixture will remain in index `index`, and the
    # child B mixture will be appended to the end.

    if K > 1:

        # Integrate the K + delta_K components and run expectation-maximization
        weights = np.hstack([weights, parent_weights * child_weights[1:]])
        weights[index] = parent_weights * child_weights[0]

        responsibilities = np.vstack([responsibilities, 
            parent_responsibilities * child_responsibilities[1:]])
        responsibilities[index] = parent_responsibilities * child_responsibilities[0]
        
        means = np.vstack([means, child_means[1:]])
        means[index] = child_means[0]

        covs = np.vstack([covs, child_covs[1:]])
        covs[index] = child_covs[0]

        state, responsibilities, ll, I = em.expectation_maximization(
            y, means, covs, weights, 
            responsibilities=responsibilities, **kwargs)

    else:
        # Simple case where we don't have to re-run E-M because there was only
        # one component to split.

        state, responsibilities, ll, I = em.expectation_maximization(
            y, child_means, child_covs, child_weights, 
            responsibilities=child_responsibilities, 
            parent_responsibilities=parent_responsibilities, **kwargs)

    return (state, responsibilities, ll, I)


def delete_component(y, means, covs, weights, responsibilities, index, **kwargs):
    r"""
    Delete a component from the mixture, and return the new optimal state.

    :param y:
        A :math:`N\times{}D` array of the observations :math:`y`,
        where :math:`N` is the number of observations, and :math:`D` is the
        number of dimensions per observation.

    :param means:
        The current estimates of the Gaussian mean values.

    :param covs:
        The current estimates of the Gaussian covariance matrices.

    :param weights:
        The current estimates of the relative mixing weights.

    :param responsibilities:
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component.

    :param index:
        The index of the component to be deleted.
    """
    logger.debug(f"Deleting component {index} of {weights.size}")

    # Create new component weights.
    parent_weights = weights[index]
    parent_responsibilities = responsibilities[index]
    
    # Eq. 54-55
    weights_ = np.clip(
        np.delete(weights, index, axis=0)/(1 - parent_weights),
        0, 1)
    
    # Calculate the new responsibility safely.
    responsibilities_ = np.clip(np.delete(responsibilities, index, axis=0) \
                             / (1 - parent_responsibilities), 0, 1)
    responsibilities_[~np.isfinite(responsibilities_)] = 0.0

    means_ = np.delete(means, index, axis=0)
    covs_ = np.delete(covs, index, axis=0)

    # Run expectation-maximizaton on the perturbed mixtures. 
    return em.expectation_maximization(
        y, means_, covs_, weights_, responsibilities=responsibilities_, **kwargs)


def merge_component(y, means, covs, weights, responsibilities, index, **kwargs):
    r"""
    Merge a component from the mixture with its "closest" component, as
    judged by the Kullback-Leibler distance.

    :param y:
        A :math:`N\times{}D` array of the observations :math:`y`,
        where :math:`N` is the number of observations, and :math:`D` is the
        number of dimensions per observation.

    :param means:
        The current estimates of the Gaussian mean values.

    :param covs:
        The current estimates of the Gaussian covariance matrices.

    :param weights:
        The current estimates of the relative mixing weights.

    :param responsibilities:
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component.

    :param index:
        The index of the component to be merged.
    """

    # Calculate the Kullback-Leibler distance to the other distributions.
    D_kl = np.inf * np.ones(weights.size)
    for k in range(weights.size):
        if k == index: continue
        D_kl[k] = utils.kullback_leibler_for_multivariate_normals(
            means[index], covs[index], means[k], covs[k])

    a_index, b_index = (index, np.nanargmin(D_kl))

    K = weights.size
    logger.debug(f"Merging component {a_index} (of {K}) with {b_index}")

    # Initialize.
    weights_k = np.sum(weights[[a_index, b_index]])
    responsibilities_k = np.atleast_2d(
        np.sum(responsibilities[[a_index, b_index]], axis=0))
    effective_membership_k = np.sum(responsibilities_k)

    means_k = np.atleast_2d(
        np.sum(responsibilities_k * y.T, axis=1) / effective_membership_k)

    covs_k = em._estimate_covariance_matrix(
        y, means_k, responsibilities_k, **kwargs)

    # Delete the b-th component.
    del_index = np.max([a_index, b_index])
    keep_index = np.min([a_index, b_index])

    means_ = np.delete(means, del_index, axis=0)
    covs_ = np.delete(covs, del_index, axis=0)
    weights_ = np.delete(weights, del_index, axis=0)
    responsibilities_ = np.delete(responsibilities, del_index, axis=0)

    means_[keep_index] = means_k
    covs_[keep_index] = covs_k
    weights_[keep_index] = weights_k
    responsibilities_[keep_index] = responsibilities_k

    return em.expectation_maximization(
        y, means_, covs_, weights_, responsibilities=responsibilities_, **kwargs)


def _svd(covariance, covariance_type, **kwargs):

    if covariance_type == "full":
        return np.linalg.svd(covariance)

    elif covariance_type == "diag":
        return np.linalg.svd(covariance * np.eye(covariance.size))

    else:
        raise ValueError("unknown covariance type")



def preferred_mixture_index(K, previous_K):

    previous_K = np.atleast_1d(previous_K)
    diff = K - previous_K
    abs_diff = np.abs(diff)

    # Get the closest thing, with a preference for split over merge.
    idx = np.argsort(abs_diff)

    if idx.size > 1 and diff[idx[0]] == diff[idx[1]]:
        return idx[np.argmax(diff[idx[:2]])]

    return idx[0]


def iteratively_remove_components(y, means, covs, weights, K, **kwargs):
    r"""
    Iteratively remove components in a mixture until we reach a target
    distribution of :math:`K` Gaussian components.

    :param y:
        A :math:`N\times{}D` array of the observations :math:`y`,
        where :math:`N` is the number of observations, and :math:`D` is the
        number of dimensions per observation.

    :param means:
        The current estimates of the Gaussian mean values.

    :param covs:
        The current estimates of the Gaussian covariance matrices.

    :param weights:
        The current estimates of the relative mixing weights.

    :param K:
        The number of target Gaussian components.
    """

    if weights.size <= K:
        raise ValueError(f"the given mixture already has <={K} components")

    Ks = []
    Is = []
    while K < weights.size:

        # Delete the component with the largest message length.
        R, ll, I_components = em._component_expectations(y, means, covs, weights,
                                                         **kwargs)
        index = np.argsort(I_components)[-1]
        
        (means, covs, weights), responsibilities, ll, I = merge_component(
            y, means, covs, weights, R, index, **kwargs)

        print("Current state is (K = {}; I = {})".format(
            weights.size, np.sum(np.hstack(I.values()))))

        Ks.append(weights.size)
        Is.append(np.sum(np.hstack(I.values())))

    return (means, covs, weights, responsibilities, ll, I)


def iteratively_split_components(y, means, covs, weights, K, **kwargs):
    r"""
    Iteratively split and refine a mixture until we reach a target distribution
    of :math:`K` Gaussian components.

    :param y:
        A :math:`N\times{}D` array of the observations :math:`y`,
        where :math:`N` is the number of observations, and :math:`D` is the
        number of dimensions per observation.

    :param means:
        The current estimates of the Gaussian mean values.

    :param covs:
        The current estimates of the Gaussian covariance matrices.

    :param weights:
        The current estimates of the relative mixing weights.

    :param K:
        The number of target Gaussian components.
    """

    if weights.size >= K:
        raise ValueError(f"the given mixture already has >={K} components")


    Ks = []
    Is = []
    while K > weights.size:

        # Split the component with the largest message length.
        R, ll, I_components = em._component_expectations(y, means, covs, weights, 
                                                         **kwargs)
        index = np.argsort(I_components)[-1]
        
        # TODO: Allow us to get back the component-wise relative message lengths
        #       so we don't have to calculate the expectation step twice.

        (state, responsibilities, ll, I) = split_component(
            y, means, covs, weights, R, index, **kwargs)

        means, covs, weights = state


        Ks.append(weights.size)
        Is.append(np.sum(np.hstack(I.values())))

        print("Current state (K = {}; I = {})".format(Ks[-1], Is[-1]))

    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.scatter(Ks, Is)
    """

    return (state, responsibilities, ll, I)


def _repartition_split_mixture(y, means, covs, weights, K, **kwargs):
    r"""
    Split the components of a mixture until we reach the target distribution of
    :math:`K` components.
    """

    R, ll, I_components = em._component_expectations(y, means, covs, weights,
                                                     **kwargs)

    # How many new components do we need?
    K_new = K - weights.size

    if weights.size == 1:
        # Trivial.
        return split_component(y, means, covs, weights, R, 0, 1 + K_new, **kwargs)


    else:
        """
        # Approximate something like min(I_components), while ensuring that we
        # will meet the required constraint.

        # TODO: This strategy is ad-hoc and probably wrong.
        #       If anything, it should be better explained.
        alpha = np.sum(I_components)/(new_K + K)

        K_available = np.ceil(I_components/alpha).astype(int) - 1

        # Need an array of components to split, and how much we should split them.
        N_splits = np.sum(K_available > 0)

        indices = np.zeros(N_splits, dtype=int)
        splits = np.zeros(N_splits, dtype=int)
        idxs = np.argsort(I_components)[::-1]

        for i, index in enumerate(idxs[:N_splits]):
            indices[i] = index
            splits[i] = min(K_available[index], new_K - np.sum(splits))

        assert sum(splits) >= new_K
        if sum(splits) > new_K:
            raise a

        state = (means, covs, weights)
        for index, split in zip(indices, splits):
            state, R, ll, I = split_component(y, *state, R, index, 1 + split,
                                              **kwargs)
        """

        # Split the top K_new in half.
        idx = np.argsort(I_components)[::-1][:K_new]

        indices = np.zeros(K_new, dtype=int)
        splits = np.zeros(K_new, dtype=int)

        for i, index in enumerate(idx):
            indices[i] = index
            splits[i] = 2

        state = (means, covs, weights)
        for index, split in zip(indices, splits):
            state, R, ll, I = split_component(y, *state, R, index, split, **kwargs)

        return (state, R, ll, I)



def _repartition_merge_mixture(y, means, covs, weights, K, **kwargs):

    print("don't know how to do this yet -- requires a thinko")

    return iteratively_remove_components_greedily(y, means, covs, weights, K,
                                                  **kwargs)


def repartition_mixture(y, means, covs, weights, K, **kwargs):

    func = _repartition_merge_mixture if weights.size > K \
                                      else _repartition_split_mixture

    return func(y, means, covs, weights, K, **kwargs)



def iteratively_operate_components(y, means, covs, weights, K, **kwargs):

    func = iteratively_split_components if K > weights.size \
                                        else iteratively_remove_components

    return func(y, means, covs, weights, K, **kwargs)
        


