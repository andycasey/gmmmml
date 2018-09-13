
"""
Operations to apply to gaussian mixtures.
"""

import logging
import numpy as np

from . import (em, utils)

logger = logging.getLogger(__name__)



def split_component(y, means, covs, weights, responsibilities, index, **kwargs):
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
    """

    logger.debug("Splitting component {} of {}".format(index, weights.size))

    K, N, D = (weights.size, *y.shape)
    
    # Compute the direction of maximum variance of the parent component, and
    # locate two points which are one standard deviation away on either side.
    U, S, V = _svd(covs[index], **kwargs)

    child_means = means[index] - np.vstack([+V[0], -V[0]]) * S[0]**0.5

    # Responsibilities are initialized by allocating the data points to the 
    # closest of the two means.
    distance = np.vstack([
        np.sum((y - child_means[0])**2, axis=1),
        np.sum((y - child_means[1])**2, axis=1)
    ])
    
    child_responsibilities = np.zeros((2, N))
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

        # Integrate the M + 1 components and run expectation-maximization
        weights = np.hstack([weights, [parent_weights * child_weights[1]]])
        weights[index] = parent_weights * child_weights[0]

        responsibilities = np.vstack([responsibilities, 
            [parent_responsibilities * child_responsibilities[1]]])
        responsibilities[index] = parent_responsibilities * child_responsibilities[0]
        
        means = np.vstack([means, [child_means[1]]])
        means[index] = child_means[0]

        covs = np.vstack([covs, [child_covs[1]]])
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


