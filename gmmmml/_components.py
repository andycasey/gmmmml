
def split_component(y, mu, cov, weight, responsibility, index, **kwargs):
    r"""
    Split a component from the current mixture and determine the new optimal
    state.

    :param y:
        A :math:`N\times{}D` array of the observations :math:`y`,
        where :math:`N` is the number of observations, and :math:`D` is the
        number of dimensions per observation.

    :param mu:
        The current estimates of the Gaussian mean values.

    :param cov:
        The current estimates of the Gaussian covariance matrices.

    :param weight:
        The current estimates of the relative mixing weight.

    :param responsibility:
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component.

    :param index:
        The index of the component to be split.

    :param covariance_type: [optional]
        The structure of the covariance matrix for individual components.
        The available options are: `free` for a free covariance matrix,
        `diag` for a diagonal covariance matrix, `tied` for a common covariance
        matrix for all components, `tied_diag` for a common diagonal
        covariance matrix for all components (default: ``free``).

    :returns:
        A six length tuple containing: the updated multivariate mean values,
        the updated covariance matrices, the updated mixture weights, the
        updated responsibility matrix, a metadata dictionary, and the change
        in message length.
    """

    # TODO: Current implementation only allows for a component to be split
    #       into two sub-components

    logger.debug("Splitting component {} of {}".format(index, weight.size))

    M = weight.size
    N, D = y.shape
    
    # Compute the direction of maximum variance of the parent component, and
    # locate two points which are one standard deviation away on either side.
    U, S, V = _svd(cov[index], kwargs["covariance_type"])

    child_mu = mu[index] - np.vstack([+V[0], -V[0]]) * S[0]**0.5

    assert np.all(np.isfinite(child_mu))

    # Responsibilities are initialized by allocating the data points to the 
    # closest of the two means.
    distance = np.vstack([
        np.sum((y - child_mu[0])**2, axis=1),
        np.sum((y - child_mu[1])**2, axis=1)
    ])
    
    child_responsibility = np.zeros((2, N))
    child_responsibility[np.argmin(distance, axis=0), np.arange(N)] = 1.0

    # Calculate the child covariance matrices.
    child_cov = _estimate_covariance_matrix(y, child_responsibility, child_mu,
        kwargs["covariance_type"], kwargs["covariance_regularization"])

    child_effective_membership = np.sum(child_responsibility, axis=1)    
    child_weight = child_effective_membership.T/child_effective_membership.sum()

    # We will need these later.
    parent_weight = weight[index]
    parent_responsibility = responsibility[index]

    """
    # Run expectation-maximization on the child mixtures.
    child_mu, child_cov, child_weight, child_responsibility, meta, dl = \
        _expectation_maximization(y, child_mu, child_cov, child_weight, 
            responsibility=child_responsibility, 
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

    if M > 1:

        # Integrate the M + 1 components and run expectation-maximization
        weight = np.hstack([weight, [parent_weight * child_weight[1]]])
        weight[index] = parent_weight * child_weight[0]

        responsibility = np.vstack([responsibility, 
            [parent_responsibility * child_responsibility[1]]])
        responsibility[index] = parent_responsibility * child_responsibility[0]
        
        mu = np.vstack([mu, [child_mu[1]]])
        mu[index] = child_mu[0]

        cov = np.vstack([cov, [child_cov[1]]])
        cov[index] = child_cov[0]

        mu, cov, weight, responsibility, meta, ml = _expectation_maximization(
            y, mu, cov, weight, responsibility, **kwargs)


    else:
        # Simple case where we don't have to re-run E-M because there was only
        # one component to split.
        child_mu, child_cov, child_weight, child_responsibility, meta, ml = \
            _expectation_maximization(y, child_mu, child_cov, child_weight, 
            responsibility=child_responsibility, 
            parent_responsibility=parent_responsibility, **kwargs)

        mu, cov, weight, responsibility \
            = (child_mu, child_cov, child_weight, child_responsibility)

    return (mu, cov, weight, responsibility, meta, ml)


def delete_component(y, mu, cov, weight, responsibility, index, **kwargs):
    r"""
    Delete a component from the mixture, and return the new optimal state.

    :param y:
        A :math:`N\times{}D` array of the observations :math:`y`,
        where :math:`N` is the number of observations, and :math:`D` is the
        number of dimensions per observation.

    :param mu:
        The current estimates of the Gaussian mean values.

    :param cov:
        The current estimates of the Gaussian covariance matrices.

    :param weight:
        The current estimates of the relative mixing weight.

    :param responsibility:
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component.

    :param index:
        The index of the component to be deleted.

    :param covariance_type: [optional]
        The structure of the covariance matrix for individual components.
        The available options are: `free` for a free covariance matrix,
        `diag` for a diagonal covariance matrix, `tied` for a common covariance
        matrix for all components, `tied_diag` for a common diagonal
        covariance matrix for all components (default: ``free``).

    :returns:
        A six length tuple containing: the updated multivariate mean values,
        the updated covariance matrices, the updated mixture weights, the
        updated responsibility matrix, a metadata dictionary, and the change
        in message length.
    """

    logger.debug("Deleting component {} of {}".format(index, weight.size))

    # Create new component weights.
    parent_weight = weight[index]
    parent_responsibility = responsibility[index]
    
    # Eq. 54-55
    new_weight = np.clip(
        np.delete(weight, index, axis=0)/(1-parent_weight),
        0, 1)
    
    # Calculate the new responsibility safely.
    new_responsibility = np.clip(
        np.delete(responsibility, index, axis=0) / (1 - parent_responsibility),
        0, 1)
    new_responsibility[~np.isfinite(new_responsibility)] = 0.0

    assert np.all(np.isfinite(new_responsibility))
    assert np.all(np.isfinite(new_weight))

    new_mu = np.delete(mu, index, axis=0)
    new_cov = np.delete(cov, index, axis=0)

    # Run expectation-maximizaton on the perturbed mixtures. 
    return _expectation_maximization(y, new_mu, new_cov, new_weight, 
        new_responsibility, **kwargs)


def merge_component(y, mu, cov, weight, responsibility, index, **kwargs):
    r"""
    Merge a component from the mixture with its "closest" component, as
    judged by the Kullback-Leibler distance.

    :param y:
        A :math:`N\times{}D` array of the observations :math:`y`,
        where :math:`N` is the number of observations, and :math:`D` is the
        number of dimensions per observation.

    :param mu:
        The current estimates of the Gaussian mean values.

    :param cov:
        The current estimates of the Gaussian covariance matrices.

    :param weight:
        The current estimates of the relative mixing weight.

    :param responsibility:
        The responsibility matrix for all :math:`N` observations being
        partially assigned to each :math:`K` component.

    :param index:
        The index of the component to be deleted.

    :param covariance_type: [optional]
        The structure of the covariance matrix for individual components.
        The available options are: `free` for a free covariance matrix,
        `diag` for a diagonal covariance matrix, `tied` for a common covariance
        matrix for all components, `tied_diag` for a common diagonal
        covariance matrix for all components (default: ``free``).

    :returns:
        A six length tuple containing: the updated multivariate mean values,
        the updated covariance matrices, the updated mixture weights, the
        updated responsibility matrix, a metadata dictionary, and the change
        in message length.
    """

    # Calculate the Kullback-Leibler distance to the other distributions.
    D_kl = np.inf * np.ones(weight.size)
    for m in range(weight.size):
        if m == index: continue
        D_kl[m] = kullback_leibler_for_multivariate_normals(
            mu[index], cov[index], mu[m], cov[m])

    a_index, b_index = (index, np.nanargmin(D_kl))

    logger.debug("Merging component {} (of {}) with {}".format(
        a_index, weight.size, b_index))


    # Initialize.
    weight_k = np.sum(weight[[a_index, b_index]])
    responsibility_k = np.sum(responsibility[[a_index, b_index]], axis=0)
    effective_membership_k = np.sum(responsibility_k)

    mu_k = np.sum(responsibility_k * y.T, axis=1) / effective_membership_k
    cov_k = _estimate_covariance_matrix(
        y, np.atleast_2d(responsibility_k), np.atleast_2d(mu_k), 
        kwargs["covariance_type"], kwargs["covariance_regularization"])

    # Delete the b-th component.
    del_index = np.max([a_index, b_index])
    keep_index = np.min([a_index, b_index])

    new_mu = np.delete(mu, del_index, axis=0)
    new_cov = np.delete(cov, del_index, axis=0)
    new_weight = np.delete(weight, del_index, axis=0)
    new_responsibility = np.delete(responsibility, del_index, axis=0)

    new_mu[keep_index] = mu_k
    new_cov[keep_index] = cov_k
    new_weight[keep_index] = weight_k
    new_responsibility[keep_index] = responsibility_k

    # Calculate log-likelihood.
    return _expectation_maximization(y, new_mu, new_cov, new_weight,
        responsibility=new_responsibility,  **kwargs)




