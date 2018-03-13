
import numpy as np
import scipy.optimize as op
import pickle



with open("lls.pkl", "rb") as fp:
    K, sll = pickle.load(fp)

K = np.array(K)
sll = np.array(sll)

from gmmmml.mixture_search import _group_over

Ku, sllu = _group_over(K, sll, np.max)


Ns = np.arange(2, 100, 4).astype(int)

saved_p = np.zeros((Ns.size, 3))
saved_p2 = np.zeros_like(saved_p)

for i, N in enumerate(Ns):


    fig, axes = plt.subplots(3)
    ax = axes[0]
    ax.set_title("N = {}".format(N))
    ax.scatter(K, sll, facecolor="#666666")
    ax.scatter(Ku, sllu)

    # normalize to the initial value.
    ax = axes[1]
    normalization = sllu[0]
    ax.scatter(K, sll/normalization, facecolor="#666666")
    ax.scatter(Ku, sllu/normalization)


    #ax = axes[2]
    #ax.scatter(K[1:], np.diff(sll))

    random = False
    np.random.seed(2)

    idx = np.random.choice(len(Ku), N, replace=False) if random else np.arange(N)



    x_fit = Ku[idx]
    y_fit = sllu[idx]/normalization



    f_relative_ll = lambda k, *p: p[0]*np.array(k, dtype=float)**p[2] + p[1]
    f_ll = lambda k, *p: normalization * (p[0] * np.exp(k * p[1]) + p[2])

    f_ll2 = lambda k, *p: (p[0] * np.exp(k * p[1]) + p[2])


    def objective_function(x, y, p):
        return np.sum((y - f_ll(x, *p))**2)

    def ln_prior(theta):
        if theta[0] > 1 or theta[0] < 0 or theta[1] > 0:
            return 0
        return -0.5 * (theta[0] - 0.5)**2 / 0.05**2

    def lnlike(theta, x, y):
        return -0.5 * np.sum((y - f_ll2(x, *theta))**2)


    def ln_prob(theta, x, y):
        lp = ln_prior(theta)
        if not np.isfinite(lp):
            print(theta, "bad")
            return -np.inf
        return lp + lnlike(theta, x, y)

    """
    p0 = []
    while True:
        try:
            f_relative_ll(K, *p0)

        except IndexError:
            p0.append(-0.5)

        else:
            break
    """


    p0 = [0.5, -0.10, 0.5]
    bounds = np.array([(-np.inf, np.inf) for p in p0])
    bounds[0] = (0, 1)
    bounds[1] = (-np.inf, 0)

    p_opt, p_cov = op.curve_fit(f_ll, x_fit, y_fit * normalization, 
        p0=p0, maxfev=100000, bounds=bounds.T)

    p_opt2, f, d = op.fmin_l_bfgs_b(lambda *args: -ln_prob(*args), fprime=None,
        args=(x_fit, y_fit), x0=p0, approx_grad=True)


    print(N, objective_function(x_fit, y_fit, p_opt2)/objective_function(x_fit, y_fit, p_opt))

    axes[1].scatter(x_fit, y_fit, c='r')
    axes[1].plot(K, f_ll(K, *p_opt)/normalization, c='r')
    axes[1].plot(K, f_ll(K, *p0)/normalization, c='g')

    axes[2].scatter(K, 100 * (f_ll(K, *p_opt) - sllu)/sllu)
    axes[2].scatter(K, 100 * (f_ll(K, *p_opt2) - sllu)/sllu, c="m")

    axes[0].plot(K, f_ll(K, *p_opt), c='r')
    axes[0].plot(K, f_ll(K, *p0), c='g')
    axes[0].scatter(K[idx], sllu[idx], c='r')
    axes[0].plot(K, f_ll(K, *p_opt2), c='m')

    axes[0].set_title("N = {}, p0 = {:.2f}, p1 = {:.2f}".format(
        N, *p_opt))


    saved_p[i][:len(p_opt)] = p_opt
    saved_p2[i][:len(p_opt2)] = p_opt2

fig, axes = plt.subplots(3)

for i, ax in enumerate(axes):
    ax.scatter(Ns, saved_p.T[i])
    ax.scatter(Ns, saved_p2.T[i], c="m")

