
"""
Visualize the search progress for a gaussian mixture model.
"""

import os
import logging as logger
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator

from . import mml


class VisualizationHandler(object):

    def __init__(self, y, data_projection_indices=(0, 1), figure_path=None,
        target=None, **kwargs):
        r"""
        Initialize a visualisation handler to show the search progress and the
        predictions of future mixtures.

        :param y:
            The data :math:`y`.

        :param data_projection_indices: [optional]
            A two-length tuple containing the indices to use when plotting the
            :math:`x` and :math:`y` projections of the data.

        :param figure_path: [optional]
            The local path to store any figures generated during the search.

        :param target: [optional]
            The target message length components (e.g., for generated data).
        """

        self._y_shape = y.shape
        self._data_projection_indices = data_projection_indices

        self.__plot_bounds_kwds = dict(c="k", lw=2)
        self.__plot_prediction_kwds = dict(c="tab:blue", lw=2)
        self.__fill_between_prediction_kwds = dict(facecolor="tab:blue",
                                                   alpha=0.5, edgecolor=None,
                                                   zorder=-1)


        self._figure_path = "" if figure_path is None else figure_path
        self._figure_iter = 1
        self._figure_prefix = os.path.join(self._figure_path,
            "iter_{:.0f}".format(np.random.uniform(0,  10000)))

        self._fig, axes = plt.subplots(1, 5, figsize=(15.6 + 2.8, 2.8))
            
        self._init_axes(y, target)
        self.snapshot()

        # lists for plotting
        self._plot_model_items = []
        self._plot_prediction_items = {}


        return None


    def _init_axes(self, y, target=None):
        """
        Initialize the axes in the figure.

        :param y:
            The data :math:`y`.

        :param target:
            The target distribution (e.g.,  in the case of generated data).
        """

        scatter_data_kwds = dict(facecolor="k", s=1)
        scatter_progress_kwds = scatter_data_kwds.copy()
        scatter_progress_kwds.update(dict(s=5))

        x_index, y_index = self._data_projection_indices

        ax = self._ax("data")
        ax.scatter(y.T[x_index], y.T[y_index], **scatter_data_kwds)
        ax.set_xlabel(r"$x_{{{0}}}$".format(x_index))
        ax.set_ylabel(r"$x_{{{0}}}$".format(y_index))
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(3))
        
        ax = self._ax("slogw")
        ax.set_xlabel(r"$K$")
        ax.set_ylabel(r"$I_\mathcal{M} + \left(\frac{D(D+3)}{4} - \frac{1}{2}\right)\sum\log{w_k}$ $[{\rm nats}]$")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        self._slogw_data = []
        self._scatter_slogw_data = ax.scatter([np.nan], [np.nan],
                                              **scatter_progress_kwds)

        ax = self._ax("slogdet")
        ax.set_xlabel(r"$K$")
        ax.set_ylabel(r"$-\frac{(D+2)}{2}\sum\log{|C_k|}$ $[{\rm nats}]$")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        self._slogdet_data = []
        self._scatter_slogdet_data = ax.scatter([np.nan], [np.nan],
                                                **scatter_progress_kwds)
        
        ax = self._ax("nll")
        ax.set_xlabel(r"$K$")
        ax.set_ylabel(r"$-\sum\log{\mathcal{L}(y\|\theta)}$ $[{\rm nats}]$")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        self._nll_data  = []
        self._scatter_nll_data = ax.scatter([np.nan], [np.nan],
                                            **scatter_progress_kwds)

        ax = self._ax("I")
        ax.set_xlabel(r"$K$")
        ax.set_ylabel(r"$I$ $[{\rm nats}]$")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        self._I_data = []
        self._scatter_I_data = ax.scatter([np.nan], [np.nan],
                                          **scatter_progress_kwds)

        if target is not None:
            print("no target  implemented  yet")

            #raise NotImplementedError("not  dun yet")

        for ax in self._fig.axes:
            ax.autoscale(enable=True)

        self._fig.tight_layout()

        return None



    def _ax(self, descriptor):
        r"""
        Return the correct axes given a descriptor.
        """

        axes = np.array(self._fig.axes).flatten()
        index = ["data", "slogw", "slogdet", "nll", "I"].index(descriptor)
        return axes[index]


    def snapshot(self, **kwargs):
        """
        Save a snapshot (figure) of the current progress and predictions.
        """

        plt.draw()
        self._fig.tight_layout()
        path = "{0:s}_{1:05d}.png".format(self._figure_prefix, self._figure_iter)
        self._fig.savefig(path, **kwargs)
        print("Created {}".format(path))
        self._figure_iter += 1

        return None


    def _clear_items(self, items):
        """
        Hide items from a figure.

        :param items:
            A list of items plotted on an axis.
        """

        L = len(items)
        for l in range(L):
            item = items.pop(0)
            item.set_visible(False)
            del item

        return None


    def emit(self, kind, params, snapshot=True):
        r"""
        Handler for events.
        """

        if kind == "model":

            self._clear_items(self._plot_model_items)

            ax = self._ax("data")
            
            x_index, y_index = self._data_projection_indices

            N, D = self._y_shape
            K = params["weight"].size

            cov_mask = np.zeros((D, D), dtype=bool)
            cov_mask[y_index, x_index] = True
            cov_mask[x_index, y_index] = True
            cov_mask[x_index, x_index] = True
            cov_mask[y_index, y_index] = True

            for k in range(K):
                mu = params["mean"][k][[x_index, y_index]]
                cov = params["cov"][k][cov_mask].reshape((2, 2))

                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals = vals[order]
                vecs = vecs[:, order]

                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

                # Show 2 standard deviations.
                width, height = 2 * 2 * np.sqrt(vals)
                ellipse = Ellipse(xy=mu, width=width, height=height, angle=theta,
                    facecolor="r", alpha=0.25)

                self._plot_model_items.append(ax.add_artist(ellipse))
                self._plot_model_items.append(ax.scatter(
                    [mu[x_index]], [mu[y_index]], facecolor="r", s=5))


            # Show the values in the other windows.
            # TODO: we shouldn't be including MML terms in this function...
            I_mixture, I_parameters = mml._gmm_parameter_message_length(K, N, D)
            I_other = I_mixture + I_parameters

            I_slogw = I_other \
                    + (0.25 * D * (D + 3) - 0.5) * np.sum(np.log(params["weight"]))
            I_sldc = -0.5 * (D + 2) * np.sum(np.linalg.slogdet(params["cov"])[1])

            updates = [
                ("slogw", self._slogw_data, self._scatter_slogw_data, I_slogw),
                ("slogdet", self._slogdet_data, self._scatter_slogdet_data, I_sldc),
                ("nll", self._nll_data, self._scatter_nll_data, params["nll"]),
                ("I", self._I_data, self._scatter_I_data, params["I"])
            ]

            for descriptor, data, scatter, new_datum in updates:
                data.append([K, new_datum])

                da = np.array(data)
                scatter.set_offsets(da)
                scatter.set_facecolor("k")
                scatter.set_sizes(30 * np.ones(len(data)))
                scatter.set_zorder(100)




        elif kind == "maximization":

            self._clear_items(self._plot_model_items)

            ax = self._ax("data")
            x_index, y_index = self._data_projection_indices

            N, D = self._y_shape
            K = params["weight"].size

            cov_mask = np.zeros((D, D), dtype=bool)
            cov_mask[y_index, x_index] = True
            cov_mask[x_index, y_index] = True
            cov_mask[x_index, x_index] = True
            cov_mask[y_index, y_index] = True


            for k in range(K):
                mu = params["mean"][k][[x_index, y_index]]
                cov = params["cov"][k][cov_mask].reshape((2, 2))

                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals = vals[order]
                vecs = vecs[:, order]

                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

                # Show 2 standard deviations.
                width, height = 2 * 2 * np.sqrt(vals)
                ellipse = Ellipse(xy=mu, width=width, height=height, angle=theta,
                    facecolor="r", alpha=0.25)

                self._plot_model_items.append(ax.add_artist(ellipse))
                self._plot_model_items.append(ax.scatter(
                    [mu[x_index]], [mu[y_index]], facecolor="r", s=5))            


        elif kind == "I_slw_bounds":

            self._plot_prediction_items.setdefault(kind, [])
            self._clear_items(self._plot_prediction_items[kind])

            K, lower, upper = (params["K"], params["lower"], params["upper"])

            ax = self._ax("slogw")

            self._plot_prediction_items[kind].extend([
                ax.plot(K, lower, **self.__plot_bounds_kwds)[0],
            ])

        elif kind == "predict_I_slw":

            self._plot_prediction_items.setdefault(kind, [])
            self._clear_items(self._plot_prediction_items[kind])

            K, p_I_slw = params["K"], params["p_I_slw"]
            p_I_slw_err_pos, p_I_slw_err_neg = params["p_I_slw_err"]

            ax = self._ax("slogw")

            self._plot_prediction_items[kind].extend([
                ax.plot(K, p_I_slw, **self.__plot_prediction_kwds)[0],
                ax.fill_between(K, 
                                p_I_slw + p_I_slw_err_neg,
                                p_I_slw + p_I_slw_err_pos,
                                **self.__fill_between_prediction_kwds)

            ])


        elif kind == "actual_I_slw":

            K, I_slw = params["K"], params["I_slw"]

            self._slogw_data.append([K, I_slw])
            scat, data = (self._scatter_slogw_data, self._slogw_data)

            data = np.array(data)
            scat.set_offsets(data)
            scat.set_facecolor("k")
            scat.set_sizes(30 * np.ones(len(data)))
            scat.set_zorder(100)


        elif kind == "I_sldc_bounds":

            ax = self._ax("slogdet")

            self._plot_prediction_items.setdefault(kind, [])
            self._clear_items(self._plot_prediction_items[kind])

            K, lower, upper = (params["K"], params["lower"], params["upper"])

            self._plot_prediction_items[kind].extend([
                ax.plot(K, lower, **self.__plot_bounds_kwds)[0],
            ])


        elif kind == "actual_I_slogdetcovs":

            K, I_sldc = params["K"], params["I_slogdetcovs"]

            self._slogdet_data.append([K, I_sldc])
            scat, data = (self._scatter_slogdet_data, self._slogdet_data)

            data = np.array(data)
            scat.set_offsets(data)
            scat.set_facecolor("k")
            scat.set_sizes(30 * np.ones(len(data)))
            scat.set_zorder(100)

        elif kind == "predict_I_slogdetcovs":

            K, p_I_sldc = params["K"], params["p_I_slogdetcovs"]
            p_I_sldc_pos_err = params["p_I_slogdetcovs_pos_err"]
            p_I_sldc_neg_err = params["p_I_slogdetcovs_neg_err"]


            self._plot_prediction_items.setdefault(kind, [])
            self._clear_items(self._plot_prediction_items[kind])

            ax = self._ax("slogdet")

            self._plot_prediction_items[kind].extend([
                ax.plot(K, p_I_sldc, **self.__plot_prediction_kwds)[0],
                ax.fill_between(K, 
                                p_I_sldc + p_I_sldc_neg_err,
                                p_I_sldc + p_I_sldc_pos_err,
                                **self.__fill_between_prediction_kwds)

            ])


        elif kind == "prediction":

            self._clear_items(self._plot_prediction_items)

            pred_color = "b"

            pred_fill_kwds = dict(facecolor=pred_color, alpha=0.5, zorder=-1)
            pred_plot_kwds = dict(lw=2, c=pred_color, zorder=0)
            theory_fill_kwds = dict(facecolor="#cccccc", zorder=-10)
            theory_plot_kwds = dict(c="#666666", zorder=-5)

            # Show slogw bounds first.
            ax = self._ax("slogw")
            self._plot_prediction_items.extend([
                ax.plot(
                    params["K"], params["p_I_analytic"], 
                    **pred_plot_kwds)[0],
                ax.fill_between(
                    params["K"],
                    params["p_I_analytic"] + params["p_I_analytic_pos_err"],
                    params["p_I_analytic"] + params["p_I_analytic_neg_err"],
                    **pred_fill_kwds),
                ax.fill_between(
                    params["K"],
                    params["t_I_analytic_lower"],
                    params["t_I_analytic_upper"],
                    **theory_fill_kwds),
                ax.plot(
                    params["K"], params["t_I_analytic_lower"],
                    **theory_plot_kwds)[0],
                ax.plot(
                    params["K"], params["t_I_analytic_upper"],
                    **theory_plot_kwds)[0]
            ])
            _rescale_based_on_data(ax, params["K"], np.hstack(
                [params["t_I_analytic_upper"], params["t_I_analytic_lower"]]))


            # Show slogdetcov predictions.
            ax = self._ax("slogdet")
            self._plot_prediction_items.extend([
                ax.plot(
                    params["K"], params["p_I_slogdetcov"],
                    **pred_plot_kwds)[0],
                ax.fill_between(
                    params["K"],
                    params["p_I_slogdetcov"] + params["p_I_slogdetcov_pos_err"],
                    params["p_I_slogdetcov"] + params["p_I_slogdetcov_neg_err"],
                    **pred_fill_kwds),
                ax.fill_between(
                    params["K"],
                    params["t_I_slogdetcov_lower"],
                    params["t_I_slogdetcov_upper"],
                    **theory_fill_kwds),
                ax.plot(
                    params["K"], params["t_I_slogdetcov_lower"], 
                    **theory_plot_kwds)[0],
                ax.plot(
                    params["K"], params["t_I_slogdetcov_upper"],
                    linestyle=":", **theory_plot_kwds)[0]
                ])
            _rescale_based_on_data(ax, params["K"], 
                np.hstack([
                    params["t_I_slogdetcov_upper"], 
                    params["t_I_slogdetcov_lower"]
                ]))


            # Show nll predictions.
            ax = self._ax("nll")
            ylim_max = np.max(params["p_nll"])
            self._plot_prediction_items.extend([
                ax.plot(params["K"], params["p_nll"], **pred_plot_kwds)[0],
                ax.fill_between(params["K"], params["t_nll_lower"], ylim_max,
                    **theory_fill_kwds),
                ax.plot(params["K"], params["t_practical_nll_lower"],
                    linestyle=":", **theory_plot_kwds)[0],
                ax.plot(params["K"], params["t_nll_lower"], **theory_plot_kwds)[0]
            ])
            _rescale_based_on_data(ax, params["K"], 
                np.hstack([np.min(params["t_nll_lower"]), params["p_nll"]]))
            ax.set_ylim(ax.get_ylim()[0], ylim_max)

            ax = self._ax("I")
            self._plot_prediction_items.extend([
                ax.plot(params["K"], params["p_I"], **pred_plot_kwds)[0],
            ])
            # Show the saddle point.
            #if np.any(np.isfinite(params["p_I"])):
            #    index = np.nanargmin(params["p_I"])
            #    self._plot_prediction_items.append(
            #        ax.axvline(params["K"][index], lw=1, linestyle=":", c=color))

            _rescale_based_on_data(ax, params["K"], params["p_I"])
            
            ylim = ax.get_ylim()
            self._plot_prediction_items.extend([
                ax.fill_between(
                    params["K"], params["t_practical_I_lower"], ylim[1],
                    **theory_fill_kwds),
                ax.plot(
                    params["K"], params["t_practical_I_lower"],
                    linestyle=":", **theory_plot_kwds)[0]
            ])
            _rescale_based_on_data(
                ax, params["K"], [ylim[1], np.min(params["t_practical_I_lower"])])
            ax.set_ylim(ax.get_ylim()[0], ylim[1])


        else:
            logger.warn(f"Ignoring vizualiation event '{kind}'")
            return None
        
        if snapshot:
            self.snapshot()

        return None



    def create_movie(self, output_path, remove_snapshots=False):
        r"""
        Create a movie of the search progress and save it to `output_path`.

        :param output_path:
            A local path on disk to save the movie to.

        :param remove_snapshots: [optional]
            Remove the individual figures (snapshots) used to create the movie.
        """

        # TODO: DOCS

        os.system("ffmpeg -y -i \"{0}_iter_%05d.png\" {0}.m4v".format(
            self._figure_path))

        if remove_snapshots:
            os.system("rm -fv {0}_iter_*.png".format(self._figure_path))

        return True


def _rescale_based_on_data(ax, x, y, y_percent_edge=5):
    """
    Re-scale a figure axis based on the given data.

    :param ax:
        The figure axes.

    :param x:
        An array-like object of the x-values.

    :param y:
        An array-like object of the y-values.

    :param y_percent_edge: [optional]
        The percentage of the peak-to-peak y-range to show either end of the
        range of `y` values.
    """

    if np.sum(np.isfinite(np.unique(x))) > 1:
        ax.set_xlim(np.nanmin(x), np.nanmax(x))

    y_finite = np.array(y)[np.isfinite(y)]
    if y_finite.size > 1:
        y_ptp = np.ptp(y_finite)
        ax.set_ylim(
            np.min(y_finite) - y_percent_edge/100.0 * y_ptp,
            np.max(y_finite) + y_percent_edge/100.0 * y_ptp
        )

    return None