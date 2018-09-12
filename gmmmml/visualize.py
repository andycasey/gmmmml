
"""
Visualize the search progress for a gaussian mixture model.
"""

import os
import logging as logger
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator

from . import mml, utils


mpl_style = {
    "font.size": 12.0,
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.latex.preview": True,
    "axes.unicode_minus": False
}

matplotlib.style.use(mpl_style)

class VisualizationHandler(object):

    def __init__(self, y, data_projection_indices=(0, 1), figure_prefix=None,
        target=None, colours=None, show_intermediate_steps=True, **kwargs):
        r"""
        Initialize a visualisation handler to show the search progress and the
        predictions of future mixtures.

        :param y:
            The data :math:`y`.

        :param data_projection_indices: [optional]
            A two-length tuple containing the indices to use when plotting the
            :math:`x` and :math:`y` projections of the data.

        :param figure_prefix: [optional]
            The local path to store any figures generated during the search.

        :param target: [optional]
            The target message length components (e.g., for generated data).

        :param colours: [optional]
            A dictionary containing the colours for these items:

                data
                predictions
                bounds

        """

        self._y_shape = y.shape
        self._data_projection_indices = data_projection_indices

        if colours is None:
            colours = dict(bounds="#666666",
                           predictions="tab:blue",
                           model="tab:blue",
                           data="#000000")
        self._colours = colours


        self._figure_prefix = "" if figure_prefix is None else figure_prefix

        if not os.path.exists(os.path.dirname(self._figure_prefix)):
            os.makedirs(os.path.dirname(self._figure_prefix), exist_ok=True)

        self._figure_iter = 1
        self._figure_path_template \
            = "{0}_iter_{1:.0f}".format(self._figure_prefix, np.random.uniform(0,  10000))


        self._fig, axes = plt.subplots(1, 5, figsize=(15.5, 3))
        self._show_intermediate_steps = show_intermediate_steps

        # lists for plotting
        self._plot_items = {}

            
        self._init_axes(y, target)
        self.snapshot()



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
         
        self._actual_sum_log_weights = []
        ax = self._ax("sum_log_weights")
        ax.set_xlabel(r"$K$")
        ax.set_ylabel(r"$\left(\frac{D(D+3)}{4} - \frac{1}{2}\right)\sum\log{w_k} \textrm{ / nats}$")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        self._scatter_actual_sum_log_weights = ax.scatter([np.nan], [np.nan])

        ax = self._ax("sum_log_det_covs")
        ax.set_xlabel(r"$K$")
        ax.set_ylabel(r"$-\frac{(D+2)}{2}\sum\log{|C_k|} \textrm{ / nats}$")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        self._actual_sum_log_det_covs = []
        self._scatter_actual_sum_log_det_covs = ax.scatter([np.nan], [np.nan])

        ax = self._ax("negative_sum_log_likelihood")
        ax.set_xlabel(r"$K$")
        ax.set_ylabel(r"$-\sum\log{\mathcal{L}(y|\theta)} \textrm{ / nats}$")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        self._actual_negative_log_likelihoods = []
        self._scatter_actual_negative_log_likelihoods = ax.scatter([np.nan], [np.nan])

        ax = self._ax("I")
        ax.set_xlabel(r"$K$")
        ax.set_ylabel(r"$I \textrm{ / nats}$")
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))
        self._actual_message_lengths = []
        self._scatter_actual_message_lengths = ax.scatter([np.nan], [np.nan])


        for ax in self._fig.axes:
            ax.autoscale(enable=True)
            self._plot_items.setdefault(ax, [])

        self._fig.tight_layout()

        return None



    def _ax(self, descriptor):
        r"""
        Return the correct axes given a descriptor.
        """

        axes = np.array(self._fig.axes).flatten()
        index = ["data", "sum_log_weights", "sum_log_det_covs", 
                 "negative_sum_log_likelihood", "I"].index(descriptor)
        return axes[index]


    def snapshot(self, **kwargs):
        """
        Save a snapshot (figure) of the current progress and predictions.
        """

        plt.draw()
        self._fig.tight_layout()
        path = "{0:s}_{1:05d}.png".format(self._figure_path_template, self._figure_iter)
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


    def emit(self, kind, params, snapshot=False):
        r"""
        Handler for events.
        """
        
        if kind == "maximization":

            ax = self._ax("data")

            self._clear_items(self._plot_items[ax])
            
            x_index, y_index = self._data_projection_indices

            N, D = self._y_shape
            K = params["weights"].size

            model_colour = self._colours["model"]

            cov_mask = np.zeros((D, D), dtype=bool)
            cov_mask[y_index, x_index] = True
            cov_mask[x_index, y_index] = True
            cov_mask[x_index, x_index] = True
            cov_mask[y_index, y_index] = True

            for k in range(K):
                mu = params["means"][k][[x_index, y_index]]
                cov = params["covs"][k][cov_mask].reshape((2, 2))

                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals = vals[order]
                vecs = vecs[:, order]

                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

                # Show 2 standard deviations.
                width, height = 2 * 2 * np.sqrt(vals)
                ellipse = Ellipse(xy=mu, width=width, height=height, angle=theta,
                    facecolor=model_colour, alpha=0.3)

                self._plot_items[ax].append(ax.add_artist(ellipse))
                self._plot_items[ax].append(ax.scatter(
                    [mu[x_index]], [mu[y_index]], facecolor=model_colour, s=5))


        elif kind == "predict_I_weights":

            ax = self._ax("sum_log_weights")
            self._clear_items(self._plot_items[ax])
            
            K, I, I_var = (params["K"], params["I"], params["I_var"])
            I_lower, I_upper = (params["I_lower"], params["I_upper"])

            bound_colour = self._colours["bounds"]
            prediction_colour = self._colours["predictions"]

            self._plot_items[ax].extend([
                ax.plot(K, I, lw=2, c=prediction_colour)[0],
                ax.fill_between(K, I - np.sqrt(I_var), I + np.sqrt(I_var),
                                facecolor=prediction_colour, alpha=0.3, zorder=-1),
                ax.fill_between(K, I_lower, I_upper, facecolor=bound_colour,
                                alpha=0.3, zorder=-1),
                ax.plot(K, I_lower, lw=1, c=bound_colour)[0],
                ax.plot(K, I_upper, lw=1, c=bound_colour)[0]
            ])

        elif kind == "actual_I_weights":

            K, I = params["K"], params["I"]

            self._actual_sum_log_weights.append(np.hstack([K, I]))

            scat, data = (self._scatter_actual_sum_log_weights, self._actual_sum_log_weights)

            data = np.array(data)
            if not self._show_intermediate_steps:
                data = np.vstack(utils._best_mixture_parameter_values(
                    data.T[0], np.hstack(self._actual_message_lengths), data.T[1])).T
                #data = np.vstack(utils._group_over(data.T[0], data.T[1], np.min)).T

            scat.set_offsets(data)
            scat.set_facecolor(self._colours["data"])
            scat.set_sizes(30 * np.ones(len(data)))
            scat.set_zorder(100)

            #_rescale_based_on_data(self._ax("sum_log_weights"), *data.T)
            


        elif kind == "actual_I_slogdetcovs":


            K, I = params["K"], params["I"]

            self._actual_sum_log_det_covs.append(np.hstack([K, I]))

            scat, data = (self._scatter_actual_sum_log_det_covs, self._actual_sum_log_det_covs)
            data = np.array(data)

            if not self._show_intermediate_steps:
                data = np.vstack(utils._best_mixture_parameter_values(
                    data.T[0], np.hstack(self._actual_message_lengths), data.T[1])).T

                #data = np.vstack(utils._group_over(data.T[0], data.T[1], np.min)).T

            scat.set_offsets(data)
            scat.set_facecolor(self._colours["data"])
            scat.set_sizes(30 * np.ones(len(data)))
            scat.set_zorder(100)

            #_rescale_based_on_data(self._ax("sum_log_det_covs"), *data.T)


        elif kind == "predict_I_slogdetcovs":

            ax = self._ax("sum_log_det_covs")
            self._clear_items(self._plot_items[ax])
            
            K, I, I_var = (params["K"], params["I"], params["I_var"])
            I_lower, I_upper = (params["I_lower"], params["I_upper"])

            prediction_colour = self._colours["predictions"]

            self._plot_items[ax].extend([
                ax.plot(K, I, lw=2, c=prediction_colour)[0],
                ax.fill_between(K, I - np.sqrt(I_var), I + np.sqrt(I_var),
                                facecolor=prediction_colour, alpha=0.3, zorder=-1),
            ])

            bound_colour = self._colours["bounds"]

            v = np.vstack([
                np.vstack([K, I + np.sqrt(I_var)]).T,
                np.array(self._actual_sum_log_det_covs)
            ])

            _rescale_based_on_data(ax, *v.T)


            # Adjust K so that it extends the region we want.
            if K.size > 1:
                Ks = np.hstack([K[0] - 0.5, K[1:-1], K[-1] + 0.5])
            else:
                Ks = K

            set_upper = not np.all(np.isfinite(I_upper))

            if set_upper:
                plt.draw()
                ylim = ax.get_ylim()
                I_upper = np.max(ylim) * np.ones_like(I_lower)


            self._plot_items[ax].extend([
                ax.plot(K, I_lower, c=bound_colour, lw=1)[0],
                ax.plot(K, I_upper, c=bound_colour, lw=1)[0],
                ax.fill_between(Ks, 
                                I_lower, 
                                I_upper,
                                facecolor=bound_colour, alpha=0.3,
                                zorder=-1)
            ])

            if set_upper:
                ax.set_ylim(ylim)
            

        elif kind == "actual_I_data":


            K, I = params["K"], params["I"]

            self._actual_negative_log_likelihoods.append(np.hstack([K, I]))

            scat = self._scatter_actual_negative_log_likelihoods
            data = np.array(self._actual_negative_log_likelihoods)

            if not self._show_intermediate_steps:
                data = np.vstack(utils._best_mixture_parameter_values(
                    data.T[0], np.hstack(self._actual_message_lengths), data.T[1])).T

                #data = np.vstack(utils._group_over(data.T[0], data.T[1], np.min)).T

            scat.set_offsets(data)
            scat.set_facecolor(self._colours["data"])
            scat.set_sizes(30 * np.ones(len(data)))
            scat.set_zorder(100)

            #_rescale_based_on_data(self._ax("negative_sum_log_likelihood"), *data.T)


        elif kind == "predict_I_data":

            ax = self._ax("negative_sum_log_likelihood")
            self._clear_items(self._plot_items[ax])
            
            K, I, I_var = (params["K"], params["I"], params["I_var"])
            I_lower = params["I_lower"]
            I_upper = np.nan

            bound_colour = self._colours["bounds"]
            prediction_colour = self._colours["predictions"]

            self._plot_items[ax].extend([
                ax.plot(K, I, lw=2, c=prediction_colour)[0],
                ax.fill_between(K, I - np.sqrt(I_var), I + np.sqrt(I_var),
                                facecolor=prediction_colour, alpha=0.3, zorder=-1),
            ])

            v = np.vstack([
                np.vstack([K, I_lower]).T,
                np.array(self._actual_negative_log_likelihoods)
            ])

            _rescale_based_on_data(ax, *v.T)


            # Adjust K so that it extends the region we want.
            if K.size > 1:
                Ks = np.hstack([K[0] - 0.5, K[1:-1], K[-1] + 0.5])
            else:
                Ks = K

            set_upper = not np.all(np.isfinite(I_upper))

            if set_upper:
                plt.draw()
                ylim = ax.get_ylim()
                I_upper = np.max(ylim) * np.ones_like(I_lower)

            self._plot_items[ax].extend([
                ax.plot(K, I_lower, c=bound_colour, lw=1)[0],
                ax.plot(K, I_upper, c=bound_colour, lw=1)[0],
                ax.fill_between(Ks, 
                                I_lower, 
                                I_upper,
                                facecolor=bound_colour, alpha=0.3,
                                zorder=-1)
            ])

            if set_upper:
                ax.set_ylim(ylim)



        elif kind == "actual_I":

            K, I = params["K"], params["I"]

            self._actual_message_lengths.append(np.hstack([K, I]))
            scat = self._scatter_actual_message_lengths
            data = np.array(self._actual_message_lengths)

            if not self._show_intermediate_steps:
                data = np.vstack(utils._best_mixture_parameter_values(
                    data.T[0], np.hstack(self._actual_message_lengths), data.T[1])).T

                #data = np.vstack(utils._group_over(data.T[0], data.T[1], np.min)).T

            scat.set_offsets(data)
            scat.set_facecolor(self._colours["data"])
            scat.set_sizes(30 * np.ones(len(data)))
            scat.set_zorder(100)


        elif kind == "predict_I":

            ax = self._ax("I")
            self._clear_items(self._plot_items[ax])
            
            K, I, I_var = (params["K"], params["I"], params["I_var"])
            I_lower = params["I_lower"]
            I_upper = np.nan

            bound_colour = self._colours["bounds"]
            prediction_colour = self._colours["predictions"]

            self._plot_items[ax].extend([
                ax.plot(K, I, lw=2, c=prediction_colour)[0],
                ax.fill_between(K, I - np.sqrt(I_var), I + np.sqrt(I_var),
                                facecolor=prediction_colour, alpha=0.3, zorder=-1),
            ])


            v = np.vstack([
                np.vstack([K, I_lower]).T,
                np.array(self._actual_message_lengths)
            ])

            _rescale_based_on_data(ax, *v.T)


            # Adjust K so that it extends the region we want.
            if K.size > 1:
                Ks = np.hstack([K[0] - 0.5, K[1:-1], K[-1] + 0.5])
            else:
                Ks = K

            set_upper = not np.all(np.isfinite(I_upper))

            if set_upper:
                plt.draw()
                ylim = ax.get_ylim()
                I_upper = np.max(ylim) * np.ones_like(I_lower)

            self._plot_items[ax].extend([
                ax.plot(K, I_lower, c=bound_colour, lw=1)[0],
                ax.plot(K, I_upper, c=bound_colour, lw=1)[0],
                ax.fill_between(Ks, 
                                I_lower, 
                                I_upper,
                                facecolor=bound_colour, alpha=0.3,
                                zorder=-1)
            ])

            if set_upper:
                ax.set_ylim(ylim)



        elif kind is None:
            None

        else:
            logger.warn(f"Ignoring vizualiation event '{kind}'")
            return None
        
        if snapshot:
            self.snapshot()

        return None





    def create_movie(self, remove_snapshots=False):
        r"""
        Create a movie of the search progress and save it to `output_path`.

        :param output_path:
            A local path on disk to save the movie to.

        :param remove_snapshots: [optional]
            Remove the individual figures (snapshots) used to create the movie.
        """

        # TODO: DOCS

        os.system("ffmpeg -y -i \"{0}_%05d.png\" {0}.m4v".format(
            self._figure_path_template))

        if remove_snapshots:
            os.system("rm -fv {0}_*.png".format(self._figure_path_template))

        return True


def _rescale_based_on_data(ax, x=None, y=None, y_percent_edge=5):
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

    if x is not None:
        if np.sum(np.isfinite(np.unique(x))) > 1:
            ax.set_xlim(np.nanmin(x) - 0.5, np.nanmax(x) + 0.5)

    if y is not None:
        y_finite = np.array(y)[np.isfinite(y)]
        if y_finite.size > 1:
            y_ptp = np.ptp(y_finite)
            ax.set_ylim(
                np.min(y_finite) - y_percent_edge/100.0 * y_ptp,
                np.max(y_finite) + y_percent_edge/100.0 * y_ptp
            )

    return None