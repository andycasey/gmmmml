

import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator



class VisualizationHandler(object):

    def __init__(self, y, target=None, x_index=0, y_index=1,
        figure_path=None, **kwargs):

        figure_path = "" if figure_path is None else figure_path

        self._color_model = "r"
        self._color_prediction = "b"
        self._color_target = "g"

        self._model = []
        self._expectation_iter = 1
        self._figure_iter = 1
        self._figure_prefix = os.path.join(
            figure_path, "iter_{}".format(int(np.random.uniform(0, 1000))))

        self._predict_slw = []
        self._predict_ll = []
        self._predict_slogdetcovs = []
        self._predict_message_lengths = []
        self._predict_slw_bounds = []
        self._predict_slogdetcov_bounds = []
        self._predict_ll_bounds = []

        self._data_slw = []
        self._data_slogdetcov = []

        self._reference_ll = None


        self.fig, axes = plt.subplots(1, 6, figsize=(15.6 + 2.8, 2.8))
        axes = np.array(axes).flatten()
        self._display = True

        self.ax_data, self.ax_I_other, self.ax_slogw, self.ax_slogdet, self.ax_sll, self.ax_I = axes

        self._xyindex = (x_index, y_index)

        self.ax_data.scatter(y.T[x_index], y.T[y_index], facecolor="k", s=1, alpha=0.5)
        self.ax_data.set_xlabel(r"$x_{{{0}}}$".format(x_index))
        self.ax_data.set_ylabel(r"$x_{{{0}}}$".format(y_index))
        self.ax_data.xaxis.set_major_locator(MaxNLocator(2))
        self.ax_data.yaxis.set_major_locator(MaxNLocator(2))


        self.ax_I.set_xlabel(r"$K$")
        self.ax_I.set_ylabel(r"$I$ $[{\rm nats}]$")
        self.ax_I.xaxis.set_major_locator(MaxNLocator(5))
        self.ax_I.yaxis.set_major_locator(MaxNLocator(5))

        self.ax_slogw.set_xlabel(r"$K$")
        self.ax_slogw.set_ylabel(r"$\left(\frac{D(D+3)}{4} - \frac{1}{2}\right)\sum\log{w_k}$ $[{\rm nats}]$")
        self.ax_slogw.xaxis.set_major_locator(MaxNLocator(5))
        self.ax_slogw.yaxis.set_major_locator(MaxNLocator(5))

        self.ax_I_other.set_xlabel(r"$K$")
        self.ax_I_other.set_ylabel(r"$I_{other}$ $[{\rm nats}]$")
        self.ax_I_other.xaxis.set_major_locator(MaxNLocator(5))
        self.ax_I_other.yaxis.set_major_locator(MaxNLocator(5))
        self._show_I_other_data = self.ax_I_other.plot([np.nan], [np.nan], c="#666666")[0]

        self.ax_slogdet.set_xlabel(r"$K$")
        self.ax_slogdet.set_ylabel(r"$-\frac{(D+2)}{2}\sum\log{|C_k|}$ $[{\rm nats}]$")
        self.ax_slogdet.xaxis.set_major_locator(MaxNLocator(5))
        self.ax_slogdet.yaxis.set_major_locator(MaxNLocator(5))

        #x = np.arange(1, 50)
        #self.ax_slogdet.plot(x, -0.5 * (y.shape[1] + 2) * x * 124.55, c="r")

        self._show_slw_data = self.ax_slogw.scatter(
            [np.nan], [np.nan], facecolor="k", s=5)
        self._show_slogdetcov_data = self.ax_slogdet.scatter([np.nan], [np.nan], facecolor="k", s=5)

        self.ax_sll.set_xlabel(r"$K$")
        self.ax_sll.set_ylabel(r"$-\sum\log{\mathcal{L}(y\|\theta)}$ $[{\rm nats}]$")
        self.ax_sll.xaxis.set_major_locator(MaxNLocator(5))
        self.ax_sll.yaxis.set_major_locator(MaxNLocator(5))

        if target is not None:
            K_target = target["weight"].size
                
            target_kwds = dict(facecolor=self._color_target, s=5, zorder=100)


            D = target["mean"].shape[1]
            self.ax_slogw.scatter(
                [K_target], [(0.25 * D * (D + 3) - 0.5) * np.sum(np.log(target["weight"]))], **target_kwds)

            self.ax_slogdet.scatter(
                [K_target],
                [-0.5 * (D + 2) * np.sum(np.linalg.slogdet(target["cov"])[1])],
                **target_kwds)

            self.ax_sll.scatter(
                [K_target],
                [target["nll"]],
                **target_kwds)

        for ax in axes:
            ax.autoscale(enable=True)

        self.fig.tight_layout()

        self.savefig()


    def _clear_model(self):

        L = len(self._model)
        for l in range(L):
            item = self._model.pop(0)
            item.set_visible(False)
            del item


    def _update_previous_predict_slws(self):
        L = len(self._predict_slw)
        for l in range(L):
            item = self._predict_slw.pop(0)
            
            # TODO: delete or just change/color etc.
            #item.set_alpha(0.1)

            item.set_visible(False)
            del item


    def _clear_previous_items(self, items):

        L = len(items)
        for l in range(L):
            item = items.pop(0)
            try:
                item.set_data([], [])
            except AttributeError:
                None
            item.set_visible(False)
            del item


    def _update_previous_predict_lls(self):
        L = len(self._predict_ll)
        for l in range(L):
            item = self._predict_ll.pop(0)
            try:
                item.set_data([], [])
            except AttributeError:
                None
            item.set_visible(False)
            del item


    def _update_previous_predict_slogdetcovs(self):
        L = len(self._predict_slogdetcovs)
        for l in range(L):
            item = self._predict_slogdetcovs.pop(0)
            try:
                item.set_data([], [])
            except AttributeError:
                None
            item.set_visible(False)
            del item

    def _update_previous_predict_message_lengths(self):
        L = len(self._predict_message_lengths)
        for l in range(L):
            item  = self._predict_message_lengths.pop(0)
            try:
                item.set_data([], [])
            except AttributeError:
                None
            item.set_visible(False)
            del item




    def emit(self, kind, params, save=True, clear_previous=True,
        plotting_kwds=None):

        plotting_kwds = dict() if plotting_kwds is None else plotting_kwds

        if kind == "model":

            self._clear_model()

            # Update view of the model.
            K = params["weight"].size

            x_index, y_index = self._xyindex
            if x_index != 0 or y_index != 1:
                raise NotImplementedError

            for k in range(K):
                mean = params["mean"][k][:2]
                cov = params["cov"][k]

                vals, vecs = np.linalg.eigh(cov[:2, :2])
                order = vals.argsort()[::-1]
                vals = vals[order]
                vecs = vecs[:,order]

                theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

                # Show 2 standard deviations
                width, height = 2 * 2 * np.sqrt(vals)
                ellip = Ellipse(xy=mean, width=width, height=height, angle=theta,
                    facecolor="r", alpha=0.1)

                self._model.append(self.ax_data.add_artist(ellip))
                self._model.append(self.ax_data.scatter([mean[0]], [mean[1]], facecolor="r", s=1))

            
            K = params["weight"].size
            D = params["mean"].shape[1]
            slogdet_cov = - 0.5 * (D + 2) * np.sum(np.log(np.linalg.det(params["cov"])))
            log_mean_det_cov = np.log(np.mean(np.linalg.det(params["cov"])))

            self._data_slw.append([K, 
                (0.25 * D * (D + 3) - 0.5) * np.sum(np.log(params["weight"]))])
            self._show_slw_data.set_offsets(np.array(self._data_slw))
            self._show_slw_data.set_facecolor("k") # Needed to make them update.
            self._show_slw_data.set_sizes(5 * np.ones(len(self._data_slw)))

            self._data_slogdetcov.append([K, slogdet_cov])
            self._show_slogdetcov_data.set_offsets(np.array(self._data_slogdetcov))
            self._show_slogdetcov_data.set_facecolor("k") # Needed to make them update.
            self._show_slogdetcov_data.set_sizes(5 * np.ones(len(self._data_slogdetcov)))

            #self.ax_slogdet.scatter([K], [slogdet_cov], facecolor="k", s=1)

            #sum_log_weights = np.sum(np.log(params["weight"]))
            #self.ax_slogw.scatter([K], [sum_log_weights], facecolor="k", s=1)


        elif kind == "predict_I_other":

            K = params["K"]
            I_other = params["I_other"]

            self._show_I_other_data.set_data(np.array([K, I_other]))
            self.ax_I_other.set_xlim(0, max(K))
            self.ax_I_other.set_ylim(0, max(I_other))


        elif kind == "expectation":
            self.ax_I.scatter(
                [params["K"]], [params["message_length"]],
                facecolor="k", s=5)
            self._expectation_iter += 1

            # plot LL as well
            K = params["K"]
            ll = np.sum(params["log_likelihood"])

            if self._reference_ll is None:
                self._reference_ll = ll

            # /self._reference_ll
            self.ax_sll.scatter([K], [-ll], facecolor="k", s=5)





        elif kind == "predict_slw":

            self._update_previous_predict_slws()

            K = params["K"]
            D = params["D"]
            p_slw = (0.25 * D * (D + 3) - 0.5) * params["p_slw"]
            p_slw_err = (0.25 * D * (D + 3) - 0.5) * params["p_slw_err"]
            
            self._predict_slw.extend([
                self.ax_slogw.plot(K, p_slw, 
                    c=self._color_prediction, zorder=-1)[0],
                self.ax_slogw.fill_between(
                    K, p_slw_err[0] + p_slw, p_slw_err[1] + p_slw, 
                    facecolor=self._color_prediction, alpha=0.5, zorder=-1),
            ])

            self.ax_slogw.set_xlim(0, max(K))

        elif kind == "slw_bounds":

            self._clear_previous_items(self._predict_slw_bounds)

            K, lower, upper = (params["K"], params["lower"], params["upper"])
            D = params["D"]
            lower = (0.25 * D * (D + 3) - 0.5) * lower
            upper = (0.25 * D * (D + 3) - 0.5) * upper
            plot_kwds = dict(c="#666666", linestyle="-", zorder=-10,
                linewidth=0.5)

            self._predict_slw_bounds.extend([
                self.ax_slogw.fill_between(K, lower, upper,
                    facecolor="#EEEEEE", zorder=-100, linestyle=":"),
                self.ax_slogw.plot(K, lower, **plot_kwds)[0],
                self.ax_slogw.plot(K, upper, **plot_kwds)[0]
            ])

        elif kind == "slogdetcov_bounds":

            self._clear_previous_items(self._predict_slogdetcov_bounds)

            K, lower, upper = (params["K"], params["lower"], params["upper"])

            bounds = np.array([lower, upper])
            lower = np.min(bounds, axis=0)
            upper = np.max(bounds, axis=0)


            plot_kwds = dict(c="#666666", linestyle="-", zorder=-10,
                linewidth=0.5)
            upper_plot_kwds = plot_kwds.copy()
            upper_plot_kwds["linestyle"] = "-."
            self._predict_slogdetcov_bounds.extend([
                self.ax_slogdet.fill_between(K, lower, upper,
                    facecolor="#EEEEEE", zorder=-100, linestyle=":"),
                self.ax_slogdet.plot(K, lower, **plot_kwds)[0],
                self.ax_slogdet.plot(K, upper, **upper_plot_kwds)[0]
            ])

        elif kind == "predict_ll":

            if clear_previous:
                self._update_previous_predict_lls()

            K = params["K"]
            p_ll = params["p_ll"]#/self._reference_ll
            p_ll_pos_err = params.get("p_ll_pos_err", 0)#/self._reference_ll
            p_ll_neg_err = params.get("p_ll_neg_err", 0)

            kwds = dict(c=self._color_prediction, zorder=-1, alpha=0.5)
            kwds.update(plotting_kwds)

            self._predict_ll.extend([
                self.ax_sll.plot(K, -p_ll, **kwds)[0],
                self.ax_sll.fill_between(
                    K, -(p_ll + p_ll_pos_err), -(p_ll + p_ll_neg_err),
                    facecolor=self._color_prediction, alpha=0.25, zorder=-1)
            ])

            #    self.ax_sll.fill_between(
            #        K, p_ll_err[0] + p_ll, p_ll_err[1] + p_ll,
            #        facecolor=self._color_prediction, alpha=0.5, zorder=-1)
            #    ])

            self.ax_sll.set_xlim(0, max(K))



        elif kind == "predict_ll_bounds":

            K = params["K"]
            likely_lower_bound = params["likely_upper_bound"]

            upper_bound = self.ax_sll.get_ylim()[1] * np.ones_like(K)

            if clear_previous:
                self._clear_previous_items(self._predict_ll_bounds)

            self._predict_ll_bounds.extend([
                self.ax_sll.fill_between(K, -likely_lower_bound, upper_bound,
                    facecolor="#EEEEEE", zorder=-100, linestyle=":"),
                self.ax_sll.plot(K, -likely_lower_bound, c="#666666", linestyle="-.",
                    zorder=-10, linewidth=0.5)[0]
                ])

            self.ax_sll.set_xlim(0, max(K))
            self.ax_sll.set_ylim(-likely_lower_bound[0], upper_bound[0])
            

        elif kind == "predict_slogdetcov":

            self._update_previous_predict_slogdetcovs()

            K = params["K"]
            p_slogdetcovs = params["p_slogdetcovs"]
            p_slogdetcovs_pos_err = params["p_slogdetcovs_pos_err"]
            p_slogdetcovs_neg_err = params["p_slogdetcovs_neg_err"]

            self._predict_slogdetcovs.extend([
                self.ax_slogdet.plot(K, p_slogdetcovs, c=self._color_prediction)[0],
                self.ax_slogdet.fill_between(K,
                    p_slogdetcovs + p_slogdetcovs_neg_err,
                    p_slogdetcovs + p_slogdetcovs_pos_err,
                    facecolor=self._color_prediction, alpha=0.5, zorder=-1)
            ])

            self.ax_slogdet.autoscale_view()
            self.ax_slogdet.relim()
            plt.draw()
            
            self.ax_slogdet.set_xlim(0, max(K))
            # Only show y-limits of the data.
            y = np.array(self._data_slogdetcov).T[1]
            ylimits = np.hstack([min(y), max(y),
                min(p_slogdetcovs + p_slogdetcovs_neg_err),
                max(p_slogdetcovs + p_slogdetcovs_pos_err),
            ]).flatten()
            ylimits = (np.min(ylimits), max(ylimits))

            frac = 0.05 * np.ptp(ylimits) 
            #self.ax_slogdet.set_ylim(min(ylimits), max(ylimits))
            
            self.ax_slogdet.set_ylim(ylimits[0] - frac, ylimits[1] + frac)


            #if min(K) > 8:
            #    raise a

        elif kind == "predict_message_length":

            self._update_previous_predict_message_lengths()

            K = params["K"]
            p_I = params["p_I"]

            self._predict_message_lengths.extend([
                self.ax_I.plot(K, p_I, c='b', alpha=0.5)[0]
            ])

            #self.ax_I.relim()
            #self.ax_I.autoscale_view()



        else:
            print("Ignoring emit kind: {}".format(kind))
            #raise ValueError("what you tryin' to tell me?!")

        # Only save on model update.
        if save:
            self.savefig()

        return None


    def savefig(self):
        plt.draw()
        self.fig.tight_layout()
        path = "{0:s}_{1:05d}.png".format(self._figure_prefix, self._figure_iter)
        self.fig.savefig(path)
        print("Created {}".format(path))
        self._figure_iter += 1


    def create_movie(self, cleanup=True):

        os.system('ffmpeg -y -i "{}_%05d.png" output.m4v'.format(self._figure_prefix))

        if cleanup:
            os.system("rm -fv {}_*.png".format(self._figure_prefix))