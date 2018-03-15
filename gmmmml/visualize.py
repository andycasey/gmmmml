

import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse




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

        self._reference_ll = None


        self.fig, axes = plt.subplots(1, 6, figsize=(15.6 + 2.8, 2.8))
        axes = np.array(axes).flatten()
        self._display = True

        self.ax_data, self.ax_slogw, self.ax_slogdet, self.ax_sll, self.ax_I_other, self.ax_I = axes

        self._xyindex = (x_index, y_index)

        self.ax_data.scatter(y.T[x_index], y.T[y_index], facecolor="k", s=1, alpha=0.5)

        self.ax_data.set_xlabel(r"$x_{{{0}}}$".format(x_index))
        self.ax_data.set_ylabel(r"$x_{{{0}}}$".format(y_index))

        self.ax_I.set_xlabel(r"$K$")
        self.ax_I.set_ylabel(r"$I$ $[{\rm nats}]$")


        self.ax_slogw.set_xlabel(r"$K$")
        self.ax_slogw.set_ylabel(r"$\left(\frac{D(D+3)}{4} - \frac{1}{2}\right)\sum\log{w_k}$ $[{\rm nats}]$")

        self.ax_I_other.set_xlabel(r"$K$")
        self.ax_I_other.set_ylabel(r"$I_{other}$ $[{\rm nats}]$")

        self.ax_slogdet.set_xlabel(r"$K$")
        self.ax_slogdet.set_ylabel(r"$-\frac{(D+2)}{2}\sum\log{|C_k|}$ $[{\rm nats}]$")


        self.ax_sll.set_xlabel(r"$K$")
        self.ax_sll.set_ylabel(r"$\sum\log{\mathcal{L}(y\|\theta)}$ $[{\rm nats}]$")

        if target is not None:
            K_target = target["weight"].size
                
            target_kwds = dict(facecolor=self._color_target, s=50, alpha=0.5)

            self.ax_slogw.scatter(
                [K_target], [np.sum(np.log(target["weight"]))], **target_kwds)

            self.ax_slogdet.scatter(
                [K_target],
                [np.sum(np.linalg.slogdet(target["cov"])[1])],
                **target_kwds)

            self.ax_sll.scatter(
                [K_target],
                [-target["nll"]],
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
            slogdet_cov = np.sum(np.log(np.linalg.det(params["cov"])))
            log_mean_det_cov = np.log(np.mean(np.linalg.det(params["cov"])))

            self.ax_slogdet.scatter([K], [slogdet_cov], facecolor="k", s=1)

            sum_log_weights = np.sum(np.log(params["weight"]))
            self.ax_slogw.scatter([K], [sum_log_weights], facecolor="k", alpha=0.5, s=1)


        elif kind == "predict_I_other":

            K = params["K"]
            I_other = params["I_other"]

            kwds = dict(c='k', lw=2)
            kwds.update(plotting_kwds)

            self.ax_I_other.plot(K, I_other, **kwds)

        elif kind == "expectation":
            self.ax_I.scatter(
                [params["K"]], [params["message_length"]],
                facecolor="k", s=1)
            self._expectation_iter += 1

            # plot LL as well
            K = params["K"]
            ll = np.sum(params["log_likelihood"])

            if self._reference_ll is None:
                self._reference_ll = ll

            # /self._reference_ll
            self.ax_sll.scatter([K], [ll], facecolor="k", s=1)





        elif kind == "predict_slw":

            self._update_previous_predict_slws()

            K = params["K"]
            p_slw = params["p_slw"]
            p_slw_err = params["p_slw_err"]
            p_slw_max = params["p_slw_max"]
            p_slw_min = params["p_slw_min"]

            self._predict_slw.extend([
                self.ax_slogw.plot(K, p_slw, 
                    c=self._color_prediction, zorder=-1)[0],
                self.ax_slogw.fill_between(
                    K, p_slw_err[0] + p_slw, p_slw_err[1] + p_slw, 
                    facecolor=self._color_prediction, alpha=0.5, zorder=-1),
                self.ax_slogw.fill_between(K, p_slw_min, p_slw_max,
                    facecolor="#666666", alpha=0.5, zorder=-1)
            ])


        elif kind == "predict_ll":

            if clear_previous:
                self._update_previous_predict_lls()

            K = params["K"]
            p_ll = params["p_ll"]#/self._reference_ll
            p_ll_err = params.get("p_ll_err", None)#/self._reference_ll

            kwds = dict(c=self._color_prediction, zorder=-1)
            kwds.update(plotting_kwds)

            self._predict_ll.extend(self.ax_sll.plot(K, p_ll, **kwds))

            #    self.ax_sll.fill_between(
            #        K, p_ll_err[0] + p_ll, p_ll_err[1] + p_ll,
            #        facecolor=self._color_prediction, alpha=0.5, zorder=-1)
            #    ])

            self.ax_sll.relim()
            self.ax_sll.autoscale_view()



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

            self.ax_slogdet.relim()
            self.ax_slogdet.autoscale_view()

        elif kind == "predict_message_length":

            self._update_previous_predict_message_lengths()

            K = params["K"]
            p_I = params["p_I"]

            #self._predict_message_lengths.extend([
            #    self.ax_I.plot(K, p_I, c='b', alpha=0.5)[0]
            #])

            self.ax_I.relim()
            self.ax_I.autoscale_view()


        elif kind == "show_mml":

            K = params["K"]
            I = params["I"]
            I_parts = params["I_parts"]

            self.ax_I.plot(K, I, c='g')


        else:
            raise ValueError("what you tryin' to tell me?!")

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