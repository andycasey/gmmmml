

import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse




class VisualizationHandler(object):

    def __init__(self, y, target=None, figure_path=None, **kwargs):

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


        self.fig, self.axes = plt.subplots(3,3)
        self.axes = np.array(self.axes).flatten()
        self._display = True

        self.axes[0].scatter(y.T[0], y.T[1], facecolor="k", alpha=0.5)

        self.axes[0].set_xlabel("Data X")
        self.axes[0].set_ylabel("Data Y")

        self.axes[1].set_xlabel("E-M iteration")
        self.axes[1].set_ylabel("I_actual")

        self.axes[2].set_xlabel("K")
        self.axes[2].set_ylabel("I_predicted")


        self.axes[3].set_xlabel("K")
        self.axes[3].set_ylabel(r"$\sum\log{|C|}$")

        self.axes[4].set_xlabel("K")
        self.axes[4].set_ylabel(r"$\log{}<|C|>$")


        self.axes[5].set_xlabel("K")
        self.axes[5].set_ylabel(r"$\sum\log{w}$")

        self.axes[6].set_xlabel("K")
        self.axes[6].set_ylabel(r"$\sum\log{L}$")

        if target is not None:
            K_target = target["weight"].size
                
            target_kwds = dict(facecolor=self._color_target, s=50, alpha=0.5)

            self.axes[1].scatter([K_target], [target["I"]], **target_kwds)

            self.axes[5].scatter(
                [K_target], [np.sum(np.log(target["weight"]))], **target_kwds)

            self.axes[3].scatter(
                [K_target],
                [np.sum(np.linalg.slogdet(target["cov"])[1])],
                **target_kwds)

            self.axes[6].scatter(
                [K_target],
                [-target["nll"]],
                **target_kwds)

        for ax in self.axes:
            ax.autoscale(enable=True)

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




    def emit(self, kind, params, save=True):

        
        if kind == "model":

            self._clear_model()

            # Update view of the model.
            K = params["weight"].size

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
                    facecolor="r", alpha=0.5)

                self._model.append(self.axes[0].add_artist(ellip))
                self._model.append(self.axes[0].scatter([mean[0]], [mean[1]], facecolor="r"))

            
            K = params["weight"].size
            slogdet_cov = np.sum(np.log(np.linalg.det(params["cov"])))
            log_mean_det_cov = np.log(np.mean(np.linalg.det(params["cov"])))

            self.axes[3].scatter([K], [slogdet_cov], facecolor="k")

            self.axes[4].scatter([K], [log_mean_det_cov], facecolor="k", alpha=0.5)

            sum_log_weights = np.sum(np.log(params["weight"]))
            self.axes[5].scatter([K], [sum_log_weights], facecolor="k", alpha=0.5)


        elif kind == "expectation":
            self.axes[1].scatter(
                [self._expectation_iter], [params["message_length"]],
                facecolor="k")
            self._expectation_iter += 1

            # plot LL as well
            K = params["K"]
            ll = np.sum(params["log_likelihood"])

            if self._reference_ll is None:
                self._reference_ll = ll

            # /self._reference_ll
            self.axes[6].scatter([K], [ll], facecolor="k")





        elif kind == "predict_slw":

            self._update_previous_predict_slws()

            K = params["K"]
            p_slw = params["p_slw"]
            p_slw_err = params["p_slw_err"]
            p_slw_max = params["p_slw_max"]

            self._predict_slw.extend([
                self.axes[5].plot(K, p_slw, 
                    c=self._color_prediction, zorder=-1)[0],
                self.axes[5].fill_between(
                    K, p_slw_err[0] + p_slw, p_slw_err[1] + p_slw, 
                    facecolor=self._color_prediction, alpha=0.5, zorder=-1),
                self.axes[5].plot(K, p_slw_max,
                    c=self._color_prediction, linestyle="--", zorder=-1)[0]
            ])


        elif kind == "predict_ll":

            self._update_previous_predict_lls()

            K = params["K"]
            p_ll = params["p_ll"]#/self._reference_ll
            p_ll_err = params["p_ll_err"]#/self._reference_ll

            self._predict_ll.extend([
                self.axes[6].plot(K, p_ll,
                    c=self._color_prediction, zorder=-1)[0]
            ])
            #    self.axes[6].fill_between(
            #        K, p_ll_err[0] + p_ll, p_ll_err[1] + p_ll,
            #        facecolor=self._color_prediction, alpha=0.5, zorder=-1)
            #    ])


        elif kind == "predict_slogdetcov":

            self._update_previous_predict_slogdetcovs()

            K = params["K"]
            p_slogdetcovs = params["p_slogdetcovs"]
            p_slogdetcovs_pos_err = params["p_slogdetcovs_pos_err"]
            p_slogdetcovs_neg_err = params["p_slogdetcovs_neg_err"]

            self._predict_slogdetcovs.extend([
                self.axes[3].plot(K, p_slogdetcovs, c=self._color_prediction)[0],
                self.axes[3].fill_between(K,
                    p_slogdetcovs + p_slogdetcovs_neg_err,
                    p_slogdetcovs + p_slogdetcovs_pos_err,
                    facecolor=self._color_prediction, alpha=0.5, zorder=-1)
            ])

            self.axes[3].relim()
            self.axes[3].autoscale_view()

        elif kind == "predict_message_length":

            self._update_previous_predict_message_lengths()

            K = params["K"]
            p_I = params["p_I"]

            self._predict_message_lengths.extend([
                self.axes[1].plot(K, p_I, c='b', alpha=0.5)[0]
            ])

            self.axes[1].relim()
            self.axes[1].autoscale_view()


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