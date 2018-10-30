

import numpy as np
import pickle
import matplotlib
import matplotlib.pyplot as plt

from mpl_utils import mpl_style

matplotlib.style.use(mpl_style)

with open("track_paths_results.pkl", "rb") as fp:
    results = pickle.load(fp)


# Make an indices plot.
indices = np.vstack([ea[0] for ea in results if np.array(ea[0]).size > 0]).T[0]

bins = np.arange(max(indices) + 2) - 0.5
heights, _ = np.histogram(indices, bins=bins)
x = np.arange(1 + max(indices)).astype(int)

fig, ax = plt.subplots(figsize=(9, 3))
ax.barh(x, heights, facecolor="tab:blue")
ax.set_yticks(x)
ax.set_yticklabels([
    r"$\textrm{Component with largest} I_k$",
    r"$\textrm{Component with second-largest} I_k$",
    r"$\textrm{Component with third-largest} I_k$",
    r"$\textrm{Component with fourth-largest} I_k$"
])
ax.set_ylim(ax.get_ylim()[::-1])

ax.text(0.975, 0.05, 
        r"$N_{{operations}} = {{{0}}}$".format(indices.size),
        transform=ax.transAxes, 
        verticalalignment="bottom",
        horizontalalignment="right")

ax.text(0.975, 0.13,
        r"$N_{{datasets}} = {0}$".format(len(results)),
        transform=ax.transAxes,
        verticalalignment="bottom",
        horizontalalignment="right")


ax.set_xlabel(r"$\textrm{Number of component splits by benchmark method}$")
ax.yaxis.set_tick_params(width=0)

fig.tight_layout()



# Difference between Message Breaking Method and Kasarapu:
diff_I = np.array([(ea[1] - ea[4]) for ea in results])

print("Max absolute difference in message length between methods: {0}".format(np.max(np.abs(diff_I))))

fig.savefig("article/paths-splits.pdf", dpi=300)
