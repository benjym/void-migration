import os
import matplotlib.style
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colormaps

matplotlib.style.use("paper.mplstyle")

cmap = colormaps["inferno"]

# for mu in [0.1,0.2,0.5,1,2,10]:
phis = [10, 30, 50, 70]
for i, phi in enumerate(phis):
    mu = np.tan(np.radians(phi))

    delta_nu = 1 / (1 / mu + 1)
    Ms = np.logspace(1, 3, 100)
    delta_mus = []

    for M in Ms:
        delta_mu = 1 / (1 / (delta_nu + 1.0 / (M / 2.0)) - 1) - 1 / (1 / delta_nu - 1)
        delta_mus.append(delta_mu)

    delta_phi = np.degrees(np.arctan(delta_mus))
    color = cmap(i / (len(phis) - 1))

    plt.loglog(Ms, delta_phi, label=rf"$\varphi={phi:0.0f}^\circ$", c=color)
plt.xlabel("M")
# plt.ylabel(r'$\Delta\mu$')
plt.ylabel(rf"$\Delta\varphi$ ($^\circ$)")
plt.yticks([0.1, 1, 10, 100])
plt.legend(
    # loc=0,
    loc="upper right",
    bbox_to_anchor=(1.04, 1.04),
    # labelspacing=0.25,
    handlelength=1,
)
plt.subplots_adjust(left=0.2, right=0.99, top=0.95, bottom=0.2)
plt.savefig(os.path.expanduser("~/Dropbox/Apps/Overleaf/Kinematic SLM/im/discretisation_error.pdf"))
