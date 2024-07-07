import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import matplotlib.cm as cm
import matplotlib.colors as colors
from void_migration.params import load_file

plt.style.use("papers/Kinematic_SLM/post_process/paper.mplstyle")

with open("json/test_slope.json5") as f:
    dict, p = load_file(f)

W = p.H * (p.nx / p.ny)
x = np.linspace(-W / 2, W / 2, p.nx)
p.dx = x[1] - x[0]

L = W / 6.0  # length of dashed line
L_0 = W / 10.0  # intersection point with y data
y_off = 0  # -0.005

cmap = colormaps["inferno"]

fig = plt.figure(figsize=[3.31894680556, 3])
ax = fig.subplots(2, 1)

fig2 = plt.figure(figsize=[3.31894680556, 3])
ax2 = fig2.subplots(1, 1)

for j, nm in enumerate(p.nm):
    for i, angle in enumerate(p.repose_angle):
        try:
            files = glob(f"output/test_slope/nm_{nm}/repose_angle_{angle}/data/nu_*.npy")

            files.sort()
            if i == 0:
                data = np.load(files[0])
                bw = data >= p.nu_cs / 2.0
                # bw = data > 0
                # bw = data >= p.nu_cs
                top = np.argmin(bw, axis=1) * p.H / p.ny

                ax[0].plot(x, top, label="Initial", color="blue", ls="-", lw=3)

            data = np.load(files[-1])
            bw = data >= p.nu_cs / 2.0
            # bw = data > 0
            # bw = data >= p.nu_cs
            top = np.argmin(bw, axis=1) * p.H / p.ny

            color = cmap(i / (len(p.repose_angle) - 1))

            ax[0].plot(x, top, label=rf"$\varphi={angle}^\circ$", color=color)
            # plt.show()

            max_height = np.max(top)
            min_height = np.min(top)
            range_height = max_height - min_height

            # left = top[: np.argmin(np.abs(top - max_height))]
            left = top[: p.nx // 2]

            fit_min_arg = np.argmin(np.abs(left - (min_height + range_height / 4.0)))
            fit_max_arg = np.argmin(np.abs(left - (max_height - range_height / 4.0)))

            if fit_min_arg == fit_max_arg:
                coefficients = [0, min_height]
                x_fit = x
            elif angle == 90:
                fit_max_arg = np.argmin(np.abs(top - max_height))
                fit_min_arg = fit_max_arg - 1

                x_fit = x[fit_min_arg : fit_max_arg + 1]
                coefficients = np.polyfit(x_fit, top[fit_min_arg : fit_max_arg + 1], 1)
            else:
                x_fit = x[fit_min_arg:fit_max_arg]

                coefficients = np.polyfit(x_fit, top[fit_min_arg:fit_max_arg], 1)

            ax[0].plot(x_fit, coefficients[0] * x_fit + coefficients[1], ls="--", lw=2, color=color)

            color = cmap(j / (len(p.nm) - 1))
            ax[1].plot(angle, np.degrees(np.arctan(coefficients[0])), ".", mec=color, mfc=color)

            ax2.plot(
                p.delta_limit[i] / p.nu_cs, np.degrees(np.arctan(coefficients[0])), ".", mec=color, mfc=color
            )

        except IndexError:
            print(f"Missing file for repose angle={angle}")
        except ValueError:
            print(f"Old data file for repose angle={angle}")
        except TypeError:
            print(f"TypeError for repose angle={angle}")

plt.sca(ax[1])
plt.plot([0, 90], [0, 90], "k--")
plt.xlabel(r"$\varphi$ (degrees)", labelpad=0)
plt.ylabel("Measured angle of repose\n(degrees)")
plt.xticks([0, 30, 60, 90])
plt.yticks([0, 30, 60, 90])
# plt.xlim([0, 90])
# plt.ylim([0, 90])

plt.subplots_adjust(left=0.2, bottom=0.15, right=0.97, top=0.97, hspace=0.4)
plt.savefig("slope_test.png")

plt.sca(ax2)
phi = np.linspace(1, 89, 100)
mu = np.tan(np.radians(phi))
delta_nu = 1 / (1 / mu + 1)
plt.plot(delta_nu, phi, "k-")
plt.xlabel(r"$\Delta\nu/\nu_{cs}$")
plt.ylabel("Measured angle of repose\n(degrees)")
plt.subplots_adjust(left=0.2, bottom=0.15, right=0.97, top=0.97, hspace=0.4)
plt.savefig("delta_limit.png")
# plt.show()
