import numpy as np
import warnings


# Implementing Eq 36 and 37 from:
# Models of stress fluctuations in granular media
# P. Claudin and J.-P. Bouchaud, M. E. Cates and J. P. Wittmer
# NOTE: in our geometry tan(psi) = 1 (because dx=dy)

# Since tan(psi) = 1, we have that c_0^2 = 1 - stress_fraction

# ISOTROPIC STRESS
# c_0^2 = lateral earth pressure coefficient, K
# Using Jaky's formula, K = 1 - sin(phi), where phi is the repose angle (actually effective angle of internal friction)
# stress_fraction = sin(phi)

# ANISOTROPIC STRESS
# Rothenburg and Bathurst: mu approx = a/2
# Can take a = beta x magnitude of 1/M sum_k P^{last}_k
# Where P_{last}^k = -1 for vertical and 1 for horizontal? (So that a = 0 for homogenous, 1 for fully ordered

# Putting this all together
# stress_fraction = 2 * (a/2)^2 / (1 + 2*(a/2)^2) = a^2 / (1 + a^2)


def calculate_stress(s, last_swap, p):
    if p.stress_mode == "isotropic":
        stress_fraction = np.sin(np.radians(p.repose_angle))
    elif p.stress_mode == "anisotropic":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            a = np.abs(
                np.nanmean(last_swap, axis=2)
            )  # between 0 and 1, 0 for isotropic, 1 for fully anisotropic
            stress_fraction = a**0.5

    sigma = np.zeros([p.nx, p.ny, 3])  # sigma_xy, sigma_yy, mu
    # NOTE: NOT CONSIDERING INCLINED GRAVITY
    weight_of_one_cell = p.solid_density * p.dx * p.dy * p.g
    for j in range(p.ny - 2, -1, -1):
        for i in range(p.nx):
            if np.sum(~np.isnan(s[i, j, :])) > 0:
                this_weight = np.sum(~np.isnan(s[i, j, :])) * weight_of_one_cell
                up = sigma[i, j + 1]
                if i == 0:
                    right_up = sigma[i + 1, j + 1]
                    # left_up = [0, 0] # walls carry no load!!
                    left_up = right_up  # FIXME: no gradient at boundary???

                elif i == p.nx - 1:
                    left_up = sigma[i - 1, j + 1]
                    # right_up = [0, 0]
                    right_up = left_up  # FIXME: no gradient at boundary???
                else:
                    left_up = sigma[i - 1, j + 1]
                    right_up = sigma[i + 1, j + 1]
                sigma[i, j, 0] = 0.5 * (left_up[0] + right_up[0]) + 0.5 * (1 - stress_fraction) * (
                    left_up[1] - right_up[1]
                )
                sigma[i, j, 1] = (
                    this_weight
                    + stress_fraction * up[1]
                    + 0.5 * (1 - stress_fraction) * (left_up[1] + right_up[1])
                    + 0.5 * (left_up[0] - right_up[0])
                )
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma[:, :, 2] = np.nan_to_num(
            np.abs(sigma[:, :, 0]) / sigma[:, :, 1], nan=0.0, posinf=1e30, neginf=0.0
        )

    # import matplotlib.pyplot as plt

    # plt.figure(45, figsize=(10, 10))
    # plt.clf()
    # plt.ion()
    # plt.imshow(a, origin="upper", cmap="bwr", vmin=-1, vmax=1)
    # plt.colorbar()
    # plt.pause(1e-3)

    return sigma
