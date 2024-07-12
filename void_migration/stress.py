import numpy as np


# Implementing Eq 36 and 37 from:
# Models of stress fluctuations in granular media
# P. Claudin and J.-P. Bouchaud, M. E. Cates and J. P. Wittmer
# NOTE: in our geometry tan(psi) = 1 (because dx=dy)

# Since tan(psi) = 1, we have that c_0^2 = 1 - p.stress_fraction
# We CAN take c_0^2 = 1/(1 + 2\tan^2\phi) (phi friction is angle)
# OR we could directly take into account the local asymmetry to define c_0^2.


# Under c_0^2 = 1/(1 + 2\tan^2\phi) = 1 - p.stress_fraction, we have that
# p.stress_fraction = 2*tan(phi)/(2*tan(phi) + 1) = 2*mu/(2*mu + 1)
def calculate_stress(s, p):
    sigma = np.zeros([p.nx, p.ny, 3])  # sigma_xy, sigma_yy, mu
    # NOTE: NOT CONSIDERING INCLINED GRAVITY
    weight_of_one_cell = p.solid_density * p.dx * p.dy * p.g
    for j in range(p.ny - 2, -1, -1):
        for i in range(1, p.nx - 1):
            if np.sum(~np.isnan(s[i, j, :])) > 0:
                this_weight = np.sum(~np.isnan(s[i, j, :])) * weight_of_one_cell
                sigma[i, j, 0] = 0.5 * (sigma[i - 1, j + 1, 0] + sigma[i + 1, j + 1, 0]) + 0.5 * (
                    1 - p.stress_fraction
                ) * (sigma[i - 1, j + 1, 1] - sigma[i + 1, j + 1, 1])
                sigma[i, j, 1] = (
                    this_weight
                    + p.stress_fraction * sigma[i, j + 1, 1]
                    + 0.5 * (1 - p.stress_fraction) * (sigma[i - 1, j + 1, 1] + sigma[i + 1, j + 1, 1])
                    + 0.5 * (sigma[i - 1, j + 1, 0] - sigma[i + 1, j + 1, 0])
                )
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma[:, :, 2] = np.nan_to_num(
            np.abs(sigma[:, :, 0]) / sigma[:, :, 1], nan=0.0, posinf=1e30, neginf=0.0
        )
    return sigma
