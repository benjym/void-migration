import os
import sys
import json5
import numpy as np


class dict_to_class:
    """
    A convenience class to store the information from the parameters dictionary. Used because I prefer using p.variable to p['variable'].
    """

    def __init__(self, dict: dict):
        list_keys: List[str] = []
        lists: List[List] = []
        for key in dict:
            setattr(self, key, dict[key])
            if isinstance(dict[key], list) and key not in ["save", "plot", "videos", "T_cycles"]:
                list_keys.append(key)
                lists.append(dict[key])
        setattr(self, "lists", lists)
        setattr(self, "list_keys", list_keys)

    def set_defaults(self):
        with open("json/defaults.json5", "r") as f:
            defaults_dict = json5.loads(f.read())

        for key in defaults_dict:
            if not hasattr(self, key):
                setattr(self, key, defaults_dict[key])

        if not hasattr(self, "folderName"):
            self.folderName = "output/"

        if not os.path.exists(self.folderName):
            os.makedirs(self.folderName)

        if len(self.save) > 0:
            if not os.path.exists(self.folderName + "data/"):
                os.makedirs(self.folderName + "data/")
        if hasattr(self, "aspect_ratio_y"):
            self.ny = int(self.nx * self.aspect_ratio_y)
        if hasattr(self, "aspect_ratio_m"):
            self.nm = int(self.nx * self.aspect_ratio_m)

        if self.gsd_mode == "mono":
            if hasattr(self, "s_m") and hasattr(self, "s_M"):
                if not self.s_m == self.s_M:
                    print("WARNING: s_m and s_M must be equal for monodisperse grains. Setting both to s_m.")
                    self.s_M = self.s_m
            if hasattr(self, "s_m"):
                self.s_M = self.s_m
            if hasattr(self, "s_M"):
                self.s_m = self.s_M
            else:
                self.s_m = 1
                self.s_M = 1

        # user can define mu or repose_angle. If both defined, mu takes precedence.
        if hasattr(self, "mu"):
            self.repose_angle = np.degrees(np.arctan(self.mu))
        if hasattr(self, "repose_angle"):
            self.mu = np.tan(np.radians(self.repose_angle))
        # print(self.mu)
        # print(self.repose_angle)
        # sys.exit()

        with np.errstate(divide="ignore", invalid="ignore"):
            inv_mu = np.nan_to_num(1.0 / self.mu, nan=0.0, posinf=1e30, neginf=0.0)
        self.delta_limit = self.nu_cs / (inv_mu + 1)  # BIT TOO LOW — ACTUALLY PRETTY GOOD FOR nx = 40
        # self.delta_limit = 1 / (inv_mu + 1) # WAY TOO HIGH
        # self.delta_limit = 1 / (self.nu_cs*inv_mu + 1) # WAY TOO HIGH
        # self.delta_limit = 1 / (inv_mu/self.nu_cs + 1) # not bad! bit too high
        # self.delta_limit = self.nu_cs / (inv_mu/self.nu_cs + 1) # way too low
        # print(self.repose_angle)
        # self.delta_limit = self.nu_cs / (inv_mu*self.nu_cs + 1) # Works well for mu > 1, too high for mu < 1 ???


def load_file(f):
    # parse file
    dict = json5.loads(f.read())
    if len(sys.argv) > 1:
        dict["input_filename"] = (sys.argv[1].split("/")[-1]).split(".")[0]
    p = dict_to_class(dict)
    p.set_defaults()
    return dict, p


def update_before_time_march(p, cycles):
    p.y = np.linspace(0, p.H, p.ny)
    p.dy = p.y[1] - p.y[0]
    p.x = np.arange(-(p.nx - 0.5) / 2 * p.dy, (p.nx - 0.5) / 2 * p.dy, p.dy)  # force equal grid spacing
    p.dx = p.x[1] - p.x[0]
    if not np.isclose(p.dx, p.dy):
        sys.exit(f"Fatal error: dx != dy. dx = {p.dx}, dy = {p.dy}")

    p.X, p.Y = np.meshgrid(p.x, p.y, indexing="ij")

    p.y += p.dy / 2.0
    p.t = 0

    # p.t_p = p.s_m / np.sqrt(p.g * p.H)  # smallest confinement timescale (at bottom) (s)
    s_bar = (p.s_m + p.s_M) / 2.0  # mean diameter (m)
    p.free_fall_velocity = np.sqrt(p.g * s_bar)  # time to fall one mean diameter (s)
    p.diffusivity = p.alpha * p.free_fall_velocity * s_bar  # diffusivity (m^2/s)

    safe = False
    stability = 0.5
    while not safe:
        p.P_u_ref = stability
        p.dt = p.P_u_ref * p.dy / p.free_fall_velocity

        p.P_lr_ref = p.diffusivity * p.dt / p.dy**2  # ignoring factor of 2 because that assumes P=0.5
        # p.P_lr_ref = p.alpha * p.P_u_ref

        p.P_u_max = p.P_u_ref * (p.s_M / p.s_m)
        p.P_lr_max = p.P_lr_ref * (p.s_M / p.s_m)

        if p.vectorized:
            if p.P_u_max <= p.P_stab and p.P_lr_max <= p.P_stab:
                safe = True
            else:
                stability *= 0.95
        else:
            if p.P_u_max + 2 * p.P_lr_max <= p.P_stab:
                safe = True
            else:
                stability *= 0.95

    if p.charge_discharge:
        p.nt = cycles.set_nt(p)
    else:
        p.nt = int(np.ceil(p.t_f / p.dt))

    if hasattr(p, "saves"):
        p.save_inc = int(p.nt / p.saves)

    return p
