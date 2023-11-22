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
            if isinstance(dict[key], list):
                list_keys.append(key)
                lists.append(dict[key])
        setattr(self, "lists", lists)
        setattr(self, "list_keys", list_keys)

    def set_defaults(self):
        if not os.path.exists(self.folderName):
            os.makedirs(self.folderName)
        if not hasattr(self, "internal_geometry"):
            self.internal_geometry = False
        if not hasattr(self, "cyclic_BC"):
            self.cyclic_BC = False
        if not hasattr(self, "theta"):
            self.theta = 0
        if not hasattr(self, "refill"):
            self.refill = False
        if not hasattr(self, "close_voids"):
            self.close_voids = False
        if not hasattr(self, "charge_discharge"):
            self.charge_discharge = False
        if not hasattr(self, "diag"):
            self.diag = 0
        if not hasattr(self, "lagrangian"):
            self.lagrangian = False
        if not hasattr(self, "g"):
            self.g = 9.81
        if not hasattr(self, "nu_cs"):
            self.nu_cs = 0.5
        if not hasattr(self, "beta"):
            self.beta = 1.0
        if not hasattr(self, "videos"):
            self.videos = ["nu", "rel_nu", "s", "U_mag"]  # options are nu, rel_nu, s, u, v and U_mag

        if hasattr(self, "gsd_mode"):
            if self.gsd_mode == "mono":
                if hasattr(self, "s_m") and hasattr(self, "s_M"):
                    if not self.s_m == self.s_M:
                        print(
                            "WARNING: s_m and s_M must be equal for monodisperse grains. Setting both to s_m."
                        )
                        self.s_M = self.s_m
                if hasattr(self, "s_m"):
                    self.s_M = self.s_m
                if hasattr(self, "s_M"):
                    self.s_m = self.s_M
                else:
                    self.s_m = 1
                    self.s_M = 1

        # user can define mu or friction_angle, but not both. Default value is mu=0.5 (friction_angle = 26.6 degrees)
        if not hasattr(self, "mu") and not hasattr(self, "friction_angle"):
            self.mu = 0.5
        if hasattr(self, "mu") and hasattr(self, "friction_angle"):
            sys.exit("Cannot define both mu and friction angle")
        if hasattr(self, "mu"):
            self.friction_angle = np.degrees(np.arctan(self.mu))
        if hasattr(self, "friction_angle"):
            self.mu = np.tan(np.radians(self.friction_angle))


def load_file(f):
    # parse file
    dict = json5.loads(f.read())
    dict["input_filename"] = (sys.argv[1].split("/")[-1]).split(".")[0]
    p = dict_to_class(dict)
    return dict, p
