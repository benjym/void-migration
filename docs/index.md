# Introduction

A python package for simulating the motion of a granular material as a result of the motion of voids. In particular, it (currently) models non-inertial problems quite well. In particular, segregation is well modelled for several systems.

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Installation

1.  Download and unzip (or clone) this repository
2.  Install python 3.11 or newer
3.  Install the required python packages with `pip install numpy matplotlib tqdm json5`
4.  Run the code with `python void_migration/void_migration.py json/collapse.json5`. The parameters for this specific case are stored in `json/collapse.json5`. Change that file name to a different `json5` file to use those values instead.

# Authors
- [Benjy Marks](mailto:benjy.marks@sydney.edu.au)
