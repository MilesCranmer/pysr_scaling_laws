# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
from pysr import PySRRegressor
from multiprocessing import cpu_count

# +
# Hierarchical CSV: first row describes (7B, 70B, 13B, 34B)
# second row: ((x, y), (x, y), ...)
# Thus, we manually parse the columns:
fname = "wpd_datasets.csv"
params = [7, 70, 13, 34]
dfs = []

# First two columns:
for (start, end), param in zip([(0, 2), (2, 4), (4, 6), (6, 8)], params):
    x, y = np.genfromtxt(fname, delimiter=",", skip_header=2)[:, start:end].T
    # Remove NaNs:
    x, y = x[~np.isnan(y) & ~np.isnan(x)], y[~np.isnan(y) & ~np.isnan(x)]

    dfs.append(pd.DataFrame({"context": x, "loss": y, "params": param}))

df = pd.concat(dfs)

# -

model = PySRRegressor(
    model_selection="best",
    populations=30,
    population_size=100,
    niterations=100,
    maxsize=30,
    ncyclesperiteration=10000,
    weight_optimize=0.001,
    adaptive_parsimony_scaling=1000,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["log"],
    constraints={"^": (-1, 3)},
    multithreading=False,  # Multiprocessing instead
    procs=cpu_count(),
    turbo=True,
)

# -

model.fit(df[["context", "params"]], df["loss"])

# -

latex_output = "table.tex"
print("I will save a LaTeX table to table.tex")
s = model.latex_table()
with open(latex_output, "w") as f:
    f.write(s)

