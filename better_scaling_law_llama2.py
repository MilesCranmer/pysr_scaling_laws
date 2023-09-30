import numpy as np
import pandas as pd
from pysr import PySRRegressor
from multiprocessing import cpu_count

# -
fname = "combined_llama2.csv"
df = pd.read_csv(fname)

# -

model = PySRRegressor(
    model_selection="best",
    populations=30,
    population_size=100,
    niterations=100,
    maxsize=20,
    ncyclesperiteration=10000,
    weight_optimize=0.001,
    adaptive_parsimony_scaling=1000,
    binary_operators=["+", "-", "*", "/", "^"],
    constraints={"^": (-1, 3), "/": (-1, 5)},
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

