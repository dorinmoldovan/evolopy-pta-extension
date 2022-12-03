# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:50:25 2016

@author: hossam
"""

from optimizer import run

# Select optimizers
# "PSO", "GWO", "CS", "CSA", "HOA", "PTA"
optimizer = ["PSO", "GWO", "CS", "CSA", "HOA"]

# Select benchmark function"
# "f1", "f1", "f2", "f3", "f4", "f5", "f6", "f7"
# "g1", "g2", "g3", "g4", "g5", "g6", "g7"
# "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10"
objectivefunc = ["f1", "f2", "f2", "f3", "f4", "f5", "f6", "f7", "g1", "g2", "g3", "g4", "g5", "g6", "g7",
                 "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10"]

# Select number of repetitions for each experiment.
# To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.
NumOfRuns = 2

# Select general parameters for all optimizers (population size, number of iterations).
params = {"PopulationSize": 50, "Iterations": 1000}

# Choose whether to Export the results in different formats
export_flags = {
    "Export_avg": True,
    "Export_details": True,
    "Export_convergence": True,
    "Export_boxplot": True,
}

run(optimizer, objectivefunc, NumOfRuns, params, export_flags)
