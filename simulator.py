# -*- coding: utf-8 -*-
"""
Created on Tue May 17 15:50:25 2016

@author: hossam
"""

from optimizer import run

# Select optimizers
# "PSO", "GWO", "CS", "CSA", "HOA", "PTA"
optimizer = ["PSO"]

# Select benchmark function"
# "f1"
objectivefunc = ["f1"]

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
