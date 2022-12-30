import plot_convergence as conv_plot
import plot_boxplot as box_plot

# Select optimizers
# "PSO", "GWO", "CS", "CSA", "HOA", "PTA"
optimizer = ["PSO", "GWO", "CS", "CSA", "HOA", "PTA"]

# Select benchmark function"
# "f1", "f2", "f3", "f4", "f5", "f6", "f7"
# "g1", "g2", "g3", "g4", "g5", "g6", "g7"
# "h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10"
objectivefunc = ["f1"]

# number of iterations
Iterations = 1000

results_directory = "results/temp_files"

conv_plot.run(results_directory, optimizer, objectivefunc, Iterations)
box_plot.run(results_directory, optimizer, objectivefunc, Iterations)