import csv

import numpy as np
from numpy import loadtxt
import time
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
import ml_benchmark
import ml_optimizers.PSO as pso
import ml_optimizers.CSA as csa
import ml_optimizers.CS as cs
import ml_optimizers.CSO as cso
import ml_optimizers.GWO as gwo
import ml_optimizers.HOA as hoa
import ml_optimizers.PTA as pta


D = 8
repetitions = 30
folders = ["heating", "cooling"]
ensembles = ["average", "RFR", "GBR", "ABR", "ETR", "CSO", "PSO", "GWO", "CS", "CSA", "HOA", "PTA"]


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred) / y_true)


def load_train_data_csv(folder, number):
    csvdata = 'ml-data/energy_data/' + folder + '/train/' + 'energy_' + folder + '_data_' + str(number) + '.csv'
    dataset = loadtxt(csvdata, delimiter=',')
    X = dataset[:, :D]
    y = dataset[:, D]
    return X, y


def load_test_data_csv(folder, number):
    csvdata = 'ml-data/energy_data/' + folder + '/test/' + 'energy_' + folder + '_data_' + str(number) + '.csv'
    dataset = loadtxt(csvdata, delimiter=',')
    X = dataset[:, :D]
    y = dataset[:, D]
    return X, y


def standardize_Y(y_train, y_test):
    y_train_mean = np.mean(y_train[:])
    y_train_std = np.std(y_train[:])
    y_train_standardized = [0] * len(y_train)
    for i in range(len(y_train)):
        y_train_standardized[i] = (y_train[i] - y_train_mean) / y_train_std
    y_test_standardized = [0] * len(y_test)
    for i in range(len(y_test)):
        y_test_standardized[i] = (y_test[i] - y_train_mean) / y_train_std
    return y_train_standardized, y_test_standardized


def standardize_X(X_train, X_test, y_train, y_test):
    x_train_mean = [0] * D
    x_train_std = [0] * D
    for i in range(D):
        x_train_mean[i] = np.mean(X_train[:, i])
        x_train_std[i] = np.std(X_train[:, i])
    X_train_standardized = [[0 for i in range(D)] for j in range(len(y_train))]
    for i in range(D):
        for j in range(len(X_train[:, i])):
            X_train_standardized[j][i] = (X_train[j][i] - x_train_mean[i]) / x_train_std[i]
    X_test_standardized = [[0 for i in range(D)] for j in range(len(y_test))]
    for i in range(D):
        for j in range(len(X_test[:, i])):
            X_test_standardized[j][i] = (X_test[j][i] - x_train_mean[i]) / x_train_std[i]
    return X_train_standardized, X_test_standardized


results_directory = "ml-results/" + time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
Path(results_directory).mkdir(parents=True, exist_ok=True)


def compute_results():
    global Flag_details
    rmse = mean_squared_error(y_test_standardized, predictions, squared=False)
    MAE = mean_absolute_error(predictions, y_test_standardized)
    r2 = r2_score(y_test_standardized, predictions)
    MAPE = mean_absolute_percentage_error(y_test_standardized, predictions)
    timerEnd = time.time()
    executionTime = timerEnd - timerStart
    ExportToFile = fold_results_directory + "/results_energy_" + folder + "_data_" + str(fold + 1) + ".csv"
    with open(ExportToFile, "a", newline="\n") as out:
        writer = csv.writer(out, delimiter=",")
        if not Flag_details:
            header = np.array(["MAPE", "RMSE", "R2", "MAE", "w1", "w2", "w3", "w4", "ExecutionTime"])
            writer.writerow(header)
            Flag_details = True
        a = np.array([MAPE, rmse, r2, MAE, weights[0], weights[1], weights[2], weights[3], executionTime])
        writer.writerow(a)
    out.close()
    print('The values of the evaluation metrics are the following: ', 'MAPE =', MAPE, 'RMSE =', rmse,
          'R2 =', r2, 'MAE =', MAE)
    print("The weights were: ")
    print(weights)


def export_results_to_csv():
    global convergence, executionTime, objfname, Flag_details, Flag, folder, fold, ensemble
    convergence = [0] * repetitions
    executionTime = [0] * repetitions
    for k in range(0, repetitions):
        convergence[k] = x.convergence
        optimizerName = x.optimizer
        objfname = x.objfname

        Path(results_directory + "/" + ensemble + "/" + folder + "/" + "data" + "_" + str(fold+1)).mkdir(parents=True, exist_ok=True)
        ExportToFile = results_directory + "/" + ensemble + "/" + folder + "/" + "data" + "_" + str(fold+1) + "/" + "experiment_details.csv"
        with open(ExportToFile, "a", newline="\n") as out:
            writer = csv.writer(out, delimiter=",")
            if (
                    Flag_details == False
            ):  # just one time to write the header of the CSV file
                header = np.concatenate(
                    [["Optimizer", "objfname", "ExecutionTime"], CnvgHeader]
                )
                writer.writerow(header)
                Flag_details = True  # at least one experiment
            executionTime[k] = x.executionTime
            a = np.concatenate(
                [[x.optimizer, x.objfname, x.executionTime], x.convergence]
            )
            writer.writerow(a)
        out.close()
    Path(results_directory + "/" + ensemble + "/" + folder + "/" + "data" + "_" + str(fold+1)).mkdir(parents=True, exist_ok=True)
    ExportToFile = results_directory + "/" + ensemble + "/" + folder + "/" + "data" + "_" + str(fold+1) + "/" + "experiment.csv"
    with open(ExportToFile, "a", newline="\n") as out:
        writer = csv.writer(out, delimiter=",")
        if (
                Flag == False
        ):  # just one time to write the header of the CSV file
            header = np.concatenate(
                [["Optimizer", "objfname", "ExecutionTime"], CnvgHeader]
            )
            writer.writerow(header)
            Flag = True

        avgExecutionTime = float("%0.2f" % (sum(executionTime) / repetitions))
        avgConvergence = np.around(
            np.mean(convergence, axis=0, dtype=np.float64), decimals=6
        ).tolist()
        a = np.concatenate(
            [[optimizerName, objfname, avgExecutionTime], avgConvergence]
        )
        writer.writerow(a)
    out.close()


def initialize_csv_data():
    global Iterations, Flag, Flag_details, CnvgHeader, l
    Iterations = 500
    Flag = False
    Flag_details = False
    CnvgHeader = []
    for l in range(0, Iterations):
        CnvgHeader.append("Iter" + str(l + 1))


def compute_predictions():
    global i, weights
    for i in range(0, len(predictions)):
        predictions[i] = weights[0] * Pred[0][i] + weights[1] * Pred[1][i] \
                         + weights[2] * Pred[2][i] + weights[3] * Pred[3][i]


for ensemble in ensembles:
    for folder in folders:
        for fold in range(5):
            print("The experiment is running for ensemble " + ensemble + ", folder " + folder + ", and energy_data_" + str(fold+1))

            fold_results_directory = results_directory + ensemble + "/" + folder + "/"
            Path(fold_results_directory).mkdir(parents=True, exist_ok=True)

            Flag_details = False

            X_train, y_train = load_train_data_csv(folder, fold + 1)
            X_test, y_test = load_test_data_csv(folder, fold + 1)
            y_train_standardized, y_test_standardized = standardize_Y(y_train, y_test)
            X_train_standardized, X_test_standardized = standardize_X(X_train, X_test, y_train, y_test)

            algorithm1 = RandomForestRegressor(random_state=42)
            algorithm2 = GradientBoostingRegressor(random_state=42)
            algorithm3 = AdaBoostRegressor(random_state=42)
            algorithm4 = ExtraTreesRegressor(random_state=42)

            algorithm1.fit(X_train_standardized, y_train_standardized)
            predictions1 = algorithm1.predict(X_test_standardized)
            algorithm2.fit(X_train_standardized, y_train_standardized)
            predictions2 = algorithm2.predict(X_test_standardized)
            algorithm3.fit(X_train_standardized, y_train_standardized)
            predictions3 = algorithm3.predict(X_test_standardized)
            algorithm4.fit(X_train_standardized, y_train_standardized)
            predictions4 = algorithm4.predict(X_test_standardized)

            predictions = [0] * len(y_test_standardized)

            Pred = [[0 for i in range(len(predictions))] for j in range(4)]
            for i in range(0, len(predictions)):
                Pred[0][i] = predictions1[i]
                Pred[1][i] = predictions2[i]
                Pred[2][i] = predictions3[i]
                Pred[3][i] = predictions4[i]

            if ensemble == "average":
                timerStart = time.time()
                for i in range(0, len(predictions)):
                    predictions[i] = 1.0 / 4 * Pred[0][i] + 1.0 / 4 * Pred[1][i] \
                                     + 1.0 / 4 * Pred[2][i] + 1.0 / 4 * Pred[3][i]
                weights = [1.0 / 4, 1.0 / 4, 1.0 / 4, 1.0 / 4]
                compute_results()
            elif ensemble == "RFR":
                timerStart = time.time()
                for i in range(0, len(predictions)):
                    predictions[i] = Pred[0][i]
                weights = [1.0, 0.0, 0.0, 0.0]
                compute_results()
            elif ensemble == "GBR":
                timerStart = time.time()
                for i in range(0, len(predictions)):
                    predictions[i] = Pred[1][i]
                weights = [0.0, 1.0, 0.0, 0.0]
                compute_results()
            elif ensemble == "ABR":
                timerStart = time.time()
                for i in range(0, len(predictions)):
                    predictions[i] = Pred[2][i]
                weights = [0.0, 0.0, 1.0, 0.0]
                compute_results()
            elif ensemble == "ETR":
                timerStart = time.time()
                for i in range(0, len(predictions)):
                    predictions[i] = Pred[3][i]
                weights = [0.0, 0.0, 0.0, 1.0]
                compute_results()
            elif ensemble == "PSO":

                initialize_csv_data()
                for rep in range(repetitions):
                    timerStart = time.time()
                    x = pso.PSO(getattr(ml_benchmark, "RMSE"), -2000, 2000, 4, 50, Iterations, Pred, y_test_standardized)
                    weights = ml_benchmark.extract_weights(x.gbest, 2000)
                    compute_predictions()
                    compute_results()

                export_results_to_csv()
            elif ensemble == "CSA":

                initialize_csv_data()
                for rep in range(repetitions):
                    timerStart = time.time()
                    x = csa.CSA(getattr(ml_benchmark, "RMSE"), -2000, 2000, 4, 50, Iterations, Pred, y_test_standardized)
                    weights = ml_benchmark.extract_weights(x.gbest, 2000)
                    compute_predictions()
                    compute_results()

                export_results_to_csv()
            elif ensemble == "CS":

                initialize_csv_data()
                for rep in range(repetitions):
                    timerStart = time.time()
                    x = cs.CS(getattr(ml_benchmark, "RMSE"), -2000, 2000, 4, 50, Iterations, Pred, y_test_standardized)
                    weights = ml_benchmark.extract_weights(x.gbest, 2000)
                    compute_predictions()
                    compute_results()

                export_results_to_csv()
            elif ensemble == "CSO":

                initialize_csv_data()
                for rep in range(repetitions):
                    timerStart = time.time()
                    x = cso.CSO(getattr(ml_benchmark, "RMSE"), -2000, 2000, 4, 50, Iterations, Pred, y_test_standardized)
                    weights = ml_benchmark.extract_weights(x.gbest, 2000)
                    compute_predictions()
                    compute_results()

                export_results_to_csv()
            elif ensemble == "GWO":

                initialize_csv_data()
                for rep in range(repetitions):
                    timerStart = time.time()
                    x = gwo.GWO(getattr(ml_benchmark, "RMSE"), -2000, 2000, 4, 50, Iterations, Pred, y_test_standardized)
                    weights = ml_benchmark.extract_weights(x.gbest, 2000)
                    compute_predictions()
                    compute_results()

                export_results_to_csv()
            elif ensemble == "HOA":

                initialize_csv_data()
                for rep in range(repetitions):
                    timerStart = time.time()
                    x = hoa.HOA(getattr(ml_benchmark, "RMSE"), -2000, 2000, 4, 50, Iterations, Pred, y_test_standardized)
                    weights = ml_benchmark.extract_weights(x.gbest, 2000)
                    compute_predictions()
                    compute_results()

                export_results_to_csv()
            elif ensemble == "PTA":

                initialize_csv_data()
                for rep in range(repetitions):
                    timerStart = time.time()
                    x = pta.PTA(getattr(ml_benchmark, "RMSE"), -2000, 2000, 4, 50, Iterations, Pred, y_test_standardized)
                    weights = ml_benchmark.extract_weights(x.gbest, 2000)
                    compute_predictions()
                    compute_results()

                export_results_to_csv()
