import csv

import numpy as np
from numpy import loadtxt
import time
import copy
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor

D = 8
repetitions = 30
folders = ["heating", "cooling"]


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred) / y_true)


def load_train_data_csv(folder, number):
    csvdata = 'ml-application/data/' + folder + '/train/' + 'energy_' + folder + '_data_' + str(number) + '.csv'
    dataset = loadtxt(csvdata, delimiter=',')
    X = dataset[:, :D]
    y = dataset[:, D]
    return X, y


def load_test_data_csv(folder, number):
    csvdata = 'ml-application/data/' + folder + '/test/' + 'energy_' + folder + '_data_' + str(number) + '.csv'
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

for folder in folders:
    for fold in range(5):
        print("The experiment is running for " + folder + " and energy_data_" + str(fold+1))

        fold_results_directory = results_directory + folder + "/"
        Path(fold_results_directory).mkdir(parents=True, exist_ok=True)

        Flag_details = False

        timerStart = time.time()
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

        for i in range(0, len(predictions)):
            predictions[i] = 1.0 / 4 * predictions1[i] + 1.0 / 4 * predictions2[i] \
                             + 1.0 / 4 * predictions3[i] + 1.0 / 4 * predictions4[i]

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
                header = np.array(["MAPE", "RMSE", "R2", "MAE", "ExecutionTime"])
                writer.writerow(header)
                Flag_details = True
            a = np.array([MAPE, rmse, r2, MAE, executionTime])
            writer.writerow(a)
        out.close()

        print('The values of the evaluation metrics are the following: ', 'MAPE =', MAPE, 'RMSE =', rmse, 'R2 =', r2,
              'MAE =', MAE)
