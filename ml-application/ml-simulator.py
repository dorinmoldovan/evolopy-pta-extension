import numpy as np
from numpy import loadtxt
import time
import copy
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np


D = 8


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


X_train, y_train = load_train_data_csv("heating", 1)
X_test, y_test = load_test_data_csv("heating", 1)
y_train_standardized, y_test_standardized = standardize_Y(y_train, y_test)
X_train_standardized, X_test_standardized = standardize_X(X_train, X_test, y_train, y_test)

from sklearn.ensemble import RandomForestRegressor
algorithm1 = RandomForestRegressor(n_estimators=100, random_state=42)

algorithm1.fit(X_train_standardized, y_train_standardized)
predictions1 = algorithm1.predict(X_test_standardized)

rmse = mean_squared_error(y_test_standardized, predictions1, squared=False)
errors = abs(predictions1 - y_test_standardized)
r2 = r2_score(y_test_standardized, predictions1)
MAPE = mean_absolute_percentage_error(y_test_standardized, predictions1)

print('The values of the evaluation metrics are the following: ', 'MAPE:', MAPE, 'RMSE:', rmse, 'R2:', r2, 'MAE:', round(np.mean(errors), 2))
