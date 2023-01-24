import numpy
import math
from sklearn.metrics import mean_squared_error


def extract_weights(L, ub):
    dim = len(L)
    L = numpy.array(L)
    w = L + ub
    sum = numpy.sum(w)
    if sum == 0:
        for i in range(dim):
            w[i] = 1.0 / 4
    else:
        for i in range(dim):
            w[i] = w[i] / sum
    return w


def RMSE(L, ub, x, y):
    w = extract_weights(L, ub)
    z = [0 for i in range(len(x[0]))]
    o = 0
    for i in range(len(x[0])):
        for j in range(len(w)):
            z[i] = z[i] + w[j] * x[j][i]
        o = o + (z[i] - y[i])**2
    o = 1.0 * o / (2 * len(x[0]))
    return o
