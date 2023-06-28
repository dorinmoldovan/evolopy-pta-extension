# -*- coding: utf-8 -*-
"""
Created on Tue May 24 13:13:28 2016

@author: Hossam Faris
"""
import math
import numpy
import random
import time
from solution import solution


def get_cuckoos(nest, best, lb, ub, n, dim):

    # perform Levy flights
    tempnest = numpy.zeros((n, dim))
    tempnest = numpy.array(nest)
    beta = 3 / 2
    sigma = (
        math.gamma(1 + beta)
        * math.sin(math.pi * beta / 2)
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)

    s = numpy.zeros(dim)
    for j in range(0, n):
        s = nest[j, :]
        u = numpy.random.randn(len(s)) * sigma
        v = numpy.random.randn(len(s))
        step = u / abs(v) ** (1 / beta)

        stepsize = 0.01 * (step * (s - best))

        s = s + stepsize * numpy.random.randn(len(s))

        for k in range(dim):
            tempnest[j, k] = numpy.clip(s[k], lb[k], ub[k])

    return tempnest


def get_best_nest(nest, newnest, fitness, n, dim, objf, ub, Pred, y_test_standardized):
    # Evaluating all new solutions
    tempnest = numpy.zeros((n, dim))
    tempnest = numpy.copy(nest)

    for j in range(0, n):
        fnew = objf(newnest[j, :], ub[0], Pred, y_test_standardized)
        if fnew <= fitness[j]:
            fitness[j] = fnew
            tempnest[j, :] = newnest[j, :]

    fmin = min(fitness)
    K = numpy.argmin(fitness)
    bestlocal = tempnest[K, :]

    return fmin, bestlocal, tempnest, fitness


# Replace some nests by constructing new solutions/nests
def empty_nests(nest, pa, n, dim, lb, ub):

    # Discovered or not
    tempnest = numpy.zeros((n, dim))

    K = numpy.random.uniform(0, 1, (n, dim)) > pa

    stepsize = random.random() * (
        nest[numpy.random.permutation(n), :] - nest[numpy.random.permutation(n), :]
    )

    tempnest = nest + stepsize * K

    for j in range(0, n):
        for k in range(dim):
            tempnest[j, k] = numpy.clip(tempnest[j, k], lb[k], ub[k])

    return tempnest


##########################################################################


def CS(objf, lb, ub, dim, n, N_IterTotal, Pred, y_test_standardized):

    # Discovery rate of alien eggs/solutions
    pa = 0.25

    nd = dim

    convergence = []
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    # RInitialize nests randomly
    nest = numpy.zeros((n, dim))
    for i in range(dim):
        nest[:, i] = numpy.random.uniform(0, 1, n) * (ub[i] - lb[i]) + lb[i]

    new_nest = numpy.zeros((n, dim))
    new_nest = numpy.copy(nest)

    bestnest = [0] * dim

    fitness = numpy.zeros(n)
    fitness.fill(float("inf"))

    s = solution()

    print('CS is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    fmin, bestnest, nest, fitness = get_best_nest(nest, new_nest, fitness, n, dim, objf, ub, Pred, y_test_standardized)

    convergence = []
    # Main loop counter
    for iter in range(0, N_IterTotal):
        # Generate new solutions (but keep the current best)

        new_nest = get_cuckoos(nest, bestnest, lb, ub, n, dim)

        # Evaluate new solutions and find best

        fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness, n, dim, objf, ub, Pred, y_test_standardized)

        new_nest = empty_nests(new_nest, pa, n, dim, lb, ub)

        # Evaluate new solutions and find best
        fnew, best, nest, fitness = get_best_nest(nest, new_nest, fitness, n, dim, objf, ub, Pred, y_test_standardized)

        if fnew < fmin:
            fmin = fnew
            bestnest = best

        if iter % 10 == 0:
            print(["At iteration " + str(iter) + " the best fitness is " + str(fmin)])
        convergence.append(fmin)

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence
    s.optimizer = "CS"
    s.objfname = objf.__name__
    s.gbest = bestnest

    return s
