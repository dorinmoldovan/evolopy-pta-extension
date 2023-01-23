import random
import numpy
from solution import solution
import time

def CSA(objf, lb, ub, dim, PopSize, iters, Pred, y_test_standardized):

    # CSA parameters

    fl = 2
    AP = 0.1

    s = solution()
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    ######################## Initializations

    mem = numpy.zeros((PopSize, dim))

    crowScore = numpy.zeros(PopSize)
    crowScore.fill(float("inf"))

    memScore = numpy.zeros(PopSize)
    memScore.fill(float("inf"))

    # pBest = numpy.zeros((PopSize, dim))

    gBest = numpy.zeros(dim)
    gBestScore = float("inf")

    pos = numpy.zeros((PopSize, dim))
    for i in range(dim):
        pos[:, i] = numpy.random.uniform(0, 1, PopSize) * (ub[i] - lb[i]) + lb[i]
        mem[:, i] = pos[:, i]

    convergence_curve = numpy.zeros(iters)

    ############################################
    print('CSA is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(0, iters):
        for i in range(0, PopSize):

            for j in range(dim):
                pos[i, j] = numpy.clip(pos[i, j], lb[j], ub[j])

            ri = random.random()
            k = random.randint(0, PopSize-1)

            if random.random() >= AP:
                for j in range(dim):
                    pos[i][j] = pos[i][j] + ri * fl * (mem[k][j] - pos[i][j])
            else:
                for j in range(dim):
                    pos[i][j] = random.random() * (ub[j] - lb[j]) + lb[j]

        for i in range(0, PopSize):
            for j in range(dim):
                pos[i, j] = numpy.clip(pos[i, j], lb[j], ub[j])

        for i in range(0, PopSize):
            crowScore[i] = objf(pos[i, :], ub, Pred, y_test_standardized)
            memScore[i] = objf(mem[i, :], ub, Pred, y_test_standardized)
            if crowScore[i] < memScore[i]:
                for j in range(dim):
                    mem[i][j] = pos[i][j]
                memScore[i] = crowScore[i]

        minimum = memScore[0]
        minIndex = 0
        for i in range(1, PopSize):
            if memScore[i] < minimum:
                minimum = memScore[i]
                minIndex = i
        if minimum < gBestScore:
            gBest = mem[minIndex, :].copy()
            gBestScore = minimum

        convergence_curve[l] = gBestScore

        if l % 10 == 0:
            print(
                [
                    "At iteration "
                    + str(l)
                    + " the best fitness is "
                    + str(gBestScore)
                ]
            )
    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "CSA"
    s.objfname = objf.__name__
    s.gbest = gBest

    return s