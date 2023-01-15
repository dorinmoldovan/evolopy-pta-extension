import random
import numpy
from solution import solution
import time


def CSO(objf, lb, ub, dim, PopSize, iters):

    # CSO parameters

    G = 10
    FL_MIN = 0.5
    FL_MAX = 0.9
    e = 1.e+300
    RP = 20
    HP = 60
    MP = 10

    TheBest = 0
    cType = [0 for i in range(PopSize)]

    E = [0 for i in range(PopSize)]

    nRoosters = (int)((PopSize * RP) / 100)
    nHens = (int)((PopSize * HP) / 100)
    nMHens = (int)((PopSize * MP) / 100)
    nChicks = PopSize - nRoosters - nHens
    hens = [0 for i in range(nHens)]
    chicks = [0 for i in range(nChicks)]
    roosters = [0 for i in range(nRoosters)]
    hRoosterRelation = {}
    cHenRelation = {}

    s = solution()
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    ######################## Initializations

    P = numpy.zeros((PopSize, dim))
    for i in range(dim):
        P[:, i] = numpy.random.uniform(0, 1, PopSize) * (ub[i] - lb[i]) + lb[i]

    tempP = numpy.zeros((PopSize, dim))
    for i in range(dim):
        tempP[:, i] = numpy.random.uniform(0, 1, PopSize) * (ub[i] - lb[i]) + lb[i]

    for i in range(PopSize):
        E[i] = objf(P[i])
        if E[i] < E[TheBest]:
            TheBest = i

    gBest = [0 for i in range(dim)]
    for j in range(dim):
        gBest[j] = P[TheBest][j]
    gBestScore = E[TheBest]

    convergence_curve = numpy.zeros(iters)

    ############################################
    print('CSO is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(0, iters):
        if l % G == 0:
            hRoosterRelation = {}
            cHenRelation = {}
            f_values = []
            for i in range(PopSize):
                f_values.append(E[i])
                cType[i] = 0
            f_values.sort()
            t_roosters = f_values[nRoosters - 1]
            t_hens = f_values[nRoosters + nHens - 1]
            rIndex = 0
            hIndex = 0
            cIndex = 0

            for i in range(PopSize):
                if E[i] <= t_roosters:
                    cType[i] = 2
                    roosters[rIndex] = i
                    rIndex = rIndex + 1
                    if rIndex >= nRoosters:
                        break
            for i in range(PopSize):
                if E[i] <= t_hens and cType[i] != 2:
                    cType[i] = 1
                    hens[hIndex] = i
                    hIndex = hIndex + 1
                    if hIndex >= nHens:
                        break
            for i in range(PopSize):
                if cType[i] == 0:
                    chicks[cIndex] = i
                    cIndex = cIndex + 1
            for i in range(nHens):
                hIndex = hens[i]
                r = random.randrange(0, nRoosters, 1)
                rIndex = roosters[r]
                hRoosterRelation[hIndex] = rIndex
            for i in range(nChicks):
                cIndex = chicks[i];
                hen = random.randrange(0, nMHens, 1)
                hIndex = hens[hen]
                cHenRelation[cIndex] = hIndex

        for i in range(PopSize):
            if cType[i] == 2:
                sigma_squared = 1
                k = random.randrange(0, nRoosters, 1)
                if E[k] < E[i]:
                    sigma_squared = numpy.exp((E[k] - E[i]) / (abs(E[i]) + e))
                for j in range (dim):
                    tempP[i][j] = P[i][j] * (1 + numpy.random.normal(0, sigma_squared))
            if cType[i] == 1:
                r1 = hRoosterRelation[i]
                r2 = r1
                while r1 != r2:
                    type = random.randrange(0, 2, 1)
                    if type == 0:
                        r2 = roosters[random.randrange(0, nRoosters, 1)]
                    else:
                        r2 = hens[random.randrange(0, nHens, 1)]

                s1 = 1
                if E[i] > E[r1]:
                    s1 = numpy.exp((E[i] - E[r1]) / (abs(E[i]) + e))
                s2 = 1
                if E[i] > E[r2]:
                    s2 = numpy.exp(E[r2] - E[i])
                for j in range(dim):
                    tempP[i][j] = P[i][j] + s1 * random.uniform(0, 1) * (P[r1][j] - P[i][j]) + random.uniform(0, 1) * s2 * (
                            P[r2][j] - P[i][j])
            if cType[i] == 0:
                m = cHenRelation[i]
                for j in range(dim):
                    tempP[i][j] = P[i][j] + random.uniform(FL_MIN, FL_MAX) * (P[m][j] - P[i][j])

        for i in range(PopSize):
            for j in range(dim):
                P[i][j] = tempP[i][j]
                if P[i][j] < lb[j]:
                    P[i][j] = lb[j]
                if P[i][j] > ub[j]:
                    P[i][j] = ub[j]

        TheBest = 0
        for i in range(PopSize):
            E[i] = objf(P[i])
            if E[i] < E[TheBest]:
                TheBest = i
        if E[TheBest] < gBestScore:
            gBestScore = E[TheBest]
            for j in range(dim):
                gBest[j] = P[TheBest][j]

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
    s.optimizer = "CSO"
    s.objfname = objf.__name__

    return s