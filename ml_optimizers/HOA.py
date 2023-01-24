# -*- coding: utf-8 -*-

import random
import numpy
from solution import solution
import time
import math
import numpy as np


def r():
    import random
    return random.uniform(0, 1);


def HOA(objf, lb, ub, dim, PopSize, iters, Pred, y_test_standardized):

    # HOA parameters

    DSP = 10
    SSP = 10
    HDR = 10
    HMP = 10
    M = 10
    vMin = -0.1
    vMax = 0.1
    sd = 1.0

    s = solution()
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    ######################## Initializations

    pos = numpy.zeros((PopSize, dim))
    vel = numpy.zeros((PopSize, dim))
    mem = numpy.zeros((HMP, PopSize, dim))

    eval = numpy.zeros(PopSize)
    evalHBest = numpy.zeros(PopSize)

    gBest = numpy.zeros(dim)
    gBestScore = float("inf")

    theBest = 0
    hType = numpy.zeros(PopSize)
    rank = numpy.zeros(PopSize)

    for i in range(dim):
        pos[:, i] = numpy.random.uniform(0, 1, PopSize) * (ub[i] - lb[i]) + lb[i]

    convergence_curve = numpy.zeros(iters)

    ############################################
    print('HOA is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for i in range(PopSize):
        fitness = objf(pos[i, :], ub, Pred, y_test_standardized)
        eval[i] = fitness
        evalHBest[i] = eval[i]
        if eval[i] < eval[theBest]:
            theBest = i

    for j in range(dim):
        gBest[j] = pos[theBest][j]
    gBestScore = eval[theBest]

    hHerdRelation = {}
    nStallions = (int)((PopSize * DSP) / 100)
    nSolitaryStallions = (int)((PopSize * SSP) / 100)
    nBottomHorses = (int)((PopSize * HDR) / 100)

    stallions = [0 for i in range(nStallions)]
    solitaryStallions = [0 for i in range(nSolitaryStallions)]
    bottomHorses = [0 for i in range(nBottomHorses)]

    for l in range(0, iters):

        sIndex = 0
        ssIndex = 0
        bhIndex = 0

        if (l % M) == 0:
            hHerdRelation = {}
            f_values = [0 for i in range(PopSize)]
            for i in range(PopSize):
                f_values[i] = eval[i]
                hType[i] = 0
                rank[i] = 0
            f_values.sort()
            t_stallions = f_values[nStallions - 1]
            t_solitary_stallions = f_values[nStallions + nSolitaryStallions - 1]
            t_bottom_horses = f_values[PopSize - nBottomHorses]
            for i in range(PopSize):
                if eval[i] <= t_stallions:
                    hType[i] = 1
                    stallions[sIndex] = i
                    sIndex = sIndex + 1
                    hHerdRelation[i] = i
                    if sIndex >= nStallions:
                        break
            for i in range(PopSize):
                if eval[i] <= t_solitary_stallions and hType[i] != 1:
                    hType[i] = 2
                    solitaryStallions[ssIndex] = i
                    ssIndex = ssIndex + 1
                    if ssIndex >= nSolitaryStallions:
                        break
            for i in range(PopSize):
                if eval[i] >= t_bottom_horses and hType[i] != 1 and hType[i] != 2:
                    hType[i] = 3
                    bottomHorses[bhIndex] = i
                    bhIndex = bhIndex + 1
                    if bhIndex >= nBottomHorses:
                        break
            for i in range(PopSize):
                if hType[i] != 1 and hType[i] != 2:
                    randomIndex = random.randrange(0, nStallions, 1)
                    herdIndex = stallions[randomIndex]
                    hHerdRelation[i] = herdIndex
            for i in range(PopSize):
                if hType[i] == 3:
                    for j in range(dim):
                        pos[i][j] = numpy.random.uniform(0, 1) * (ub[j] - lb[j]) + lb[j]
                    fitness = objf(pos[i, :], ub, Pred, y_test_standardized)
                    eval[i] = fitness
                    if eval[i] < evalHBest[i]:
                        evalHBest[i] = eval[i]
                    if evalHBest[i] < gBestScore:
                        TheBest = i
                        for j in range(dim):
                            gBest[j] = pos[TheBest][j]
                        gBestScore = evalHBest[i]

        rank = [0 for i in range(PopSize)]
        for i in range(PopSize):
            rng = 1
            k = 1
            if hType[i] != 2:
                herdIndex = hHerdRelation[i]
                for j in range(PopSize):
                    if i != j and hType[j] != 2 and hHerdRelation[j] == herdIndex:
                        k = k + 1
                        ## check this line
                        if eval[i] < eval[j]:
                            rng = rng + 1
                        if eval[i] == eval[j]:
                            if i > j:
                                rng = rng + 1
            rank[i] = rng * 1.0 / k
        herdCenter = [[0 for i in range(dim)] for j in range(PopSize)]
        denominators = [0 for i in range(PopSize)]
        for i in range(PopSize):
            if hType[i] != 2:
                herdIndex = hHerdRelation[i]
                denominators[herdIndex] = denominators[herdIndex] + rank[i]
        for i in range(PopSize):
            if hType[i] != 2:
                herdIndex = hHerdRelation[i]
                for j in range(dim):
                    herdCenter[herdIndex][j] = (herdCenter[herdIndex][j] + rank[i] * pos[i][j]) / denominators[herdIndex]
        for i in range(PopSize):
            gait = 1 + r()
            if hType[i] == 2:
                r_val = r()
                nHerd = stallions[0]
                distanceToNHerd = 0
                for j in range(dim):
                    distanceToNHerd = distanceToNHerd + (pos[i][j] - herdCenter[nHerd][j]) * (
                            pos[i][j] - herdCenter[nHerd][j])
                    distanceToNHerd = math.sqrt(distanceToNHerd)
                for k in range(nStallions):
                    sHerd = stallions[k]
                    distanceToCenter = 0
                    for j in range(dim):
                        distanceToCenter = distanceToCenter + (pos[i][j] - herdCenter[sHerd][j]) * (
                                pos[i][j] - herdCenter[sHerd][j])
                        distanceToCenter = math.sqrt(distanceToCenter)
                    if distanceToCenter < distanceToNHerd:
                        distanceToNHerd = distanceToCenter
                        nHerd = sHerd
                for j in range(dim):
                    vel[i][j] = vel[i][j] + r_val * gait * (herdCenter[nHerd][j] - pos[i][j])
            else:
                for j in range(dim):
                    herdIndex = hHerdRelation[i]
                    vel[i][j] = vel[i][j] + rank[i] * gait * (herdCenter[herdIndex][j] - pos[i][j])

        for i in range(PopSize):
            for j in range(dim):
                vel[i][j] = numpy.clip(vel[i][j], vMin, vMax)
                pos[i][j] = pos[i][j] + vel[i][j]
                pos[i][j] = numpy.clip(pos[i][j], lb[j], ub[j])

        for i in range(PopSize):
            for j in range(dim):
                for k in range(HMP):
                    mem[k][i][j] = pos[i][j] * np.random.normal(0, sd)
                    mem[k][i][j] = numpy.clip(mem[k][i][j], lb[j], ub[j])
            fitness = objf(pos[i, :], ub, Pred, y_test_standardized)
            x = [0 for j in range(dim)]
            for k in range(HMP):
                x = mem[k][i]
                value = objf(x, ub, Pred, y_test_standardized)
                if value < fitness:
                    fitness = value
                    for j in range(dim):
                        pos[i][j] = x[j];
            eval[i] = fitness
            if eval[i] < evalHBest[i]:
                evalHBest[i] = eval[i]
            if evalHBest[i] < gBestScore:
                TheBest = i
                for j in range(dim):
                    gBest[j] = pos[TheBest][j]
                gBestScore = evalHBest[i]

        for i in range(PopSize):
            rank[i] = 0

        for i in range(PopSize):
            rng = 1
            k = 1
            if hType[i] != 2:
                herdIndex = hHerdRelation[i]
                for j in range(PopSize):
                    if i != j and hType[j] != 2 and hHerdRelation[j] == herdIndex:
                        k = k + 1
                        ## Check this line
                        if eval[i] < eval[j]:
                            rng = rng + 1
                        if eval[i] == eval[i]:
                            if i > j:
                                rng = rng + 1
            rank[i] = rng * 1.0 / k
        for i in range(PopSize):
            for j in range(dim):
                herdCenter[i][j] = 0
            denominators[i] = 0
        for i in range(PopSize):
            if hType[i] != 2:
                herdIndex = hHerdRelation[i]
                denominators[herdIndex] = denominators[herdIndex] + rank[i]
        for i in range(PopSize):
            if hType[i] != 2:
                herdIndex = hHerdRelation[i]
                for j in range(dim):
                    herdCenter[herdIndex][j] = (herdCenter[herdIndex][j] + rank[i] * pos[i][j]) / denominators[herdIndex]
        for i in range(PopSize):
            if hType[i] == 2:
                nHerd = stallions[0]
                index = 0
                distanceToNHerd = 0
                for j in range(dim):
                    distanceToNHerd = distanceToNHerd + (pos[i][j] - herdCenter[nHerd][j]) * (
                            pos[i][j] - herdCenter[nHerd][j])
                    distanceToNHerd = math.sqrt(distanceToNHerd)
                for k in range(nStallions):
                    sHerd = stallions[k]
                    distanceToCenter = 0
                    for j in range(dim):
                        distanceToCenter = distanceToCenter + (pos[i][j] - herdCenter[sHerd][j]) * (
                                pos[i][j] - herdCenter[sHerd][j])
                        distanceToCenter = math.sqrt(distanceToCenter)
                    if distanceToCenter < distanceToNHerd:
                        distanceToNHerd = distanceToCenter
                        nHerd = sHerd
                        index = k
                if eval[nHerd] < eval[i]:
                    for j in range(dim):
                        aux = pos[nHerd][j]
                        pos[nHerd][j] = pos[i][j]
                        pos[i][j] = aux

                    for k in range(PopSize):
                        if hType[k] != 2:
                            hh = hHerdRelation[k]
                            if hh == nHerd:
                                hHerdRelation[k] = i

                    hType[i] = 1
                    hType[nHerd] = 2
                    hHerdRelation.pop(nHerd)
                    hHerdRelation[i] = i
                    stallions[index] = i

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
    s.optimizer = "HOA"
    s.objfname = objf.__name__
    s.gbest = gBest

    return s