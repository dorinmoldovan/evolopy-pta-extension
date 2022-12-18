import random
import numpy
from solution import solution
import time

def PTA(objf, lb, ub, dim, PopSize, iters):

    # PTA parameters

    ## polination probability
    PP = 0.5
    ## mutation threshold
    MT = 0.3
    ## flowering rate
    FR = 0.8
    ## flowers proximity
    FP = 1
    ## mutation rate
    MR = 2 * 1.0 / 3

    s = solution()
    if not isinstance(lb, list):
        lb = [lb] * dim
    if not isinstance(ub, list):
        ub = [ub] * dim

    ######################## Initializations

    plums = numpy.zeros((PopSize, dim))

    flowerScore = numpy.zeros(PopSize)
    flowerScore.fill(float("inf"))

    plumScore = numpy.zeros(PopSize)
    plumScore.fill(float("inf"))

    Alpha_pos = numpy.zeros(dim)
    Alpha_score = float("inf")

    Beta_pos = numpy.zeros(dim)
    Beta_score = float("inf")

    # pBest = numpy.zeros((PopSize, dim))

    gBest = numpy.zeros(dim)
    gBestScore = float("inf")

    flowers = numpy.zeros((PopSize, dim))
    for i in range(dim):
        flowers[:, i] = numpy.random.uniform(0, 1, PopSize) * (ub[i] - lb[i]) + lb[i]
        plums[:, i] = flowers[:, i]

    for i in range(0, PopSize):
        flowerScore[i] = objf(flowers[i, :])
        plumScore[i] = objf(plums[i, :])

    minimum = plumScore[0]
    minIndex = 0
    for i in range(1, PopSize):
        if plumScore[i] < minimum:
            minimum = plumScore[i]
            minIndex = i
    if minimum < gBestScore:
        gBest = plums[minIndex, :].copy()
        gBestScore = minimum

    convergence_curve = numpy.zeros(iters)

    ############################################
    print('PTA is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")

    for l in range(0, iters):

        FP = 1 + l / iters

        for i in range(0, PopSize):
            # Calculate objective function
            fitness = flowerScore[i]

            # Update Alpha and Beta
            if fitness < Alpha_score:
                Beta_score = Alpha_score  # Update beta
                Beta_pos = Alpha_pos.copy()
                Alpha_score = fitness
                # Update alpha
                Alpha_pos = flowers[i, :].copy()

            if fitness > Alpha_score and fitness < Beta_score:
                Beta_score = fitness  # Update beta
                Beta_pos = flowers[i, :].copy()

        for i in range(0, PopSize):
            # for j in range(dim):
            #     flowers[i, j] = numpy.clip(flowers[i, j], lb[j], ub[j])

            rp = random.random()

            if rp >= PP:
                n = (i + l) % PopSize
                for j in range(dim):
                    ri = random.random()
                    rj = random.random()
                    # similarity with neighbor plums (look at neighbors) - more plums one next to another
                    flowers[i][j] = flowers[i][j] + ri * FR * (plums[n][j] - flowers[i][j]) \
                                    + rj * 2 * FR * (gBest[j] - flowers[i][j])
                    # replace a part of the plums
            elif rp >= MT:
                for j in range(dim):
                    if random.random() > MR:
                        flowers[i][j] = (random.random() * (ub[j] - lb[j]) + lb[j])
            else:
                for j in range(dim):
                    r1 = random.random()  # r1 is a random number in [0,1]
                    r2 = random.random()  # r2 is a random number in [0,1]
                    r3 = random.random()
                    A = 2 * FP * r1 - FP
                    B = FP * r2
                    C = FP * r3 + 2 * FP

                    D_alpha = abs(B * Alpha_pos[j] - flowers[i, j])
                    X1 = Alpha_pos[j] - (A - C) * D_alpha
                    D_beta = abs(B * Beta_pos[j] - flowers[i, j])
                    X2 = Beta_pos[j] - (A + C) * D_beta

                    flowers[i][j] = (2 * X1 + X2) / 3

        for i in range(0, PopSize):
            for j in range(dim):
                flowers[i, j] = numpy.clip(flowers[i, j], lb[j], ub[j])

        for i in range(0, PopSize):
            flowerScore[i] = objf(flowers[i, :])
            plumScore[i] = objf(plums[i, :])
            if flowerScore[i] < plumScore[i]:
                for j in range(dim):
                    plums[i][j] = flowers[i][j]
                plumScore[i] = flowerScore[i]

        minimum = plumScore[0]
        minIndex = 0
        for i in range(1, PopSize):
            if plumScore[i] < minimum:
                minimum = plumScore[i]
                minIndex = i
        if minimum < gBestScore:
            gBest = plums[minIndex, :].copy()
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
    s.optimizer = "PTA"
    s.objfname = objf.__name__

    return s