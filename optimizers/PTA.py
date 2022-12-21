import random
import numpy
from solution import solution
import time

def PTA(objf, lb, ub, dim, PopSize, iters):

    # PTA parameters

    ## polination probability
    PP = 0.5
    ## mutation rate
    MR = 0.5
    ## mutation threshold
    MT = 0.8

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

    Ripe_pos = numpy.zeros(dim)
    Ripe_score = float("inf")

    Unripe_pos = numpy.zeros(dim)
    Unripe_score = float("inf")

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
        for i in range(0, PopSize):
            # Calculate objective function
            fitness = flowerScore[i]

            # Update Alpha and Beta
            if fitness < Ripe_score:
                Unripe_score = Ripe_score  # Update beta
                Unripe_pos = Ripe_pos.copy()
                Ripe_score = fitness
                # Update alpha
                Ripe_pos = plums[i, :].copy()

            if fitness > Ripe_score and fitness < Unripe_score:
                Unripe_score = fitness  # Update beta
                Unripe_pos = plums[i, :].copy()

        for i in range(0, PopSize):
            # for j in range(dim):
            #     flowers[i, j] = numpy.clip(flowers[i, j], lb[j], ub[j])

            rp = random.random()

            if rp >= PP:
                for j in range(dim):
                    ri = random.random()
                    rj = random.random()
                    # similarity with neighbor plums (look at neighbors) - more plums one next to another
                    flowers[i][j] = flowers[i][j] + 2 * ri * (Ripe_pos[j] - flowers[i][j]) + rj * (Unripe_pos[j] - flowers[i][j])
                    # replace a part of the plums
            elif rp >= MR:
                for j in range(dim):
                    r = random.random()
                    if r > MT:
                        flowers[i][j] = random.random() * (ub[j] - lb[j]) + lb[j]
                    else:
                        flowers[i][j] = (flowers[i][j] + 2 * plums[i][j]) / 3
            else:
                for j in range(dim):
                    r1 = random.random()  # r1 is a random number in [0,1]
                    r2 = random.random()  # r2 is a random number in [0,1]
                    A = 2 * r1 - 1
                    B = 2 * r2
                    D_ripe = B * Ripe_pos[j] - plums[i, j]
                    X1 = Ripe_pos[j] - A * D_ripe

                    r1 = random.random()  # r1 is a random number in [0,1]
                    r2 = random.random()  # r2 is a random number in [0,1]
                    A = r1 - 1
                    B = r2
                    D_unripe = B * Unripe_pos[j] - plums[i, j]
                    X2 = Unripe_pos[j] + A * D_unripe
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