import random
import numpy
from solution import solution
import time

def PTA(objf, lb, ub, dim, PopSize, iters):

    # PTA parameters
    ## epsilon
    eps = 1.e-300
    ## fruitness threshold
    FT = 0.8
    ## ripeness threshold
    RT = 0.2
    ## minimum fruitiness rate
    FRmin = 0.5
    ## maximum fruitiness rate
    FRmax = 1

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
            fitness = plumScore[i]

            if fitness < Ripe_score:
                Unripe_score = Ripe_score
                Unripe_pos = Ripe_pos.copy()
                Ripe_score = fitness
                Ripe_pos = plums[i, :].copy()

            if fitness > Ripe_score and fitness < Unripe_score:
                Unripe_score = fitness
                Unripe_pos = plums[i, :].copy()

        for i in range(0, PopSize):
            rp = random.random()
            if rp >= FT:
                for j in range(dim):
                    flowers[i][j] = flowers[i][j] + random.uniform(FRmin, FRmax) * (plums[i][j] - flowers[i][j])
            elif rp >= RT:
                for j in range(dim):
                    r1 = random.random()
                    r2 = random.random()

                    flowers[i][j] = flowers[i][j] + 2 * r1 * (Ripe_pos[j] - flowers[i][j]) \
                                    + 2 * r2 * (Unripe_pos[j] - flowers[i][j])
            else:
                sigma_ripe = 1
                if plumScore[i] >= Ripe_score:
                    sigma_ripe = numpy.exp((Ripe_score - plumScore[i]) / (abs(plumScore[i]) + eps))
                for j in range(dim):
                    flowers[i][j] = plums[i][j] * (1 + numpy.random.normal(0, sigma_ripe))

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