# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:46:20 2016

@author: Hossam Faris
"""

import numpy
import math


# define the function blocks
def prod(it):
    p = 1
    for n in it:
        p *= n
    return p


def Ufun(x, a, k, m):
    y = k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))
    return y


def f1(x):
    s = numpy.sum(x ** 2)
    return s


def f2(x):
    o = numpy.sum(abs(x)) + numpy.prod(abs(x))
    return o


def f3(x):
    dim = len(x) + 1
    o = 0
    for i in range(1, dim):
        o = o + (numpy.sum(x[0:i])) ** 2
    return o


def f4(x):
    o = max(abs(x))
    return o


def f5(x):
    dim = len(x)
    o = numpy.sum(
        100 * (x[1:dim] - (x[0 : dim - 1] ** 2)) ** 2 + (x[0 : dim - 1] - 1) ** 2
    )
    return o


def f6(x):
    o = numpy.sum(numpy.floor(x + 0.5) ** 2)
    return o


def f7(x):
    dim = len(x)

    w = [i for i in range(len(x))]
    for i in range(0, dim):
        w[i] = i + 1
    o = numpy.sum(w * (x ** 4)) + numpy.random.uniform(0, 1)
    return o


def g1(x):
    o = sum(-x * (numpy.sin(numpy.sqrt(abs(x)))))
    return o


def g2(x):
    dim = len(x)
    o = numpy.sum(x ** 2 - 10 * numpy.cos(2 * math.pi * x)) + 10 * dim
    return o


def g3(x):
    dim = len(x)
    o = (
        -20 * numpy.exp(-0.2 * numpy.sqrt(numpy.sum(x ** 2) / dim))
        - numpy.exp(numpy.sum(numpy.cos(2 * math.pi * x)) / dim)
        + 20
        + numpy.exp(1)
    )
    return o


def g4(x):
    dim = len(x)
    w = [i for i in range(dim)]
    w = [i + 1 for i in w]
    o = numpy.sum(x ** 2) / 4000 - prod(numpy.cos(x / numpy.sqrt(w))) + 1
    return o


def g5(x):
    dim = len(x)
    w = [i for i in range(dim)]
    w = [i + 1 for i in w]
    o = numpy.sum(x ** 2) / 2000 - prod(numpy.cos(x * x / w)) ** 4 + 1
    return o


def g6(x):
    dim = len(x)
    o = (math.pi / dim) * (
        10 * numpy.sin(math.pi * (1 + (x[0] + 1) / 4))
        + numpy.sum(
            (((x[: dim - 1] + 1) / 4) ** 2)
            * (1 + 10 * numpy.sin(math.pi * (1 + (x[1 :] + 1) / 4)) ** 2)
        )
        + ((x[dim - 1] + 1) / 4) ** 2
    ) + numpy.sum(Ufun(x, 10, 100, 4))
    return o


def g7(x):
    dim = len(x)
    o = 0.1 * (
            numpy.sin(3 * math.pi * x[0])**2
            + numpy.sum(
                ((x[: dim] - 1)**2)
                * (1 + numpy.sin(3 * math.pi * x[: dim] + 1)**2)
            )
            + (x[dim - 1]-1)**2 * (1 + numpy.sin(2 * math.pi * x[dim-1])**2)
    ) + numpy.sum(Ufun(x, 5, 100, 4))
    return o


def getFunctionDetails(a):
    # [name, lb, ub, dim]
    param = {
        "f1": ["f1", -100, 100, 30],
        "f2": ["f2", -10, 10, 30],
        "f3": ["f3", -100, 100, 30],
        "f4": ["f4", -100, 100, 30],
        "f5": ["f5", -30, 30, 30],
        "f6": ["f6", -100, 100, 30],
        "f7": ["f7", -1.28, 1.28, 30],
        "g1": ["g1", -500, 500, 30],
        "g2": ["g2", -5.12, 5.12, 30],
        "g3": ["g3", -32, 32, 30],
        "g4": ["g4", -600, 600, 30],
        "g5": ["g5", -300, 300, 30],
        "g6": ["g6", -50, 50, 30],
        "g7": ["g7", -50, 50, 30],
    }
    return param.get(a, "nothing")
