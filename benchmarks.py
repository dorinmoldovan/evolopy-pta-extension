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


def f1(x):
    s = numpy.sum(x ** 2)
    return s


def f2(x):
    o = sum(abs(x)) + prod(abs(x))
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
    o = numpy.sum(math.floor(x + 0.5) ** 2)
    return o


def f7(x):
    dim = len(x)

    w = [i for i in range(len(x))]
    for i in range(0, dim):
        w[i] = i + 1
    o = numpy.sum(w * (x ** 4)) + numpy.random.uniform(0, 1)
    return o


def f8(x):
    o = sum(-x * (numpy.sin(numpy.sqrt(abs(x)))))
    return o


def f9(x):
    dim = len(x)
    o = numpy.sum(x ** 2 - 10 * numpy.cos(2 * math.pi * x)) + 10 * dim
    return o


def f10(x):
    dim = len(x)
    o = (
        -20 * numpy.exp(-0.2 * numpy.sqrt(numpy.sum(x ** 2) / dim))
        - numpy.exp(numpy.sum(numpy.cos(2 * math.pi * x)) / dim)
        + 20
        + numpy.exp(1)
    )
    return o


def f11(x):
    dim = len(x)
    w = [i for i in range(dim)]
    w = [i + 1 for i in w]
    o = numpy.sum(x ** 2) / 4000 - prod(numpy.cos(x / numpy.sqrt(w))) + 1
    return o


def f12(x):
    dim = len(x)
    w = [i for i in range(dim)]
    w = [i + 1 for i in w]
    o = numpy.sum(x ** 2) / 2000 - prod(numpy.cos(x * x / w)) ** 4 + 1
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
        "f8": ["f8", -500, 500, 30],
        "f9": ["f9", -5.12, 5.12, 30],
        "f10": ["f10", -32, 32, 30],
        "f11": ["f11", -600, 600, 30],
        "f12": ["f12", -300, 300, 30],
    }
    return param.get(a, "nothing")
