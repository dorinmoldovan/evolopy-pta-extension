# -*- coding: utf-8 -*-
"""
Created on Tue May 17 12:46:20 2016

@author: Hossam Faris
"""

import numpy

def f1(x):
    s = numpy.sum(x ** 2)
    return s

def getFunctionDetails(a):
    # [name, lb, ub, dim]
    param = {
        "f1": ["f1", -100, 100, 30],
    }
    return param.get(a, "nothing")
