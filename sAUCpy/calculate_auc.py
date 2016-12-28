# -*- coding: utf-8 -*-
'''
Provides the :class:`SAUC <sAUCpy.sAUCpy.SAUC>` class, a semi-parametric AUC method``.
'''

# class(object):

from numpy import *

# def calculate_auc(x, y=None, data=None):
#     def finv(x):
#         [-log((1/x) - 1) for i in x]

# def finv(x):
#     return [(-log(i) - 1) for i in x]
# x = [.2, .3, .4, .2]
# finv(x)

class SAUC(object):
    def __init__(self):
        print
        "This is init."

    def calculate(self, ya, yb):
        m = len(self.ya)
        p = len(self.yb)
        I = numpy.empty(m,p)
        
        for i in range(x):
            for j in range(y):
                if ya[i] > yb[j]: I[i,j] = 1
                elif ya[i] == yb[j]: I[i,j] = 0.5
                else: I[i,j] = 0
        return(I)

# This works
from numpy import *

ya = [3,2,4,3]
yb = [1,2,2,3]

def finv(x):
    return (-log(x) - 1)
    
m = len(ya)
p = len(yb)

I = numpy.zeros(shape = (m, p))

def calculate_auc(ya, yb, data = None):
    for i in range(m):
        for j in range(p):
            if ya[i] > yb[j]: I[i,j] = 1
            elif ya[i] == yb[j]: I[i,j] = 0.5
            else: I[i,j] = 0
            auchat = numpy.mean(I)
            finvhat = finv(auchat)
            vya = numpy.apply_along_axis(numpy.mean, 1, I)
            vyb = numpy.apply_along_axis(numpy.mean, 0, I)
            svarya = numpy.var(vya)
            svaryb = numpy.var(vyb)
            vhat_auchat = (svarya/m) + (svaryb/p)
            v_finv_auchat = vhat_auchat/((auchat**2)*(1-auchat)**2)
            logitauchat = log(auchat/(1-auchat))
            var_logitauchat = vhat_auchat /((auchat**2)*(1-auchat)**2)
            return(var_logitauchat)

calculate_auc(ya=ya, yb= yb)




