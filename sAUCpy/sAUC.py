import numpy

def finv(x):
    return (-(numpy.log(x)) - 1)

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

ya = [3,2,4,3]
yb = [1,2,2,3]

calculate_auc(ya=ya, yb= yb)

def sAUC(x, y, data):
    assert x is not None, "Argument x (for e.g. response ~ x1 + x2) is missing."
    assert y is not None, "Argument y (treatment group) is missing."
    assert data is not None, "Argument data is missing. Please, specify name of dataframe."

sAUC(x=2,y=2,data=3)


from itertools import product
import pandas

# def expandgrid(d):
#   return pd.DataFrame([row for row in product(*d.values())], 
#                       columns=d.keys())

# dictionary = {'color': ['red', 'green'], 
#               'vehicle': ['car', 'van'], 
#               'cylinders': [6, 8]}

# f= pandas.DataFrame(dictionary)

# expandgrid(dictionary)

import itertools
def expand_grid(*itrs):
   product = list(itertools.product(*itrs))
   return {'Var{}'.format(i+1):[x[i] 
        for x in product] 
            for i in range(len(itrs))}

a = [1,2,3]
b = [5,7,9]
c = [1,3]
pd.DataFrame(expand_grid(a, b, c))


df=pd.DataFrame({"Animal":["dog","dolphin","chicken","ant","spider"],
                 "Legs":[4,0,2,6,8],
                 "Age" : [4,3,3,2,2],
                 "Gender":['f','m','f','m','f']})

def sAUC(y, x, group, data):
    input_response = y
    input_covariates = x
    input_treatment = group
    predictors.append(group)
    all_vars = predictors
    df = data[all_vars] 
    print(df)
    Warning("Data are being analyzed. Please, be patient.")



pass_vars(y='Gender', x = ['Animal', 'Legs'], group='Age', data=df)


sAUC()

## Get all levels of factors
import pandas as pd
import numpy as np
# Create an example dataframe
data = {'name': ['Jason', 'Molly','Molly', 'Tina', 'Jake', 'Amy'],
        'year': [2012, 2012, 2013, 2014, 2014, 2012],
        'reports': [4, 24, 31, 2, 3,2],
        'Dummy': [1,0,1,0,0,1]}
df = pd.DataFrame(data)
df.columns

ff = df.values.ravel()
ff

v = pd.DataFrame(np.reshape(ff, (-1,len(df.columns))))
v.columns = df.columns

## Create dummy matrix
import patsy
# patsy.dmatrices('Dummy ~ 1 + name + reports + year', v)
int(pd.get_dummies(v)) #, prefix=['col1', 'col2']))

## Group by
import pandas
from dplython import (DplyFrame, X, select, sift, sample_frac, sample_n, head, arrange,mutate, group_by,summarize)

df = pd.DataFrame({'A' : ['foo', 'foo', 'foo', 'foo',
                          'foo', 'foo', 'bar', 'bar'],
                  'B' : ['one', 'one', 'two', 'two',
                          'two', 'two', 'one', 'one'],
                   'C' : ['1', '0', '0', '1',
                          '1', '0', '1', '0'],
                   'D' : np.random.randn(8)})

grouped = df.groupby(['A','B','C'])['D']
df
# result = df.groupby(['A', 'B', 'C'])['D'].apply(list)

ke = grouped.groups.keys()
va = grouped.groups.values()

dictionary = dict(zip(ke, va))

# from collections import defaultdict
# dd = defaultdict(list)
# foo = [
#       {'host': 'localhost', 'db_name': 'test', 'table': 'partners'},
#       {'host': 'localhost', 'db_name': 'test', 'table': 'users'},
#       {'host': 'localhost', 'db_name': 'test', 'table': 'sales'},
#       {'host': 'localhost', 'db_name': 'new', 'table': 'partners'},
#       {'host': 'localhost', 'db_name': 'new', 'table': 'users'},
#       {'host': 'localhost', 'db_name': 'new', 'table': 'sales'},
# ]
# for d in foo:
#     dd[(d['host'], d['db_name'])].append(d)


pd.DataFrame(d.items(), columns=['Date', 'DateValue'])

for name, group in grouped:
    ii = name
    jj = group
    print(ii)
    print(jj)
    
items = grouped.items()


dff = DplyFrame(df)
ds_grouped = (dff >>
        group_by(X.A, X.B,X.C) >>
        arrange(X.C) >>
        sift(X.C == '1'))
ds_grouped














