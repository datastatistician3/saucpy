import numpy
import pandas

#def finv(x):
 #   return(-(numpy.log((1/x))-1))
 
fasd = pandas.read_csv("../data/one_final.csv")

fasd['group'] = fasd['group'].astype('category')
fasd['x1'] = fasd['x1'].astype('category')
fasd['x2'] = fasd['x2'].astype('category')

finv = lambda x: (-(numpy.log((1/x))-1))
za = [3,2,4,3]
zb = [1,2,2,3]

m = len(za)
p = len(zb)
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
            logitauchat = numpy.log(auchat/(1-auchat))
            var_logitauchat = vhat_auchat /((auchat**2)*(1-auchat)**2)
            return(var_logitauchat)

calculate_auc(ya=za, yb= zb)

response = ["y"]
input_covariates = ["x1","x2"]

#y = "group"
treatment_group = ["group"]
data = fasd

data[response]


#split
#get levels
#expand.grid
#model.atrix



#def sAUC(response, treatment_group, input_covariates, data):
    assert response is not None, "Argument response is missing."
    assert treatment_group is not None, "Argument treatment_group is missing."
    assert input_covariates is not None, "Argument input_covariates group is missing. Please put covariates as list. For e.g. ['x1','x2']"
    assert data is not None, "Argument data is missing. Please, specify name of dataframe."

input_treatment = treatment_group

print("Data are being analyzed. Please, be patient.\n\n")

d = pandas.DataFrame(data)
group_covariates = input_treatment + input_covariates
d[group_covariates]
grouped_d = d.groupby(group_covariates)['y']
#grouped_d = pandas({"Count" : d.groupby(group_covariates)[response].size()}).reset_index()
#result = d.groupby(group_covariates).apply(list)



ke = grouped_d.groups.keys()
va = grouped_d.groups.values()

dictionary = dict(zip(ke, va))

for key, value in sorted(dictionary.items()):
    print(key)
    df_dict = ([d[response].ix[value]])



# get levels
cat_columns = d.select_dtypes(['category']).columns

cat_columns

d[cat_columns] = d[cat_columns].apply(lambda x: x.cat.categories)

d['x2'].cat.categories
d[input_covariates].apply(lambda x: x.cat.codes)

cat.categories
d[cat_columns]



pandas.Factor(3)




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
import pandas as pd
import numpy as np
from dplython import (DplyFrame, X, select, sift, sample_frac, sample_n, head, arrange,mutate, group_by,summarize)

df = pd.DataFrame({'A' : ['foo', 'foo', 'foo', 'foo',
                          'foo', 'foo', 'bar', 'bar'],
                  'B' : ['one', 'one', 'two', 'two',
                          'two', 'two', 'one', 'one'],
                   'C' : ['1', '0', '0', '1',
                          '1', '0', '1', '0'],
                   'D' : np.random.randn(8)})

grouped = df.groupby(['A','B','C','D'])
grouped

result = df.groupby(['A', 'B', 'C'])['D'].apply(list)
result
ke = grouped.groups.keys()
va = grouped.groups.values()

dictionary = dict(zip(ke, va))
dictionary
########################## Split in python like in R ####################
a = [0,1] * 2
b = [1,2]*2
a
split(a,b)
([1, 3, 5, 7, 9], [2, 4, 6, 8, 10])

import itertools
def split(x, *f):
    return list(itertools.compress(x, f)), list(itertools.compress(x, (not i for i in f)))
# If you need more general input (multiple numbers), something like the following will return an n-tuple:

split(d, input_covariates)
    
    
def split(x, f):
    count = max(f) + 1
    return tuple( list(itertools.compress(x, (el == i for el in f))) for i in range(count) )  

split([1,2,3,4,5,6,7,8,9,10], b+c)

split([1,2,3,4,5,6,7,8,9,10], [0,1,1,0,2,3,4,0,1,2])

split(d, group_covariates)


########################## End of Split in python like in R ####################


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



def split(x, f):
    count = max(f) + 1
    return tuple( list(itertools.compress(x, (el == i for el in f))) for i in range(count) )  

split([1,2,3,4,5,6,7,8,9,10], [0,1,1,0,2,3,4,0,1,2])









