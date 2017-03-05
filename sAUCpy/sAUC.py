import numpy
from numpy import log
import pandas

#def finv(x):
   # return(-log((1/x)-1))

fasd = pandas.read_csv("../data/one_final.csv")

fasd['group'] = fasd['group'].astype('category')
fasd['x1'] = fasd['x1'].astype('category')
fasd['x2'] = fasd['x2'].astype('category')

finv = lambda x: (-(numpy.log((1/x)-1)))
ya = [0, -2, -1, -3, -2, -1,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1 , 1 , 1,  1 , 1 , 1 , 1,  1, 1,  1 , 1 , 1,  1 , 1, 1 , 1 , 1]
yb = [2, 0, 4, 0, 2, 2, 2, 2, 1.5, 1, 1, 1, 1, 1, 1, 1]
finv(0.6)

def calculate_auc(ya, yb):
    m = len(ya)
    p = len(yb)
    I = numpy.zeros(shape = (m, p))
    for i in range(m):
        for j in range(p):
            if ya[i] > yb[j]: 
                I[i,j] = 1
            elif ya[i] == yb[j]: 
                I[i,j] = 0.5
            else: 
                I[i,j] = 0
    print(I)
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
    return(var_logitauchat, logitauchat)

calculate_auc(ya=ya, yb= yb)

response = list("y")
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
#    assert response is not None, "Argument response is missing."
#    assert treatment_group is not None, "Argument treatment_group is missing."
#    assert input_covariates is not None, "Argument input_covariates group is missing. Please put covariates as list. For e.g. ['x1','x2']"
#    assert data is not None, "Argument data is missing. Please, specify name of dataframe."

input_treatment = treatment_group

print("Data are being analyzed. Please, be patient.\n\n")

d = pandas.DataFrame(data)
group_covariates = input_treatment + input_covariates
d[group_covariates]
grouped_d = d.groupby(group_covariates)['y']


print(grouped_d.groups[keys])

keys = list(sorted(grouped_d.groups.keys()))
dict_df = {}
for index in range(len(keys)):
    print(index)
    #g1 = d.ix[grouped_d.groups[keys[index]]]
    #g2 = d.ix[grouped_d.groups[keys[index + 1]]]
    #g3 = d.ix[grouped_d.groups[keys[index + 2]]]
    #g4 = d.ix[grouped_d.groups[keys[index + 3]]]
    #g5 = d.ix[grouped_d.groups[keys[index + 4]]]
    #g6 = d.ix[grouped_d.groups[keys[index + 5]]]
    #g7 = d.ix[grouped_d.groups[keys[index + 6]]]
    #g8 = d.ix[grouped_d.groups[keys[index + 7]]]
    dict_df[index] = d.ix[grouped_d.groups[keys[index]]]
    
for i  in range(7) :
    print(d.ix[grouped_d.groups[keys[i]]])
dict_df[1] 
    
#dict_for_df = {}
#for i in ('a','b','c','d'):    # Don't use "id" as a counter; it's a python function
 #   x = numpy.random.random()        # first value
  #  y = numpy.random.random()        # second value
   # dict_for_df[i] = [x,y]
    
#dict_for_df['a'] 

#dict_for_df = {}
#for i in range(len(keys)):    
 #   print(i)     
  #  dict_for_df[i] = d.ix[grouped_d.groups[keys[i]]]

dict_df[4].iloc[:,0].tolist()

v1 = (calculate_auc(dict_df[0].iloc[:,0].tolist(),dict_df[4].iloc[:,0].tolist()))
my_1, my_card_1 = v1

v2 = (calculate_auc(dict_df[1].iloc[:,0].tolist(),dict_df[5].iloc[:,0].tolist()))
my_2, my_card_2 = v2

v3 = (calculate_auc(dict_df[2].iloc[:,0].tolist(),dict_df[6].iloc[:,0].tolist()))
my_3, my_card_3 = v3

v4 = (calculate_auc(dict_df[3].iloc[:,0].tolist(),dict_df[7].iloc[:,0].tolist()))
my_4, my_card_4 = v4

var_logitauchat = [my_1, my_2, my_3, my_4 ]
gamma1 = [my_card_1, my_card_2, my_card_3, my_card_4]










ke = grouped_d.groups.keys()
va = grouped_d.groups.values()

dictionary = dict(zip(ke, va))

from collections import OrderedDict
res = OrderedDict()
for v, k in dictionary:
    if k in res:
        res[k].append(v)
    else: res[k] = [v]
    

[{'type':k, 'items':v} for k,v in res.items()]


from collections import defaultdict
res = defaultdict(list)
for v, k in dictionary: res[k].append(v)

#grouped_d = pandas({"Count" : d.groupby(group_covariates)[response].size()}).reset_index()
#result = d.groupby(group_covariates).apply(list)
for i in range(int(0.5*len(grouped_d))):
    print(grouped_d)


for key, value in (sorted(dictionary.items())):
    key_list = list(key)
    print(key)
    df_dict = dict([d[response].ix[value]][0])
    print((df_dict))
    
#    for i in df_dict:
#    print((df_dict))
    #print([d[response].ix[value]])
#    for i in range(len(dictionaryTry)):
#    print(i)
#    dictionaryTry[list_names[i]] = dictionaryTry.pop('1',0)    
#    df_dict = (dict(df_dict[0]))
    
#    print(dict([[d[response].ix[value]][0]['y']].pop(0)))
         
#    super_dict = {}
#    for d in df_dict:
#        for k, v in d.iteritems():  # d.items() in Python 3+
#            super_dict.setdefault(k, []).append(v)
    
    #d = dict([("age", 25)])
    #final_dict = dict(key,df_dict[0]['y'])
#    final_dict = dict(key, ([d[response].ix[value]])[0]['y'])
#
#    print(final_dict)
       
# get levels
cat_columns = d.select_dtypes(['category']).columns

#d[cat_columns] = d[cat_columns].apply(lambda x: x.cat.codes)

x1_levels = d['x1'].cat.categories
x2_levels = d['x2'].cat.categories

#matrix = numpy.concatenate((d['x1'].cat.categories, d['x2'].cat.categories), axis=0)

ds_expand = (pandas.DataFrame(expand_grid(x1_levels, x2_levels)))
ds_expand['y'] = var_logitauchat

ds_expand['Var1'] = ds_expand['Var1'].astype('category')
ds_expand['Var2'] = ds_expand['Var2'].astype('category')
ds_expand.sort('Var2', ascending = [True])

from patsy import *

Z = pandas.DataFrame(dmatrix("Var1+Var2", ds_expand))
Z.columns = ['intercept', 'x1', 'x2']
Z.sort('x2', ascending = [True])

tau  =  numpy.diag([1/i for i in var_logitauchat])

from numpy.linalg import inv
ztauz = inv(Z.T.dot(tau).dot(Z))

var_betas = numpy.diag(ztauz)
std_error = numpy.sqrt(var_betas)
betas = ztauz.dot(Z.T).dot(tau).dot(gamma1)

from scipy.stats import norm
threshold = norm.ppf(0.975)

lo = betas - threshold*std_error
up = betas + threshold*std_error
ci = numpy.vstack((betas,lo,up)).T



x = numpy.matrix( ((2,3), (3, 5)) )
y = numpy.matrix( ((1,2), (5, -1)) )
x * y




























d[input_covariates].apply(lambda x: x.cat.codes)

input_covariates
d[cat_columns]


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
pandas.DataFrame(expand_grid(a, b, c))


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









