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

df=pd.DataFrame({"Animal":["dog","dolphin","chicken","ant","spider"],
                 "Legs":[4,0,2,6,8],
                 "Age" : [4,3,3,2,2],
                 "Gender":['f','m','f','m','f']})

def sAUC(predictors, response, group, data):
    predictors.append(response); predictors.append(group)
    all_vars = predictors
    df = data[all_vars] 
    print(df)


pass_vars(x_vars = ['Animal', 'Legs'],yvar='Gender', group='Age', data=df)




def prod(*args):
    return(sum(args))
c = [2,3]
pd.DataFrame(expand_grid(a, b, c))
