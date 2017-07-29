from pandas import DataFrame, read_csv
from saucpy import saucpy

# Data analysis
fasd = read_csv("../data/fasd.csv")
fasd['group'] = fasd['group'].astype('category')
fasd['x1']    = fasd['x1'].astype('category')
fasd['x2']    = fasd['x2'].astype('category')

ds = read_csv("../data/ds.csv")
ds['group'] = ds['group'].astype('category')
ds['x1']    = ds['x1'].astype('category')
ds['x2']    = ds['x2'].astype('category')
ds['x3']    = ds['x3'].astype('category')

print(saucpy.sAUCpy.sauc(response = "y", treatment_group = ["group"], input_covariates = ["x1"], data = fasd))
# sAUCpy.sauc(response = "y", treatment_group = ["group"], input_covariates = ["x1","x2"], data = fasd)
#
# sAUCpy.sauc(response = "y", treatment_group = ["group"], input_covariates = ["x1","x2", "x3"], data = ds)

saucpy.sAUCpy.calculate_auc(ya = [2,3,2,2,1,1], yb=[2,3,2,1,2,3,4])