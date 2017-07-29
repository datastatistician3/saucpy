import numpy
from pandas import DataFrame
from patsy import dmatrix
from scipy.stats import norm

class sAUC(object):
    def calculate_auc(ya, yb):
        m = len(ya)
        p = len(yb)
        I = numpy.zeros(shape=(m, p))
        for i in range(m):
            for j in range(p):
                if ya[i] > yb[j]:
                    I[i, j] = 1
                elif ya[i] == yb[j]:
                    I[i, j] = 0.5
                else:
                    I[i, j] = 0
        #finv = lambda x: (-(numpy.log((1/x)-1)))
        auchat = numpy.mean(I)
        #finvhat = finv(auchat)
        vya = numpy.apply_along_axis(numpy.mean, 1, I)
        vyb = numpy.apply_along_axis(numpy.mean, 0, I)
        svarya = numpy.var(vya)
        svaryb = numpy.var(vyb)
        vhat_auchat = (svarya / m) + (svaryb / p)
        #v_finv_auchat = vhat_auchat/((auchat**2)*(1-auchat)**2)
        logitauchat = numpy.log(auchat / (1 - auchat))
        var_logitauchat = vhat_auchat / ((auchat**2) * (1 - auchat)**2)
        return([var_logitauchat, logitauchat])

    #print(calculate_auc(ya=[2,0.4,3.6,2.41], yb= [1.2,0.4,1.6,1.5]))
        
    # expand.grid
    def expand_grid(*itrs):
        def product(*args, **kwds):
            pools = map(tuple, args) * kwds.get('repeat', 1)
            result = [[]]
            for pool in pools:
                result = [x+[y] for x in result for y in pool]
                for prod in result:
                    yield tuple(prod)
        new_product = list(product(*itrs))
        return {'Var{}'.format(i + 1): [x[i]
                    for x in new_product]
                for i in range(len(itrs))}

#    expand_grid([1, 2, 3], [2, 1])

    def semiparametricAUC(response, treatment_group, input_covariates, data):
        assert response is not None, "Argument response is missing."
        assert treatment_group is not None, "Argument treatment_group is missing."
        assert input_covariates is not None, "Argument input_covariates is missing. Please put covariates as list. For e.g. ['x1','x2']"
        assert data is not None, "Argument data is missing. Please, specify name of pandas DataFrame."

        print("Data are being analyzed. Please, be patient.\n\n")

        d = DataFrame(data)
        group_covariates = treatment_group + input_covariates

        # split
        grouped_d = d.groupby(group_covariates)[response]

        keys = list((grouped_d.groups.keys()))

        def calculate_auc(ya, yb):
            m = len(ya)
            p = len(yb)
            I = numpy.zeros(shape=(m, p))
            for i in range(m):
                for j in range(p):
                    if ya[i] > yb[j]:
                        I[i, j] = 1
                    elif ya[i] == yb[j]:
                        I[i, j] = 0.5
                    else:
                        I[i, j] = 0
            #finv = lambda x: (-(numpy.log((1/x)-1)))
            auchat = numpy.mean(I)
            #finvhat = finv(auchat)
            vya = numpy.apply_along_axis(numpy.mean, 1, I)
            vyb = numpy.apply_along_axis(numpy.mean, 0, I)
            svarya = numpy.var(vya)
            svaryb = numpy.var(vyb)
            vhat_auchat = (svarya / m) + (svaryb / p)
            #v_finv_auchat = vhat_auchat/((auchat**2)*(1-auchat)**2)
            logitauchat = numpy.log(auchat / (1 - auchat))
            var_logitauchat = vhat_auchat / ((auchat**2) * (1 - auchat)**2)
            return([var_logitauchat, logitauchat])

        dict_df = {}
        auchat_container = {}
        my_card_1 = {}
        for i in range(len(keys)):
            # print(i)
            dict_df[i] = d.loc[grouped_d.groups[keys[i]]]

        for j in range(int(0.5 * len(dict_df))):
            # print(j)
            auchat_container[j], my_card_1[j] = (calculate_auc(dict_df[j].loc[:, response].tolist(
            ), dict_df[j + int(0.5 * len(dict_df))].loc[:, response].tolist()))

        var_logitauchat = [v for v in auchat_container.values()]
        gamma1 = [v for v in my_card_1.values()]

        # get levels
        df_keys = DataFrame(keys)
        df_keys.columns = group_covariates
        ds_only_covariates = df_keys[input_covariates]

        select_row = int(0.5 * len(ds_only_covariates))

        ds_expand = ds_only_covariates[:select_row]

        #ds_levels = {}
        # for i in input_covariates:
        #   ds_levels[i] = (d[i].cat.categories)

        #ds_expand = (DataFrame(expand_grid(*ds_levels.values())))

        def convert_to_factor(df):
            df = DataFrame(df)
            for i in df.columns:
                df[i] = df[i].astype('category')
            return(df)

        ds_expand = convert_to_factor(ds_expand)
        #ds_expand.columns = input_covariates

        var_list = '+'.join(input_covariates)

        #from patsy import *
        # model.matrix
        Z = (dmatrix(var_list, ds_expand))

        # get levels
        di = Z.design_info

        Z.columns = di.column_names
        #Z.sort('x2', ascending = [True])

        tau = numpy.diag([1 / i for i in var_logitauchat])

        #from numpy.linalg import inv
        ztauz = numpy.linalg.inv(Z.T.dot(tau).dot(Z))

        var_betas = numpy.diag(ztauz)
        std_error = numpy.sqrt(var_betas)
        betas = ztauz.dot(Z.T).dot(tau).dot(gamma1)

        #from scipy.stats import norm
        threshold = norm.ppf(0.975)

        lo = betas - threshold * std_error
        up = betas + threshold * std_error

        p_values = (norm.cdf(-numpy.abs(betas), loc=0, scale=std_error)) * 2

        results = DataFrame(numpy.vstack((betas, std_error, lo, up, p_values)).T)
        results.columns = ["Coefficients", "Std. Error", "2.5%", "97.5%", "Pr(>|z|)"]
        results.index = di.column_names
        print("\nModel Summary")
        return(results)