# -*- coding: utf-8 -*-
'''
Provides the :class:`SAUC <sAUCpy.sAUCpy.SAUC>` class, a semi-parametric AUC method``.
'''
from numpy import *

class sAUCpy(object):
    """
    Semi-parametric Area Under the Curve (AUC) Regreesion model.

    Parameters
    ----------
    fit_intercept : boolean, optional
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).

    normalize : boolean, optional, default False
        If True, the regressors X will be normalized before regression.
        This parameter is ignored when `fit_intercept` is set to False.
        When the regressors are normalized, note that this makes the
        hyperparameters learnt more robust and almost independent of the number
        of samples. The same property is not valid for standardized data.
        However, if you wish to standardize, please use
        `preprocessing.StandardScaler` before calling `fit` on an estimator
        with `normalize=False`.

    copy_X : boolean, optional, default True
        If True, X will be copied; else, it may be overwritten.

    n_jobs : int, optional, default 1
        The number of jobs to use for the computation.
        If -1 all CPUs are used. This will only provide speedup for
        n_targets > 1 and sufficient large problems.

    Attributes
    ----------
    coef_ : array, shape (n_features, ) or (n_targets, n_features)
        Estimated coefficients for the linear regression problem.
        If multiple targets are passed during the fit (y 2D), this
        is a 2D array of shape (n_targets, n_features), while if only
        one target is passed, this is a 1D array of length n_features.

    residues_ : array, shape (n_targets,) or (1,) or empty
        Sum of residuals. Squared Euclidean 2-norm for each target passed
        during the fit. If the linear regression problem is under-determined
        (the number of linearly independent rows of the training matrix is less
        than its number of linearly independent columns), this is an empty
        array. If the target vector passed during the fit is 1-dimensional,
        this is a (1,) shape array.

        .. versionadded:: 0.18

    intercept_ : array
        Independent term in the linear model.

    Notes
    -----
    From the implementation point of view, this is just plain Ordinary
    Least Squares (scipy.linalg.lstsq) wrapped as a predictor object.

    """

    def __init__(self):
        print
        "This is init."

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
    
    ya = [3,2,4,3]
    yb = [1,2,2,3]
    
    calculate_auc(ya=ya, yb= yb)




