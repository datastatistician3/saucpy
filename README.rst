.. role:: math(raw)
   :format: html latex
..

.. contents::
   :depth: 3
..

|Build Status|

Semi-parametric Area Under the Curve (sAUC) Regression
======================================================

Perform AUC analyses with discrete covariates and a semi-parametric
estimation

Model
-----

We consider applications that compare a response variable y between two
groups (A and B) while adjusting for k categorical covariates
:math:`X_1,X_2,...,X_k`. The response variable y is a continuous or
ordinal variable that is not normally distributed. Without loss of
generality, we assume each covariate is coded such that
:math:`X_i=1,...,n_i`,for :math:`i=1,...,k`. For each combination of the
levels of the covariates, we define the Area Under the ROC curve (AUC)
in the following way:

.. math:: \pi_{x_1 x_2...x_k}=P(Y^A>Y^B|X_1=x_1,X_2=x_2,...,X_k=x_k )+\frac{1}{2} P(Y^A=Y^B|X_1=x_1,X_2=x_2,...,X_k=x_k ),

where :math:`x_1=1,...,n_1,...,x_k=1,...,n_k`, and :math:`Y^A` and
:math:`Y^B` are two randomly chosen observations from Group A and B,
respectively. The second term in the above equation is for the purpose
of accounting ties.

For each covariate :math:`X_i`, without loss of generality, we use the
last category as the reference category and define (:math:`n_i-1`) dummy
variables :math:`X_i^{(1)},X_i^{(2)},...,X_i^{(n_i-1)}` such that

.. math::

   X_i^{(j)} (x)= \left\{\begin{array}
   {rrr}
   1, j = x \\
   0, j \ne x,
   \end{array}\right.

where :math:`i=1,...,k; j=1,...,n_i-1; x=1,...,n_i`. We model the
association between AUC :math:`\pi_{x_1 x_2...x_k}` and covariates using
a logistic model. Such a model specifies that the logit of
:math:`\pi_{x_1 x_2...x_k}` is a linear combination of terms that are
products of the dummy variables defined above. Specifically,

.. math:: logit(\pi_{x_1 x_2...x_k } )=Z_{x_1 x_2...x_k} \boldsymbol{\beta},

where :math:`Z_{x_1 x_2...x_k}` is a row vector whose elements are
zeroes or ones and are products of
:math:`X_1^{(1)} (x_1 ),...,X_1^{(n_i-1) } (x_1),...,X_k^{(1)} (x_k),...,X_k^{(n_k-1)} (x_k)`,
and :math:`\boldsymbol{\beta}` is a column vector of nonrandom unknown
parameters. Now, define a column vector :math:`\pi` by stacking up
:math:`\pi_{x_1 x_2...x_k}` and define a matrix Z by stacking up
:math:`Z_{x_1 x_2...x_k}`, as :math:`x_i` ranges from 1 to
:math:`n_i, i=1,...,k`, our final model is

.. math:: logit(\pi)=Z\boldsymbol{\beta} ...(1)

The reason for us to use a logit transformation of the AUC instead of
using the original AUC is for variance stabilization. We will illustrate
the above general model using examples.

Estimation
----------

First, we denote the number of observations with covariates
:math:`X_1=i_1,...,X_k=i_k` in groups A and B by :math:`N_{i_1...i_k}^A`
and :math:`N_{i_1...i_k}^B`, respectively. We assume both
:math:`N_{i_1...i_k}^A` and :math:`N_{i_1...i_k}^B` are greater than
zero in the following development. An unbiased estimator of
:math:`\pi_{i_1...i_k}` proposed by Mann and Whitney (1947) is

.. math:: \hat{\pi}_{i_1...i_k}=\frac{\sum_{l=1}^{N_{i_1...i_k}^A} \sum_{j=1}^{N_{i_1...i_k}^B} I_{lj}}{N_{i_1...i_k}^A N_{i_1...i_k}^B},

where

.. math::

   I_{i_1... i_k; lj}= \left\{\begin{array}
   {rrr}
   1, Y_{i_1...i_k; l}^A>Y_{i_1...i_k; j}^B \\
   \frac{1}{2}, Y_{i_1...i_k; l}^A=Y_{i_1...i_k; j}^B \\
   0, Y_{i_1...i_k; l}^A<Y_{i_1...i_k; j}^B
   \end{array}\right.

and :math:`Y_{i_1...i_k; l}^A` and :math:`Y_{i_1...i_k; j}^B` are
observations with :math:`X_1=i_1,...,X_k=i_k` in groups A and B,
respectively. Delong, Delong and Clarke-Pearson (1988) have shown that

.. math:: \hat{\pi}_{i_1...i_k} \approx N(\pi_{i_1...i_k},\sigma_{i_1...i_k}^2).

In order to obtain an estimator for :math:`\sigma_{i_1...i_k}^2`, they
first computed

.. math:: V_{i_1...i_k; l}^A=\frac{1}{N_{i_1...i_k}^B } \sum_{j=1}^{N_{i_1...i_k}^B} I_{lj},    l=1,...,N_{i_1...i_k}^A

and

.. math:: V_{i_1...i_k;j}^B=\frac{1}{N_{i_1...i_k}^A } \sum_{l=1}^{N_{i_1...i_k}^A} I_{lj},     j=1,...,N_{i_1...i_k}^B

Then, an estimate of the variance of the nonparametric AUC was

.. math:: \hat{\sigma}_{i_1...i_k}^2=\frac{(s_{i_1...i_k}^A )^2}{N_{i_1...i_k}^A} + \frac{(s_{i_1...i_k}^B )^2}{N_{i_1...i_k}^B},

where

:math:`(s_{i_1...i_k}^A )^2` and :math:`(s_{i_1...i_k}^B )^2` were the
sample variances of

:math:`V_{i_1...i_k; l}^A; l=1,...,N_{i_1...i_k}^A` and
:math:`V_{i_1...i_k; j}^B; j=1,...,N_{i_1...i_k}^B,` respectively.
Clearly, we need both :math:`N_{i_1...i_k}^A` and
:math:`N_{i_1...i_k}^B` are greater than two in order to compute
:math:`\hat{\sigma}_{i_1...i_k}^2`.

Now, in order to estimate parameters in Model (1), we first derive the
asymptotic variance of :math:`\hat{\gamma}_{i_1...i_k}` using the delta
method, which results in

.. math:: \hat{\gamma}_{i_1...i_k}=logit(\hat{\pi}_{i_1...i_k}) \approx N(logit(\pi_{i_1...i_k}),\tau_{i_1...i_k}^2),

where

.. math:: \hat{\tau}_{i_1...i_k}^2=\frac{\hat{\gamma}_{i_1...i_k}^2}{\hat{\pi}_{i_1...i_k}^2  (1-\hat{\pi}_{i_1...i_k})^2}

Rewriting the above model, we obtain

.. math:: \hat{\gamma}_{i_1...i_k}=logit(\pi_{i_1...i_k }) =Z_{i_1...i_k} \boldsymbol{\beta} + \epsilon_{i_1...i_k}

where,

:math:`\epsilon_{i_1,...,i_k} \approx N(0,\tau_{i_1,...,i_k}^2)`. Then,
by stacking up the :math:`\hat{\gamma}_{1_i,...,i_k}` to be
:math:`\hat{\gamma}, Z_{i_1...i_k}` to be :math:`\boldsymbol{Z}`, and
:math:`\epsilon_{i_1,...,i_k}` to be :math:`\boldsymbol{\epsilon}`, we
have

.. math:: \boldsymbol{\hat{\gamma}} =logit \boldsymbol{\hat{\pi}} = \boldsymbol{Z\beta + \epsilon},

where, :math:`E(\epsilon)=0` and
:math:`\hat{T}=Var(\epsilon)=diag(\hat{\tau}_{i_1... i_k}^2)` which is a
diagonal matrix. Finally, by using the generalized least squares method,
we estimate the parameters ÃƒÆ’Ã…Â½Ãƒâ€šÃ‚Â² and its variance-covariance matrix as
follows;

.. math:: \boldsymbol{\hat{\beta} ={(\hat{Z}^T  \hat{T}^{-1}  Z)}^{-1} Z^T  \hat{T}^{-1} \hat{\gamma}}

and

.. math:: \hat{V}(\boldsymbol{\hat{\beta}}) = \boldsymbol{{(\hat{Z}^T  \hat{T}^{-1}  Z)}^{-1}}

The above equations can be used to construct a 100(1-ÃƒÆ’Ã…Â½Ãƒâ€šÃ‚Â±)% Wald
confidence intervals for :math:`\boldsymbol{\beta_i}` using formula

.. math:: \hat{\beta}_i \pm Z_{1-\frac{\alpha}{2}} \sqrt{\hat{V}(\hat{\beta}_i)},

where :math:`Z_{1-\frac{\alpha}{2}}` is the
:math:`(1-\frac{\alpha}{2})^{th}` quantile of the standard normal
distribution. Equivalently, we reject

:math:`H_0:\beta_i = 0` if
:math:`|\hat{\beta}_i| > Z_{1-\frac{\alpha}{2}} \sqrt{\hat{V}(\hat{\beta}_i)},`

The p-value for testing :math:`H_0` is
:math:`2 * P(Z > |\hat{\beta}_i|/\sqrt{\hat{V}\hat{\beta}_i}),`

where Z is a random variable with the standard normal distribution.

Now, the total number of cells (combinations of covariates
:math:`X_1,...,X_k` is :math:`n_1 n_2...n_k`. As mentioned earlier, for
a cell to be usable in the estimation, the cell needs to have at least
two observations from Group A and two observations from Group B. As long
as the total number of usable cells is larger than the dimension of
:math:`\boldsymbol{\beta}`, then the matrix
:math:`{\boldsymbol{\hat{Z}^T \hat{T}^{-1} Z}}` is invertible and
consequently,\ :math:`\boldsymbol{\hat{\beta}}` is computable and model
(1) is identifiable.

.. |Build Status| image:: https://travis-ci.com/sbohora/sAUC.svg?token=shyYTzvvbsLRHsRAWXTg
   :target: https://travis-ci.com/sbohora/sAUC
