#!/usr/bin/env python
# coding: utf-8

# DO NOT EDIT
# Autogenerated from the notebook distributed_estimation.ipynb.
# Edit the notebook and then sync the output with this file.
#
# flake8: noqa
# DO NOT EDIT

# # Distributed Estimation
#
# This notebook goes through a couple of examples to show how to use
# `distributed_estimation`.  We import the `DistributedModel` class and make
# the exog and endog generators.

import numpy as np
from scipy.stats.distributions import norm
from statsmodels.base.distributed_estimation import DistributedModel


def _exog_gen(exog, partitions):
    """partitions exog data"""

    n_exog = exog.shape[0]
    n_part = np.ceil(n_exog / partitions)

    ii = 0
    while ii < n_exog:
        jj = int(min(ii + n_part, n_exog))
        yield exog[ii:jj, :]
        ii += int(n_part)


def _endog_gen(endog, partitions):
    """partitions endog data"""

    n_endog = endog.shape[0]
    n_part = np.ceil(n_endog / partitions)

    ii = 0
    while ii < n_endog:
        jj = int(min(ii + n_part, n_endog))
        yield endog[ii:jj]
        ii += int(n_part)


# Next we generate some random data to serve as an example.

X = np.random.normal(size=(1000, 25))
beta = np.random.normal(size=25)
beta *= np.random.randint(0, 2, size=25)
y = norm.rvs(loc=X.dot(beta))
m = 5

# This is the most basic fit, showing all of the defaults, which are to
# use OLS as the model class, and the debiasing procedure.

debiased_OLS_mod = DistributedModel(m)
debiased_OLS_fit = debiased_OLS_mod.fit(zip(_endog_gen(y, m), _exog_gen(X, m)),
                                        fit_kwds={"alpha": 0.2})

# Then we run through a slightly more complicated example which uses the
# GLM model class.

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Gaussian

debiased_GLM_mod = DistributedModel(m,
                                    model_class=GLM,
                                    init_kwds={"family": Gaussian()})
debiased_GLM_fit = debiased_GLM_mod.fit(zip(_endog_gen(y, m), _exog_gen(X, m)),
                                        fit_kwds={"alpha": 0.2})

# We can also change the `estimation_method` and the `join_method`.  The
# below example show how this works for the standard OLS case.  Here we
# using a naive averaging approach instead of the debiasing procedure.

from statsmodels.base.distributed_estimation import _est_regularized_naive, _join_naive

naive_OLS_reg_mod = DistributedModel(m,
                                     estimation_method=_est_regularized_naive,
                                     join_method=_join_naive)
naive_OLS_reg_params = naive_OLS_reg_mod.fit(zip(_endog_gen(y, m),
                                                 _exog_gen(X, m)),
                                             fit_kwds={"alpha": 0.2})

# Finally, we can also change the `results_class` used.  The following
# example shows how this work for a simple case with an unregularized model
# and naive averaging.

from statsmodels.base.distributed_estimation import (
    _est_unregularized_naive,
    DistributedResults,
)

naive_OLS_unreg_mod = DistributedModel(
    m,
    estimation_method=_est_unregularized_naive,
    join_method=_join_naive,
    results_class=DistributedResults,
)
naive_OLS_unreg_params = naive_OLS_unreg_mod.fit(zip(_endog_gen(y, m),
                                                     _exog_gen(X, m)),
                                                 fit_kwds={"alpha": 0.2})