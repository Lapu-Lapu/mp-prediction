from src.models.utils import load_data
import pymc3 as pm
import numpy as np
import pickle
from theano import tensor as tt


df, pmps, participants, mp_types, participant_to_idx, mp_type_to_idx, pmp_to_idx = load_data('data/processed/processed_data_online.json')


with pm.Model() as varyingslopes:
    sd_dist = pm.HalfCauchy.dist(beta=2)
    packed_chol = pm.LKJCholeskyCov('chol_cov', eta=2, n=2, sd_dist=sd_dist)
    chol = pm.expand_packed_triangular(2, packed_chol, lower=True)

    sd_dist2 = pm.HalfCauchy.dist(beta=2)
    packed_chol2 = pm.LKJCholeskyCov('chol_cov2', eta=2, n=2, sd_dist=sd_dist2)
    chol2 = pm.expand_packed_triangular(2, packed_chol2, lower=True)

    cov = pm.math.dot(chol, chol.T)
    sigma_ab = pm.Deterministic('sigma_ab', tt.sqrt(tt.diag(cov)))
    corr = tt.diag(sigma_ab**-1).dot(cov.dot(tt.diag(sigma_ab**-1)))
    r = pm.Deterministic('Rho', corr[np.triu_indices(2, k=1)])
    ab_barbar = pm.Normal('ab_barbar', mu=0, sd=10, shape=2)
    ab_bar = pm.MvNormal('ab_bar', mu=ab_barbar, chol=chol, shape=(len(mp_types), 2))
    ab_std = pm.Normal('ab_std', 0, 1, shape=(2, len(participants)))
    ab = ab_bar[df.id_mptype] + tt.sum(chol2 * ab_std[:, df.id_participant])
    x = df['partialMSE']
    p = 0.5 / (1 + np.exp(-(ab[0] + ab[1] * x)))
    s = pm.Bernoulli('s', p=p, observed=df['result'])
    trace_cov2 = pm.sample(1000, tune=1000, init='adapt_diag')

with open("data/processed/varyingslopes_trace2.pkl", "wb") as fh:
    pickle.dump((varyingslopes, trace_cov), fh)
