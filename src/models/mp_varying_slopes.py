from src.models.utils import load_data
import pymc3 as pm
import numpy as np
import pickle
from theano import tensor as tt


df, pmps, participants, mp_types, participant_to_idx, mp_type_to_idx, pmp_to_idx = load_data()


traces = {}
for label, d in df.groupby('mp_type'):
    print(f'train {label}')
    with pm.Model() as varyingslopes:
        sd_dist = pm.HalfCauchy.dist(beta=2)
        packed_chol = pm.LKJCholeskyCov('chol_cov', eta=2, n=2, sd_dist=sd_dist)
        chol = pm.expand_packed_triangular(2, packed_chol, lower=True)
        cov = pm.math.dot(chol, chol.T)
        sigma_ab = pm.Deterministic('sigma_ab', tt.sqrt(tt.diag(cov)))
        corr = tt.diag(sigma_ab**-1).dot(cov.dot(tt.diag(sigma_ab**-1)))
        r = pm.Deterministic('Rho', corr[np.triu_indices(2, k=1)])
        ab_bar = pm.Normal('ab_bar', mu=0, sd=10, shape=2)
        ab = pm.MvNormal('ab', mu=ab_bar, chol=chol, shape=(len(pmps), 2))
        x = d['partialMSE']
        p = 0.5 / (1 + np.exp(-(ab[d.id_pmp, 0] + (ab[d.id_pmp, 1]) * x)))
        s = pm.Bernoulli('s', p=p, observed=d['result'])
        trace_cov2 = pm.sample(1000, tune=1000, init='adapt_diag')
        traces[label] = trace_cov2

with open("data/processed/varyingslopes_trace2.pkl", "wb") as fh:
    pickle.dump((varyingslopes, traces), fh)
