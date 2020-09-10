import pandas as pd
import numpy as np
import pickle
import pymc3 as pm


df, pmps, participants, mp_types, participant_to_idx, mp_type_to_idx, pmp_to_idx = load_data()

with pm.Model() as multilevel:
    a_bar = pm.Normal('a_bar', 0, 1.5)
    sigma_a = pm.Exponential('sigma_a', 1)
    b_bar = pm.Normal('b_bar', 0, 1.5)
    sigma_b = pm.Exponential('sigma_b', 1)
    # c_bar = pm.Normal('c_bar', 0, 1.5)
    # sigma_c = pm.Exponential('sigma_c', 1)
    a = pm.Normal('a', 0, 1, shape=len(pmps))
    b = pm.Normal('b', 0, 1, shape=len(pmps))
    # c = pm.Normal('c', 0, 1, shape=len(mp_types))
    # x = d['partialMSE'] - d['partialMSE'].mean()
    # x = x / x.max()
    x = df['partialMSE']
    p = 0.5 / (1 + np.exp(-(a_bar + sigma_a * a[df.id_pmp] + (b_bar + sigma_b * b[df.id_pmp]) * x)))
    # p = 0.5 / (1 + np.exp(-(a_bar + sigma_a* a[df.id_participant] + sigma_c*c[df.id_mp_type] + (b_bar + sigma_b * b[df.id_mp_type]) * x)))
    s = pm.Bernoulli('s', p=p, observed=df['result'])
    trace_multilevel = pm.sample(1000, tune=1000, init='adapt_diag')

traces = {}
for mp_type, d in df.groupby('mp_type'):
    if mp_type in ['mapgpdm', 'mapcgpdm']:
        continue
    with pm.Model() as logistic_model:
        a = pm.Normal('a', 0, 10)
        b = pm.Normal('b', 0, 10)
        # x = d['partialMSE'] - d['partialMSE'].mean()
        # x = x / x.max()
        x = d['partialMSE']
        p = 0.5 / (1 + np.exp(-(a + b * x)))
        s = pm.Bernoulli('s', p=p, observed=d['result'])
        trace = pm.sample(1000, tune=1000, init='adapt_diag')
    traces[mp_type] = trace

with pm.Model() as mult_mixed:
    a_bar = pm.Normal('a_bar', 0, 1.5)
    sigma_a = pm.Exponential('sigma_a', 1)
    b_bar = pm.Normal('b_bar', 0, 1.5)
    sigma_b = pm.Exponential('sigma_b', 1)
    # c_bar = pm.Normal('c_bar', 0, 1.5)
    sigma_c = pm.Exponential('sigma_c', 1)
    a = pm.Normal('a', 0, 1, shape=len(participants))
    b = pm.Normal('b', b_bar, sigma_b, shape=len(mp_types))
    c = pm.Normal('c', 0, 1, shape=len(mp_types))
    # x = d['partialMSE'] - d['partialMSE'].mean()
    # x = x / x.max()
    x = df['partialMSE']
    p = 0.5 / (1 + np.exp(-(a_bar + sigma_a * a[df.id_participant] + sigma_c * c[df.id_mp_type] + b[df.id_mp_type] * x)))
    # p = 0.5 / (1 + np.exp(-(a_bar + sigma_a* a[df.id_participant] + sigma_c*c[df.id_mp_type] + (b_bar + sigma_b * b[df.id_mp_type]) * x)))
    s = pm.Bernoulli('s', p=p, observed=df['result'])
    trace_multimixed = pm.sample(1000, tune=1000, init='adapt_diag')
