import pandas as pd
import numpy as np
import pickle
import pymc3 as pm

df_vr = pd.read_json('data/processed/processed_data_vr.json')
print('vr data:', df_vr.shape)

df_online = pd.read_json('data/processed/processed_data_online.json')
print('online data:', df_online.shape)

df = pd.concat([df_vr, df_online], join='inner')
print('data:', df.shape)

traces = {}
for mp_type, d in df.groupby('mp_type'):
    if mp_type in ['mapgpdm', 'mapcgpdm']:
        continue
    with pm.Model() as logistic_model:
        a = pm.Normal('a', 0, 10)
        b = pm.Normal('b', 0, 10)
        x = d['partialMSE'] - d['partialMSE'].mean()
        x = x / x.max()
        p = 1 / (1 + np.exp(-(a + b * x)))
        s = pm.Bernoulli('s', p=p, observed=d['result'])
        trace = pm.sample(1000, tune=1000, init='adapt_diag')
    traces[mp_type] = trace

with open("data/processed/logreg_traces.pkl", "wb") as fh:
    pickle.dump(traces, fh)
