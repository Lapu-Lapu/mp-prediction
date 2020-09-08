import pandas as pd
import numpy as np
import pickle
import pymc3 as pm

def key_to_idx(keys):
    return dict(zip(keys, range(len(keys))))

# df = pd.read_json('data/processed/processed_data.json')
df = pd.read_json('data/processed/processed_data_online.json')

df['participant'] = df.participant.apply(str)
df['pmp'] = df.apply(lambda row: ';'.join([row.participant, row.mp_type]), axis=1)
pmps = df.pmp.unique()
participants = df.participant.unique()
mp_types = df.mp_type.unique()
participant_to_idx = key_to_idx(participants)
mp_type_to_idx = key_to_idx(mp_types)
pmp_to_idx = key_to_idx(pmps)

df['id_mptype'] = df.mp_type.apply(lambda x: mp_type_to_idx[x])
df['id_participant'] = df.participant.apply(lambda x: participant_to_idx[x])
df['id_pmp'] = df.pmp.apply(lambda x: pmp_to_idx[x])

with pm.Model() as logistic_model:
    a_bar = pm.Normal('a_bar', 0, 1.5)
    sigma_a = pm.Exponential('sigma_a', 1)
    b_bar = pm.Normal('b_bar', 0, 1.5)
    sigma_b = pm.Exponential('sigma_b', 1)
    # c_bar = pm.Normal('c_bar', 0, 1.5)
    # sigma_c = pm.Exponential('sigma_c', 1)
    a = pm.Normal('a', 0, 1, shape=len(pmps))
    b = pm.Normal('b', b_bar, sigma_b, shape=len(pmps))
    # c = pm.Normal('c', 0, 1, shape=len(mp_types))
    # x = d['partialMSE'] - d['partialMSE'].mean()
    # x = x / x.max()
    x = df['partialMSE']
    p = 0.5 / (1 + np.exp(-(a_bar + sigma_a * a[df.id_participant] + b[df.id_mptype] * x)))
    # p = 0.5 / (1 + np.exp(-(a_bar + sigma_a* a[df.id_participant] + sigma_c*c[df.id_mptype] + (b_bar + sigma_b * b[df.id_mptype]) * x)))
    s = pm.Bernoulli('s', p=p, observed=df['result'])
    trace = pm.sample(1000, tune=1000, init='adapt_diag')

with open("data/processed/multilevel_trace.pkl", "wb") as fh:
    pickle.dump(trace, fh)
