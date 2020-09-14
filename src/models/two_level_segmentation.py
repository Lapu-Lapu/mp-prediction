from src.models.utils import load_data
import numpy as np
import pickle
import pymc3 as pm


df, pmps, participants, mp_types, participant_to_idx, mp_type_to_idx, pmp_to_idx = load_data('data/processed/processed_online_data.json')

df['occ'] = df['occluded_contact'].astype(int)
print(df.shape)
df = df[df.movement.apply(lambda m: m in ['pass-bottle', 'return-bottle'])]
print(df.shape)
with pm.Model() as multilevel:
    # a_barbar = pm.Normal('a_barbar', 0, 1.5)
    # sigma_abar = pm.Exponential('sigma_abar', 1)
    # a_bar = pm.Normal('a_bar', a_barbar, sigma_abar, shape=(len(mp_types), 2))
    a_bar = pm.Normal('a_bar', 0, 1.5, shape=(len(mp_types), 2))
    # b_barbar = pm.Normal('b_barbar', 0, 1.5)
    # sigma_bbar = pm.Exponential('sigma_bbar', 1)
    # b_bar = pm.Normal('b_bar', b_barbar, sigma_bbar, shape=(len(mp_types), 2))
    b_bar = pm.Normal('b_bar', 0, 1.5, shape=(len(mp_types), 2))
    sigma_b = pm.Exponential('sigma_b', 1, shape=(len(mp_types), 2))
    # c_bar = pm.Normal('c_bar', 0, 1.5)
    # sigma_c = pm.Exponential('sigma_c', 1)
    sigma_a = pm.Exponential('sigma_a', 1, shape=(len(mp_types), 2))
    a = pm.Normal('a', 0, 1, shape=(len(participants), 2))
    b = pm.Normal('b', 0, 1, shape=(len(participants), 2))
    # c = pm.Normal('c', 0, 1, shape=len(mp_types))
    # x = d['partialMSE'] - d['partialMSE'].mean()
    # x = x / x.max()
    x = df['partialMSE']
    c = pm.Beta('c', 2, 2, shape=(len(mp_types), 2))
    p = c[df.id_mp_type, df.occ] / (1 + np.exp(-(a_bar[df.id_mp_type, df.occ] + sigma_a[df.id_mp_type, df.occ] * a[df.id_participant, df.occ] + (b_bar[df.id_mp_type, df.occ] + sigma_b[df.id_mp_type, df.occ] * b[df.id_participant, df.occ]) * x)))
    # p = 0.5 / (1 + np.exp(-(a_bar + sigma_a* a[df.id_participant] + sigma_c*c[df.id_mp_type] + (b_bar + sigma_b * b[df.id_mp_type]) * x)))
    s = pm.Bernoulli('s', p=p, observed=df['result'])
    trace_multilevel = pm.sample(2000, tune=1000, init='adapt_diag')

with open("data/processed/twolevelsegmentation_trace2.pkl", "wb") as fh:
    pickle.dump((multilevel, trace_multilevel), fh)

trace_multilevel['b_bar'].mean(axis=0)
