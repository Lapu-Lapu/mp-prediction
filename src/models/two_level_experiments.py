from src.models.utils import load_data, add_index
import pandas as pd
import numpy as np
import pickle
import pymc3 as pm


df, pmps, participants, mp_types, participant_to_idx, mp_type_to_idx, pmp_to_idx = load_data()
settings, setting_to_idx, df = add_index(df, 'setting')


with pm.Model() as multilevel:
    a_barbar = pm.Normal('a_barbar', 0, 1.5)
    sigma_abar = pm.Exponential('sigma_abar', 1)
    a_bar = pm.Normal('a_bar', a_barbar, sigma_abar, shape=(len(mp_types), 2))
    b_barbar = pm.Normal('b_barbar', 0, 1.5)
    sigma_bbar = pm.Exponential('sigma_bbar', 1)
    b_bar = pm.Normal('b_bar', b_barbar, sigma_bbar, shape=(len(mp_types), 2))
    sigma_b = pm.Exponential('sigma_b', 1, shape=(len(mp_types), 2))
    sigma_a = pm.Exponential('sigma_a', 1, shape=(len(mp_types), 2))
    a = pm.Normal('a', 0, 1, shape=(len(participants), 2))
    b = pm.Normal('b', 0, 1, shape=(len(participants), 2))
    x = df['partialMSE']
    c = pm.Beta('c', 2, 2, shape=(len(mp_types), 2))
    p = c[df.id_mp_type, df.id_setting] / (
        1 + np.exp(
            -(a_bar[df.id_mp_type, df.id_setting]
              + sigma_a[df.id_mp_type, df.id_setting] * a[df.id_participant, df.id_setting]
              + (b_bar[df.id_mp_type, df.id_setting]
                 + sigma_b[df.id_mp_type, df.id_setting] * b[df.id_participant, df.id_setting]
                 ) * x
              )
        )
    )
    s = pm.Bernoulli('s', p=p, observed=df['result'])
    trace_multilevel = pm.sample(1000, tune=1000, init='adapt_diag')

with open("data/processed/twolevelexperiments_trace.pkl", "wb") as fh:
    pickle.dump((multilevel, trace_multilevel), fh)

trace_multilevel['b_bar'].mean(axis=0)
