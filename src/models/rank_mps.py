from src.models.utils import load_data
import pandas as pd
import numpy as np
import pickle
import pymc3 as pm


df, pmps, participants, mp_types, participant_to_idx, mp_type_to_idx, pmp_to_idx = load_data()

with pm.Model() as multilevel:
    a_barbar = pm.Normal('a_barbar', 0, 1.5)
    sigma_abar = pm.Exponential('sigma_abar', 1)
    a_bar = pm.Normal('a_bar', a_barbar, sigma_abar, shape=len(mp_types))
    sigma_a = pm.Exponential('sigma_a', 1, shape=len(mp_types))
    a = pm.Normal('a', 0, 1, shape=len(participants))
    x = df['partialMSE']
    p = 1 / (1 + np.exp(-(a_bar[df.id_mp_type] + sigma_a[df.id_mp_type] * a[df.id_participant])))
    s = pm.Bernoulli('s', p=p, observed=df['result'])
    trace_multilevel = pm.sample(4000, tune=1000, init='adapt_diag', cores=8)

with open("data/processed/rank_mps_trace.pkl", "wb") as fh:
    pickle.dump((multilevel, trace_multilevel), fh)
