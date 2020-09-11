from src.models.utils import load_data, add_index
import numpy as np
import pickle
import pymc3 as pm


df, pmps, participants, mp_types, participant_to_idx, mp_type_to_idx, pmp_to_idx = load_data()
settings, setting_to_idx, df = add_index(df, 'setting')

df['occ'] = df['occluded_contact'].astype(int)
print(mp_type_to_idx)
breakpoint()

with pm.Model() as multilevel:
    a_bar = pm.Normal('a_bar', 0, 1.5, shape=(len(mp_types), 2))
    sigma_a = pm.Exponential('sigma_a', 1, shape=(len(mp_types), 2))
    a = pm.Normal('a', 0, 1, shape=(len(participants), 2, 2))
    p = 1 / (1 + np.exp(-(a_bar[df.id_mp_type, df.occ] + sigma_a[df.id_mp_type, df.occ] * a[df.id_participant, df.occ, df.id_setting])))
    # p = 0.5 / (1 + np.exp(-(a_bar + sigma_a* a[df.id_participant] + sigma_c*c[df.id_mp_type] + (b_bar + sigma_b * b[df.id_mp_type]) * x)))
    s = pm.Bernoulli('s', p=p, observed=df['result'])
    trace_multilevel = pm.sample(2000, tune=1000, init='adapt_diag')

with open("data/processed/segmentation_effect_trace.pkl", "wb") as fh:
    pickle.dump((multilevel, trace_multilevel), fh)

# trace_multilevel['b_bar'].mean(axis=0)
