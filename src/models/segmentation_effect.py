from src.models.utils import load_data
import numpy as np
import pickle
import pymc3 as pm


df, pmps, participants, mp_types, participant_to_idx, mp_type_to_idx, pmp_to_idx = load_data('data/processed/processed_data_online.json')
# df, pmps, participants, mp_types, participant_to_idx, mp_type_to_idx, pmp_to_idx = load_data()


print(df.shape)
df = df[df.movement.apply(
    lambda m: m in ['pass-bottle', 'return-bottle'])]
df['occ'] = df['occluded_contact'].astype(int)
print(df.shape)
# n_occ = np.sum(df.occ)
# df = df.loc[np.random.choice(df[df.occ==0].index, n_occ)]
print(df.shape)

with pm.Model() as multilevel:
    a_bar = pm.Normal('a_bar', 0, 1.5, shape=(len(mp_types), 2))
    sigma_a = pm.Exponential('sigma_a', 1, shape=(len(mp_types), 2))
    a = pm.Normal('a', 0, 1, shape=(len(participants), 2))
    p = 1 / (1 + np.exp(
        -(a_bar[df.id_mp_type, df.occ] + (sigma_a[df.id_mp_type, df.occ]
                                          * a[df.id_participant, df.occ]))
    ))
    s = pm.Bernoulli('s', p=p, observed=df['result'])
    trace_multilevel = pm.sample(1000, tune=1000, init='adapt_diag', cores=7)

with open("data/processed/segmentation_effect_trace4.pkl", "wb") as fh:
    pickle.dump((multilevel, trace_multilevel, mp_type_to_idx), fh)
