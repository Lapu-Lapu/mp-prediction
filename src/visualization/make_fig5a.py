import pandas as pd
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import pickle

from src.data.utils import trial_id, process_data_dict
from src.data.utils import _parse_params, open_pkls
from src.globs import beta_std

df = pd.read_json('data/processed/processed_data_vr.json')

participant_id = df.participant.unique()
id_to_idx = dict(zip(participant_id, range(len(participant_id))))
df['pid'] = df.participant.apply(lambda id: id_to_idx[id])

D2 = open_pkls('data/raw/online/scores/combined_errors.pkl')
D2 = process_data_dict(D2)
D2 = D2.set_index('training_id')

U = set(df.trial_id).intersection(D2.index)
df = df[df['trial_id'].apply(lambda x: x in U)]

for idx in df.index:
    trial_id = df.loc[idx, 'trial_id']
    val = D2.loc[trial_id, "continuation_T"] * 60 / D2.loc[trial_id,
                                                           "continuation_MSE"]
    df.loc[idx, 'invMSE'] = val
    df.loc[idx, 'partialMSE'] = 1 / val

df['X'] = (df.partialMSE - df.partialMSE.mean()) / (df.partialMSE.max() -
                                                    df.partialMSE.mean())

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

df['model'] = df.apply(_parse_params, axis=1)

group = 'test_part'
fig, ax = plt.subplots()
xlim = (-0.03, 0.1)
for i, (label, d) in enumerate(df.groupby('mp_type')):
    if label in ['mapgpdm', 'mapcgpdm']:
        continue
    gb = d.groupby('model')
    x = np.array(gb.X.mean())
    xerr = np.array(gb.X.sem())
    y = np.array(gb.result.mean())
    yerr = np.array(gb.result.agg(beta_std))
    p = ax.errorbar(x,
                    y,
                    xerr=xerr,
                    yerr=yerr,
                    fmt='.',
                    alpha=0.7,
                    label=label)
    color = p[0].get_color()

    trace = traces[label]
    xpred = np.linspace(*xlim)
    for i in range(20):
        i = np.random.choice(range(2000))
        a, b = trace['a'][i], trace['b'][i]
        ypred = 1 / (1 + np.exp(-(a + b * xpred)))
        ax.plot(xpred, ypred, alpha=0.1, color=color)
plt.ylim((0.1, 0.55))
plt.xlim(xlim)
plt.legend(loc='upper right')
plt.title('MP-Model Confusion - VR')
plt.ylabel('Confusion Rate')
plt.xlabel('Centered MSE')
plt.savefig('reports/figures/fig5a.pdf')
