import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from src.models.utils import beta_std, center_data
from src.models.utils import load_data


df, pmps, participants, mp_types, participant_to_idx, mp_type_to_idx, pmp_to_idx = load_data()

mp_to_indices = {mp: [idx for s, idx in pmp_to_idx.items() if s.split(';')[1] == mp] for mp in mp_types}

with open("data/processed/varyingslopes_trace2.pkl", "rb") as fh:
    model, traces = pickle.load(fh)

fig, ax = plt.subplots()
for label, d in df.groupby('mp_type'):
    if label in ['mapgpdm', 'mapcgpdm']:
        continue
    d['X'] = d['partialMSE']
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
    xpred = np.linspace(x.min(), x.max())
    for i in range(30):
        i = np.random.choice(range(2000))
        trace = traces[label]
        mp_indices = mp_to_indices[label]
        ab = trace['ab_bar'][i]
        ypred = 0.5 / (1 + np.exp(-(ab[0] + ab[1] * xpred)))
        ax.plot(xpred, ypred, alpha=0.05, color=color)
plt.legend(loc='upper right')
plt.title(f'MP-Model Confusion: {label}')
plt.ylabel('Confusion Rate')
plt.xlabel('MSE')
plt.show()
