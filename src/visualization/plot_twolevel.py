import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from src.models.utils import beta_std, center_data
from src.models.utils import load_data


df, pmps, participants, mp_types, participant_to_idx, mp_type_to_idx, pmp_to_idx = load_data()

mp_to_indices = {mp: [idx for s, idx in pmp_to_idx.items() if s.split(';')[1] == mp] for mp in mp_types}

with open("data/processed/twolevel_trace.pkl", "rb") as fh:
    model, trace = pickle.load(fh)

fig, ax = plt.subplots()
for label, d in df.groupby('mp_type'):
    if label in ['mapgpdm', 'mapcgpdm']:
        continue
    # if label not in ['tmp']:
    #     continue
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
    xpred = np.linspace(0, 2)
    for i in range(30):
        i = np.random.choice(range(2000))
        mp_indices = mp_to_indices[label]
        a = trace['a_bar'][i, mp_type_to_idx[label]]
        b = trace['b_bar'][i, mp_type_to_idx[label]]
        c = trace['c'][i, mp_type_to_idx[label]]
        ypred = c / (1 + np.exp(-(a + b * xpred)))
        ax.plot(xpred, ypred, alpha=0.05, color=color)
plt.legend(loc='upper right')
plt.ylim(0.09, 0.61)
plt.xlim(-0.01, 1.95)
plt.title(f'MP-Model Confusion')
plt.ylabel('Confusion Rate')
plt.xlabel('MSE')
plt.savefig('../SFBTRR135experiments/natMPs/talks/SAP2020_presentation/media/multilevel.svg')
plt.show()
