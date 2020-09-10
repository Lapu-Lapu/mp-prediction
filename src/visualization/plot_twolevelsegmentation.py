import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from src.models.utils import beta_std, center_data
from src.models.utils import load_data


df, pmps, participants, mp_types, participant_to_idx, mp_type_to_idx, pmp_to_idx = load_data('data/processed/processed_online_data.json')

# occ == 1 if contact occured during occlusion
df['occ'] = df['occluded_contact'].astype(int)

mp_to_indices = {mp: [idx for s, idx in pmp_to_idx.items() if s.split(';')[1] == mp] for mp in mp_types}

with open("data/processed/twolevelsegmentation_trace.pkl", "rb") as fh:
    model, trace = pickle.load(fh)
mp_type_to_idx = {'vcgpdm': 0, 'vgpdm': 1, 'tmp': 2, 'dmp': 3, 'mapgpdm': 4, 'mapcgpdm': 5}


mp_color = {
    'dmp': '#1f77b4',
    'tmp': '#2ca02c',
    'vcgpdm': '#9467bd',
    'vgpdm': '#e377c2'
}
occ_fmt = {
    0: 'x',
    1: '+'
}

fig, ax = plt.subplots()
for label, d in df.groupby('mp_type'):
    if label in ['mapgpdm', 'mapcgpdm']:
        continue
    # if label not in ['tmp']:
    #     continue
    d['X'] = d['partialMSE']
    for occ, D in d.groupby('occluded_contact'):
        gb = D.groupby('model')
        x = np.array(gb.X.mean())
        xerr = np.array(gb.X.sem())
        y = np.array(gb.result.mean())

        yerr = np.array(gb.result.agg(beta_std))
        p = ax.errorbar(x,
                        y,
                        xerr=xerr,
                        yerr=yerr,
                        alpha=0.7,
                        elinewidth=0.2,
                        label=label if occ else None,
                        color=mp_color[label],
                        fmt=occ_fmt[occ])
        # color = p[0].get_color()
        xpred = np.linspace(0, 2)
        for occ in range(2):
            i = np.random.choice(range(2000))
            mp_indices = mp_to_indices[label]
            a = trace['a_bar'].mean(axis=0)[mp_type_to_idx[label], occ]
            b = trace['b_bar'].mean(axis=0)[mp_type_to_idx[label], occ]
            c = trace['c'].mean(axis=0)[mp_type_to_idx[label], occ]
            ypred = c / (1 + np.exp(-(a + b * xpred)))
            fmt = '--' if occ == 0 else '-'
            ax.plot(xpred, ypred, alpha=1, color=mp_color[label], linestyle=fmt)
plt.legend(loc='upper right')
plt.ylim(0, 0.65)
plt.xlim(0, 1.4)
plt.title(f'Effect of Contact on Perception')
plt.ylabel('Confusion Rate')
plt.xlabel('MSE')
plt.savefig('../SFBTRR135experiments/natMPs/talks/SAP2020_presentation/media/segmentation.svg')
plt.show()

# TODO: compute posterior of ypred[no occ] > ypred[occ] in xrange of data.
