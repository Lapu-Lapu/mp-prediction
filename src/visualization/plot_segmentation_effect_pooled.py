import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import arviz as az
from src.globs import pp

with open('data/processed/segmentation_effect_trace_pooled.pkl', 'rb') as fh:
    model, trace, mp_type_to_idx = pickle.load(fh)
# with open('data/processed/twolevelsegmentation_trace2.pkl', 'rb') as fh:
#     model, trace = pickle.load(fh)
# mp_type_to_idx = {
#     'vcgpdm': 0, 'vgpdm': 1, 'tmp': 2, 'dmp': 3, 'mapgpdm': 4, 'mapcgpdm': 5
# }
a_bar = trace['a_bar']
# n_samples, mp_idx, occ = a_bar.shape
# mps = ['vcgpdm', 'vgpdm', 'tmp', 'dmp']
# for mp in mps:
#     i = mp_type_to_idx[mp]
#     print(mp, np.mean(np.diff(a_bar, axis=2) > 0, axis=0)[i])

import seaborn as sns
import pandas as pd
from itertools import product

X = 1/(1+np.exp(-a_bar[:]))
D = [pd.DataFrame({
        'y': X[:, occ],
        'mp': 'all',
        'occ': 'Yes' if occ==1 else 'No'
    }) for occ in [0, 1]]
df = pd.concat(D)
sns.violinplot(data=df, y="y", x='mp', hue="occ",
               split=True, inner="quart", linewidth=1,
               # )
               palette={"Yes": "b", "No": ".85"})
sns.despine(left=True)
plt.show()
