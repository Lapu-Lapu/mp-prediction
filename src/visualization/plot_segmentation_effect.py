import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import arviz as az
from src.globs import pp

with open('data/processed/segmentation_effect_trace2.pkl', 'rb') as fh:
    model, trace, mp_type_to_idx = pickle.load(fh)
# with open('data/processed/twolevelsegmentation_trace2.pkl', 'rb') as fh:
#     model, trace = pickle.load(fh)
# mp_type_to_idx = {
#     'vcgpdm': 0, 'vgpdm': 1, 'tmp': 2, 'dmp': 3, 'mapgpdm': 4, 'mapcgpdm': 5
# }
a_bar = trace['a_bar']
n_samples, mp_idx, occ = a_bar.shape
print(np.mean(np.diff(a_bar, axis=2) > 0, axis=0))

import seaborn as sns
import pandas as pd
from itertools import product

X = 1/(1+np.exp(-a_bar[:]))
D = [pd.DataFrame({
        'y': X[:, mp_type_to_idx[mp], occ],
        'mp': mp,
        'occ': 'Yes' if occ==1 else 'No'
    }) for mp, occ in product(['tmp', 'dmp', 'vcgpdm', 'vgpdm'], [0, 1])]
df = pd.concat(D)
sns.violinplot(data=df, x="mp", y="y", hue="occ",
               split=True, inner="quart", linewidth=1,
               palette={"Yes": "b", "No": ".85"})
sns.despine(left=True)
plt.show()
# pm.forestplot(trace,
#               var_names=['a_bar'],
#               combined=True,
#               transform=lambda x: 1/(1+np.exp(-x)))
# plt.title(f"{mp_type_to_idx}")
# plt.savefig('../SFBTRR135experiments/natMPs/talks/SAP2020_presentation/media/segmentation_effect.pdf')
# plt.show()

# fig, ax = plt.subplots()
# for mp in ['tmp', 'dmp', 'vcgpdm', 'vgpdm']:
#     idx = mp_type_to_idx[mp]
#     ax = az.plot_kde(1/(1+np.exp(-trace['a_bar'][:, idx])), label=pp[mp],
#                      # plot_kwargs={'color': mp_color[mp]},
#                      ax=ax,
#                      rug=True, rotated=True)
# ax.set_title('MP Confusion Rate')
# ax.set_xticklabels([])
# ax.set_ylabel("Confusion Rate")
# # plt.savefig('../SFBTRR135experiments/natMPs/talks/SAP2020_presentation/media/mp_ranking.svg')
# plt.show()
