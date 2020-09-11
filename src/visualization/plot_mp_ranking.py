import pickle
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from src.models.utils import load_data

df, pmps, participants, mp_types, participant_to_idx, mp_type_to_idx, pmp_to_idx = load_data()
# idx_to_mp_type = {v, k for k, v in mp_type_to_idx.items()}

with open("data/processed/rank_mps_trace.pkl", "rb") as fh:
    model, trace = pickle.load(fh)

mp_color = {
    'dmp': '#1f77b4',
    'tmp': '#2ca02c',
    'vcgpdm': '#9467bd',
    'vgpdm': '#e377c2',
    'mapcgpdm': 'yellow',
    'mapgpdm': 'purple'
}


fig, ax = plt.subplots()
for mp in ['tmp', 'dmp', 'vcgpdm', 'vgpdm']:
    idx = mp_type_to_idx[mp]
    ax = az.plot_kde(1/(1+np.exp(-trace['a_bar'][:, idx])), label=mp,
                     plot_kwargs={'color': mp_color[mp]}, ax=ax,
                     rug=True, rotated=True)
ax.set_title('MP Confusion Rate')
ax.set_xticklabels([])
plt.show()
