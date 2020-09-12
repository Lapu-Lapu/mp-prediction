import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np
import pickle

with open('data/processed/segmentation_effect_trace.pkl', 'rb') as fh:
    model, trace, mp_type_to_idx = pickle.load(fh)

a_bar = trace['a_bar']
n_samples, mp_idx, occ = a_bar.shape
print(np.mean(np.diff(a_bar, axis=2) > 0, axis=0))
pm.forestplot(trace, var_names=['a_bar'])
plt.title(f"{mp_type_to_idx}")
plt.show()
