import pymc3 as pm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import arviz as az
from src.globs import pp

with open('data/processed/segmentation_effect_trace_tmpvsdyn.pkl', 'rb') as fh:
    model, trace, mp_type_to_idx = pickle.load(fh)
a_bar = trace['a_bar']

import seaborn as sns
import pandas as pd
from itertools import product

X = 1/(1+np.exp(-a_bar[:]))
D = [pd.DataFrame({
    'y': X[:, is_dyn, occ],
    'mp': "dyn" if is_dyn ==1 else "tmp",
    'occ': 'Yes' if occ==1 else 'No'
    }) for is_dyn, occ in product([0, 1], [0, 1])]
df = pd.concat(D)
sns.violinplot(data=df, y="y", x='mp', hue="occ",
               split=True, inner="quart", linewidth=1,
               # )
               palette={"Yes": "b", "No": ".85"})
sns.despine(left=True)
plt.show()
