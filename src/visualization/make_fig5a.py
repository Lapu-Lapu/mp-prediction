import pandas as pd
import pymc3 as pm
import numpy as np
import matplotlib.pyplot as plt
import pickle

from src.data.utils import trial_id, process_data_dict
from src.globs import beta_std

df = pd.read_json('data/processed/processed_data_vr.json')

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
