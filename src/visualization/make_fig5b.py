import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from itertools import product
import pymc3 as pm
from src.globs import beta_std

df = pd.read_json('data/processed/processed_data_online.json')

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

fig, ax = plt.subplots()
for i, (label, d) in enumerate(df.groupby('mp_type')):
    if label in ['mapgpdm', 'mapcgpdm']:
        continue

    gb = d.groupby('model')
    x = np.array(gb.partialMSE.mean())
    x = np.array(gb.X.mean())
    xmean = d['partialMSE'].mean()
    xmax = (d['partialMSE'] - xmean).max()
    xerr = np.array(gb.partialMSE.sem())
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
    xpred = np.linspace(-0.04, 0.11)
    for i in range(20):
        i = np.random.choice(range(2000))
        a, b = trace['a'][i], trace['b'][i]
        ypred = 1 / (1 + np.exp(-(a + b * xpred)))
        ax.plot(xpred, ypred, alpha=0.1, color=color)
plt.xlim((-0.04, 0.11))
plt.legend(loc='upper right')
plt.title('MP-Model Confusion - Online')
plt.ylabel('Confusion Rate')
plt.xlabel('Centered MSE')
plt.savefig('reports/figures/fig5b.pdf')
