import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from itertools import product
import pymc3 as pm
from src.globs import beta_std

df = pd.read_json('data/processed/processed_data_online.json')

traces = {}
for i, (touch, d) in enumerate(df.groupby('occluded_contact')):
    with pm.Model() as logistic_model:
        a = pm.Normal('a', 0, 10)
        b = pm.Normal('b', 0, 10)
        x = d['partialMSE'] - d['partialMSE'].mean()
        x = x / x.max()
        p = 1 / (1 + np.exp(-(a + b * x)))
        s = pm.Bernoulli('s', p=p, observed=d['result'])
        trace = pm.sample(1000, tune=1000, init='adapt_diag')
    traces[touch] = trace

group = 'test_part'
c = ['blue', 'green']
legend = {
    True: 'Contact during Occlusion',
    False: 'No Contact during Occlusion'
}
data = df
fig, ax = plt.subplots()
for i, (touch, d) in enumerate(data.groupby('occluded_contact')):
    gb = d.groupby('model')
    x = np.array(gb.X.mean())
    xerr = np.array(gb.X.sem())
    y = np.array(gb.result.mean())
    yerr = np.array(gb.result.agg(beta_std))
    ax.errorbar(x,
                y,
                xerr=xerr,
                yerr=yerr,
                fmt='.',
                label=legend[touch],
                alpha=0.3,
                color=c[i])
    trace = traces[touch]
    xpred = np.linspace(-0.04, 0.11)
    for _ in range(20):
        j = np.random.choice(range(2000))
        a, b = trace['a'][j], trace['b'][j]
        ypred = 1 / (1 + np.exp(-(a + b * xpred)))
        ax.plot(xpred, ypred, alpha=0.1, color=c[i])
plt.title('Effect of Contact on Predictability')
plt.xlabel('Centered MSE')
plt.ylabel("Confusion Rate")
plt.legend()
plt.tight_layout()
plt.savefig('reports/figures/fig6.pdf')
