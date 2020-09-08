import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.models.utils import beta_std, center_data


def plot(label, d):
    d['X'] = center_data(d, key='partialMSE')
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
    # ax.plot(d.X, d.result, '.', alpha=0.1)

    trace = traces[label]
    xpred = np.linspace(*xlim)
    # xpred = np.linspace(-1, 1)
    for i in range(20):
        i = np.random.choice(range(2000))
        a, b = trace['a'][i], trace['b'][i]
        ypred = 0.5 / (1 + np.exp(-(a + b * xpred)))
        ax.plot(xpred, ypred, alpha=0.1, color=color)

df = pd.read_json('data/processed/processed_data.json')

with open("data/processed/logreg_traces.pkl", "rb") as fh:
    traces = pickle.load(fh)

group = 'test_part'
# xlim = (-0.05, 0.1)
xlim = (0.0, 1.8)
fig, ax = plt.subplots()
for label, d in df.groupby('mp_type'):
    if label in ['mapgpdm', 'mapcgpdm']:
        continue
    # if label == 'tmp':
    plot(label, d)
plt.ylim((0.15, 0.55))
plt.xlim(xlim)
plt.legend(loc='upper right')
plt.title('MP-Model Confusion')
plt.ylabel('Confusion Rate')
plt.xlabel('MSE')
plt.show()
