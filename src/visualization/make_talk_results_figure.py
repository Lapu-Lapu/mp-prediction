import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.globs import beta_std


def plot(label, d):
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

df_vr = pd.read_json('data/processed/processed_data_vr.json')
print('vr data:', df_vr.shape)

df_online = pd.read_json('data/processed/processed_data_online.json')
print('online data:', df_online.shape)

df = pd.concat([df_vr, df_online], join='inner')
print('data:', df.shape)

with open("data/processed/logreg_traces.pkl", "rb") as fh:
    traces = pickle.load(fh)

group = 'test_part'
xlim = (-0.03, 0.1)
# xlim = (-0.03, -0.021)
fig, ax = plt.subplots()
for label, d in df.groupby('mp_type'):
    if label in ['mapgpdm', 'mapcgpdm']:
        continue
    if label == 'dmp':
        plot(label, d)
plt.ylim((0.1, 0.55))
plt.xlim(xlim)
plt.legend(loc='upper right')
plt.title('MP-Model Confusion')
plt.ylabel('Confusion Rate')
plt.xlabel('Centered MSE')
plt.show()
