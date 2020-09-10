import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from src.models.utils import beta_std, center_data

# df = pd.read_json('data/processed/processed_data.json')
df = pd.read_json('data/processed/processed_data_online.json')

with open("data/processed/multilevel_trace.pkl", "rb") as fh:
    trace = pickle.load(fh)

def key_to_idx(keys):
    return dict(zip(keys, range(len(keys))))

participants = df.participant.unique()
mp_types = df.mp_type.unique()
participant_to_idx = key_to_idx(participants)
mp_type_to_idx = key_to_idx(mp_types)

df['id_mp_type'] = df.mp_type.apply(lambda x: mp_type_to_idx[x])

# xpred = np.linspace(-10, 10)
# for mp_type, participant in product(mp_types, ['19_w_19', 7465, 7432]):
#     fig, ax = plt.subplots()
#     for i in range(200):
#         i = np.random.choice(range(2000))
#         a, b = trace['a_bar'][i], trace['b_bar'][i]
#         # c = trace['c_bar'][i]
#         ypred = 0.5 / (1 + np.exp(-(a + b * xpred)))
#         ax.plot(xpred, ypred, alpha=0.01, color='k')
#     for i in range(200):
#         i = np.random.choice(range(2000))
#         p_idx = participant_to_idx[participant]
#         mp_idx = mp_type_to_idx[mp_type]
#         a, b = trace['a'][i, p_idx], trace['b'][i, mp_idx]
#         c = trace['c'][i, mp_idx] * trace['sigma_c'][i]
#         ypred = 0.5 / (1 + np.exp(-(a + c + b * xpred)))
#         ax.plot(xpred, ypred, alpha=0.01, color='r')
#     ax.set_title(f'{mp_type}, {participant}')
#     plt.show()

a_mean = trace['a'].mean(axis=0)
c_mean = trace['c'].mean(axis=0)
b_mean = trace['b'].mean(axis=0)

def intersubjectX(row):
    pidx = participant_to_idx[row.participant]
    mpidx = mp_type_to_idx[row.mp_type]
    return - (a_mean[pidx] + c_mean[mpidx] + b_mean[mpidx] * row.partialMSE)

fig, ax = plt.subplots()
for label, d in df.groupby('mp_type'):
    if label in ['mapgpdm', 'mapcgpdm']:
        continue
    d['X'] = d['partialMSE']
    # d['X'] = d.apply(intersubjectX, axis=1)
    gb = d.groupby('model')
    x = np.array(gb.X.mean())
    xerr = np.array(gb.X.sem())
    y = np.array(gb.result.mean())
    # y = np.array(gb.result.agg(lambda x: np.log(1/x.mean()-1)))
    # y = np.zeros(len(gb))
    # j = 0
    # for model, m in gb:
    #     yagg = 0
    #     for participant, p in m.groupby('participant'):
    #         pidx = participant_to_idx[participant]
    #         if p.result.mean() < 0.5 and p.result.mean() > 0.001:
    #             yt = np.log(0.5/p.result.mean()-1)
    #         else:
    #             yt = 0
    #         yagg += yt + a_mean[pidx]
    #     y[j] = yagg / len(participant_to_idx)
    #     j += 1
    # print('a', y)

    yerr = np.array(gb.result.agg(beta_std))
    p = ax.errorbar(x,
                    y,
                    xerr=xerr,
                    yerr=yerr,
                    fmt='.',
                    alpha=0.7,
                    label=label)
    color = p[0].get_color()
    xpred = np.linspace(x.min(), x.max())
    for i in range(30):
        for participant, color in zip([8755, 7465, 7432], list('rgb')):
            i = np.random.choice(range(2000))
            p_idx = participant_to_idx[participant]
            mp_idx = mp_type_to_idx[label]
            a, b = trace['a'][i, p_idx], trace['b'][i, mp_idx]
            c = trace['c'][i, mp_idx] * trace['sigma_c'][i]
            ypred = 0.5 / (1 + np.exp(-(a + c + b * xpred)))
            ax.plot(xpred, ypred, alpha=0.05, color=color)
# plt.ylim((0.15, 0.55))
# plt.xlim(xlim)
plt.legend(loc='upper right')
plt.title(f'MP-Model Confusion: {label}')
plt.ylabel('Confusion Rate')
plt.xlabel('MSE')
plt.show()
