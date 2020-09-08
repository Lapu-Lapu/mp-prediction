import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

df = pd.read_json('data/processed/processed_data.json')

with open("data/processed/multilevel_trace.pkl", "rb") as fh:
    trace = pickle.load(fh)

def key_to_idx(keys):
    return dict(zip(keys, range(len(keys))))

participants = df.participant.unique()
mp_types = df.mp_type.unique()
participant_to_idx = key_to_idx(participants)
mp_type_to_idx = key_to_idx(mp_types)

df['id_mptype'] = df.mp_type.apply(lambda x: mp_type_to_idx[x])

xpred = np.linspace(-10, 10)
for mp_type, participant in product(mp_types, ['19_w_19', 7465, 7432]):
    fig, ax = plt.subplots()
    for i in range(200):
        i = np.random.choice(range(2000))
        a, b = trace['a_bar'][i], trace['b_bar'][i]
        # c = trace['c_bar'][i]
        ypred = 0.5 / (1 + np.exp(-(a + b * xpred)))
        ax.plot(xpred, ypred, alpha=0.01, color='k')
    for i in range(200):
        i = np.random.choice(range(2000))
        p_idx = participant_to_idx[participant]
        mp_idx = mp_type_to_idx[mp_type]
        a, b = trace['a'][i, p_idx], trace['b'][i, mp_idx]
        c = trace['c'][i, mp_idx]
        ypred = 0.5 / (1 + np.exp(-(a + c + b * xpred)))
        ax.plot(xpred, ypred, alpha=0.01, color='r')
    ax.set_title(f'{mp_type}, {participant}')
    plt.show()
