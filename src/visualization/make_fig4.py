import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.globs import beta_std

df = pd.read_csv('data/raw/vr/catchtrials/catchtrials.csv')
df['shift'] = df.apply(lambda row: row.first_seq
                       if 'ms' in row.first_seq else row.second_seq,
                       axis=1)

catch_cf = df.groupby('shift').result.mean()
catch_err = df.groupby('shift').result.agg(beta_std)

labels = ['400ms', '700ms', '1000ms']
catch_cf = catch_cf.loc[labels]

df_online = pd.read_json("data/processed/catchtrial_online.json")
df_online['offset'] = df_online.test_stimulus.apply(lambda d: d['offset'])

online_cf = df_online.groupby('offset').result.mean()
online_err = df_online.groupby('offset').result.agg(beta_std)
fig, ax = plt.subplots()
ax.bar(x=[3.75, 6.67, 10.0],
       height=online_cf,
       yerr=online_err,
       width=0.5,
       label='Online')
ax.bar(x=[4.0, 7.0, 10.0],
       height=catch_cf,
       yerr=catch_err,
       width=0.5,
       label='VR')
plt.legend()
plt.xticks([3.75, 4, 6.67, 7, 10.],
           [None, '375/\n400ms', None, '667/\n700ms', '1000ms'])
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
plt.title('Catch Trial Confusion')
plt.ylabel('Confusion Rate')
plt.xlabel('Shift')
plt.tight_layout()
plt.savefig('reports/figures/fig4.pdf')
