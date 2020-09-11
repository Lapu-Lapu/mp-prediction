import pandas as pd
import numpy as np

df_vr = pd.read_json('data/processed/processed_data_vr.json')
df_vr['setting'] = 'vr'
df_vr['occluded_contact'] = df_vr.apply(
        lambda row:
        (row.movement in ['pass-botte', 'return-bottle']),
        axis=1)
print('vr data:', df_vr.shape)
df_vr['fn'] = df_vr.apply(lambda row:
          f'model({row.mp_type})-dataset({row.movement})-npsi({int(row.npsi)})-hold({row.hold})' if ~np.isnan(row.npsi) else '',
                          axis=1)

ghost_experiment = pd.read_csv(
    'data/raw/online/attention_info/test_dmp.csv')
attcheck_set = set(ghost_experiment[ghost_experiment.ghosting].fn)
attcheck_idx = df_vr.fn.apply(lambda s: s in attcheck_set)
df_vr['attention_check'] = attcheck_idx
df_vr = df_vr[~df_vr['attention_check']]
# df_vr.drop(columns=['fn'], inplace=True)
print('vr data:', df_vr.shape)

df_online = pd.read_json('data/processed/processed_data_online.json')
df_online['setting'] = 'online'
print('online data:', df_online.shape)

df = pd.concat([df_vr, df_online], join='inner', ignore_index=True)
print('data:', df.shape)

df.to_json('data/processed/processed_data.json')
