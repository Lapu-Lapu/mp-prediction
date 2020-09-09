import pandas as pd

df_vr = pd.read_json('data/processed/processed_data_vr.json')
df_vr['setting'] = 'vr'
print('vr data:', df_vr.shape)

df_online = pd.read_json('data/processed/processed_data_online.json')
df_online['setting'] = 'online'
print('online data:', df_online.shape)

df = pd.concat([df_vr, df_online], join='inner', ignore_index=True)
print('data:', df.shape)

df.to_json('data/processed/processed_data.json')
