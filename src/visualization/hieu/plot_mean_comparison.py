import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from src.visualization.make_fig3 import trial_id_desktop

fps = {'desktop': 'data/raw/previous/joint_results.json',
       'vr': 'data/processed/processed_data_vr.json',
       'online': 'data/processed/processed_data_online.json',
       'hieu': 'data/processed/processed_hieu.json'
       }
df = {name: pd.read_json(fp) for name, fp in fps.items()}

# make sure model ids are identical (e.g. like this vgpdm_dyn(26)-lvm(27))

mparams = ['npsi', 'numprim', 'dyn', 'lvm']
for key in df:
    df[key].loc[:, mparams] = df[key].loc[:, mparams].fillna(-99)

# find model intersection of all data sets
m = [(set(tuple(x) for x in df[k].loc[:, mparams].values)) for k in fps.keys()]
intersect = set.intersection(*m)

# filter data
for k in fps:
    mask = [(tuple(x) in intersect) for x in df[k].loc[:, mparams].values]
    print(k, df[k].shape)
    df[k] = df[k][mask]
    print(k, df[k].shape)

# make sure every set has columns mp_type, result
# [df[k].columns for k in fps]
df['desktop'] = df['desktop'].rename(columns={'y': 'result'})

df['desktop']['mp_type'] = df['desktop'].mp_type.replace(['mapcgpdm', 'mapgpdm'], ['cgpdm', 'gpdm'])
df['vr']['mp_type'] = df['vr'].mp_type.replace(['mapcgpdm', 'mapgpdm'], ['cgpdm', 'gpdm'])
df['online']['mp_type'] = df['online'].mp_type.replace(['mapcgpdm', 'mapgpdm'], ['cgpdm', 'gpdm'])

df['desktop'] = df['desktop'][ df['desktop'].mp_type != 'catchtrial']
df['hieu'] = df['hieu'].rename(columns={'y': 'result'})

# compute mean and std of result grouped by mp_type
cfs = {k: df[k].groupby('mp_type').result.mean() for k in fps}

# plot
N = 6
ind = np.arange(N)  # the x locations for the groups
width = 0.15  # the width of the bars

mps = ['cgpdm', 'dmp', 'gpdm', 'tmp', 'vcgpdm', 'vgpdm']
c = {k: v for k, v in zip(fps.keys(), 'rgby')}

fig = plt.figure()
ax = fig.add_subplot(111)
for i, k in enumerate(fps.keys()):
    for j in ind:
        ax.bar(width / 2 + j - 2*width + i*width, cfs[k][mps[j]], width, label=k if j == 0 else None,
                color=c[k])
ax.set_ylabel('Confusion Rate')
ax.set_title('Comparison between Experiments')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(cfs[k].sort_index().index)
plt.legend()
plt.show()
