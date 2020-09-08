import numpy as np
import pandas as pd
from src.models.utils import beta_std, model


def _parse_params(row: pd.Series) -> (str, str):
    m = row["mp_type"]
    param_str = m + "_"
    for prm_name in model[m]["params"]:
        param_str += prm_name
        param_str += str(int(row[prm_name]))
    return param_str


def trial_id_desktop(row):
    return _parse_params(row)


def tested_in_both_experiments(s):
    return s in intersection


df_old = pd.read_json('data/raw/previous/joint_results.json')
df_new = pd.read_json('data/processed/processed_data_vr.json')

df_old['modelstr'] = df_old.apply(trial_id_desktop, axis=1)

df = df_new

old_set = set(df_old.apply(trial_id_desktop, axis=1))
new_set = set(df_new.apply(trial_id_desktop, axis=1))

print("Elements in old set:", len(old_set))
print("Elements in new set:", len(new_set))
intersection = old_set.intersection(new_set)
print("Elements in intersections:", len(intersection))

print("df_new.shape:", df_new.shape)
print("df_old.shape:", df_old.shape)

df_old = df_old[df_old.modelstr.apply(tested_in_both_experiments)]

cr_old = df_old.groupby('mp_type').agg({"y": "mean"})
bstd_old = df_old.groupby('mp_type').agg({"y": beta_std})

df_new_cr = df_new.groupby("mp_type").agg({"result": "mean"})
bstd_new = df_new.groupby(["mp_type"]).agg({"beta_stderr_modelstr": "mean"})

old = pd.concat([cr_old, bstd_old], axis=1).reindex(cr_old.index)
new = pd.concat([df_new_cr, bstd_new], axis=1).reindex(df_new_cr.index)

comp_res = pd.concat([old, new], axis=1).reindex(old.index)
comp_res.columns = ['cr_old', 'bstd_old', 'cr_new', 'bstd_new']

df_online = pd.read_json('data/processed/processed_data_online.json')
print(df_online.shape)
df_online = pd.read_json(
    'data/processed/processed_online_with_attentioncheck.json')
print(df_online.shape)

ghost_experiment = pd.read_csv('data/raw/online/attention_info/test_dmp.csv')
attcheck_set = set(ghost_experiment[ghost_experiment.ghosting].fn)
attcheck_idx = df_online.fn.apply(lambda s: s in attcheck_set)
attcheck = df_online[attcheck_idx]

df_online['attention_check'] = attcheck_idx
# failed_attention = attcheck.result.apply(lambda r: r == 1)
# bad_vps = set(df_online[attcheck_idx][failed_attention].subject)

sbj_att = attcheck.groupby('subject').result.mean()
sbj_att.describe()

bad_vps = set(sbj_att[sbj_att > 0.4].index)

print(df_online.shape)
df_online = df_online[df_online.subject.apply(lambda vp: not (vp in bad_vps))]
print(df_online.shape)
df_online = df_online[~df_online['attention_check']]
print(df_online.shape)

comp_res['online'] = df_online.groupby('mp_type').result.mean()
comp_res['online_beta'] = df_online.groupby('mp_type').result.agg(beta_std)

import matplotlib.pyplot as plt

mp_idx = [
    'tmp',
    'dmp',
    'vcgpdm',
    'mapcgpdm',
    'vgpdm',
    'mapgpdm',
]
pretty = {
    'tmp': 'TMP',
    'dmp': 'DMP',
    'vgpdm': 'vGPDM',
    'vcgpdm': 'vCGPDM',
    'mapgpdm': 'GPDM',
    'mapcgpdm': 'CGPDM',
}
comp_res = comp_res.loc[mp_idx]
# ['DMP', 'cGPDM', 'GPDM', 'TMP', 'vCGPDM', 'vGPDM'])
comp_res

desktop = comp_res.sort_values('cr_old', ascending=False).cr_old.index
vr = comp_res.sort_values('cr_new', ascending=False).cr_old.index
online = comp_res.sort_values('online', ascending=False).cr_old.index
print(desktop)
print(vr)
print(online)

N = 6

ind = np.arange(N)  # the x locations for the groups
width = 0.25  # the width of the bars

fig = plt.figure()
ax = fig.add_subplot(111)
rects1 = ax.bar(width / 2 + ind - width,
                comp_res.cr_old,
                width,
                color='royalblue',
                yerr=comp_res.bstd_old)
rects2 = ax.bar(width / 2 + ind,
                comp_res.cr_new,
                width,
                color='seagreen',
                yerr=comp_res.bstd_new)
rects3 = ax.bar(width / 2 + ind + width,
                comp_res.online,
                width,
                color='yellow',
                yerr=comp_res.online_beta)

# add some
ax.set_ylabel('Confusion Rate')
ax.set_title('Comparison between Experiments')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels((pretty[idx] for idx in mp_idx))

ax.legend((rects1[0], rects2[0], rects3[0]), ('Desktop', 'VR', 'Online'))

plt.ylim(0, 0.55)
ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='dashed')
plt.savefig('reports/figures/fig3.pdf')
