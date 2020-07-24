from src.data.utils import (open_pkls,
                            load_post_processed_data, get_mptype,
                            process_data_dict, get_params, trial_id,
                            parse_filename)
from src.globs import beta_std

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import partial


def add_model_scores(S, id_str: str, row: pd.Series) -> pd.Series:
    """Concatenate model scores to trial data."""
    scores = S.loc[row[id_str]]
    return pd.concat((row, scores))

# import data_processing as io

# ## Prepare training results data

D = [
    d for fn in [
        'data/raw/vr/scores/combined_model_errors.pkl',
        "data/raw/vr/scores/combined_errors_1.pkl",
        "data/raw/vr/scores/combined_errors_2.pkl"
    ] for d in open_pkls(fn)
]
training_data = process_data_dict(D)

gb = training_data.groupby('training_id')
modelscores = gb.agg({
    score: lambda x: x.mean()
    for score in ["MSE", "ELBO", 'dyn_elbo', 'lvm_elbo']
})

# ## Prepare experiment data

df = load_post_processed_data('data/raw/vr/')


class CAF:
    def __init__(self, fn='data/raw/vr/caf/avatar.cfg'):
        with open(fn) as f:
            self.cfg_full = f.readlines()
        self.cfg = [c for c in self.cfg_full if c[0] == 'a']

    def return_caf(self, idx):
        return self.cfg[idx - 1].replace('-', '')

    def return_hold(self, row):
        ser = parse_filename(self.return_caf(row['stimulus_id']))
        return ser


caf = CAF()
caf_info = df.apply(caf.return_hold, axis=1)

PP = {
    "returnbottle": "return-bottle",
    "passbottlehold": "pass-bottle-hold",
    "passbottle": "pass-bottle"
}

caf_info['movement'] = caf_info.dataset.apply(lambda x: PP[x])
caf_info['mp_type'] = caf_info.apply(get_mptype, axis=1)
caf_info['trial_id'] = caf_info.apply(trial_id, axis=1)

for key in [
        'dataset', 'dyn', 'hold', 'lvm', 'npsi', 'numprim', 'movement',
        'mp_type', 'trial_id'
]:
    df[key] = caf_info[key]

df = df.apply(partial(add_model_scores, modelscores, 'trial_id'), axis=1)


def append_confusion_stats(df: pd.DataFrame,
                           id_str='trial_id') -> pd.DataFrame:
    """Group df by id_str, compute mean and beta standarderror on group.

    Returns df with mean and beta standarderror appended as column.
    """
    confusion_rates = df.groupby(id_str)['result'].agg(
        **{
            'confusion_rate_' + id_str: 'mean',
            'beta_stderr_' + id_str: beta_std
        })
    return df.apply(partial(add_model_scores, confusion_rates, id_str), axis=1)


df = append_confusion_stats(df, 'trial_id')


def build_modelstr(x: pd.Series) -> str:
    return "_".join(x.trial_id.split('_')[:2])


df['modelstr'] = df.apply(build_modelstr, axis=1)

df = append_confusion_stats(df, 'modelstr')

# ## Write processed data

df = df.drop([
    'block', 'stimulus_id', 'correct_sequence', 'part_input', 'first_seq',
    'second_seq', 'first_seq_from', 'date', 'expName', 'second_seq_from',
    'trialnumber'
],
             axis=1)

df.to_json('data/processed/processed_data_vr.json')
