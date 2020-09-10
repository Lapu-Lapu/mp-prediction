import numpy as np
import pandas as pd


def beta_std(s):
    p = s.sum() + 1
    q = len(s) - s.sum() + 1
    return np.sqrt(p * q / (p + q + 1) / (p + q)**2)


def center_data(d, key='partialMSE'):
    x = d[key] - d[key].mean()
    return x/x.max()


def key_to_idx(keys):
    return dict(zip(keys, range(len(keys))))


def add_index(df, key):
    unique = df[key].unique()
    mapping = key_to_idx(unique)
    df[f'id_{key}'] = df[key].apply(lambda x: mapping[x])
    return unique, mapping, df


def load_data(fn='data/processed/processed_data.json'):
    df = pd.read_json(fn)

    df['participant'] = df.participant.apply(str)
    df['pmp'] = df.apply(lambda row: ';'.join([row.participant, row.mp_type]), axis=1)

    participants, participant_to_idx, df = add_index(df, 'participant')
    mp_types, mp_type_to_idx, df = add_index(df, 'mp_type')
    pmps, pmp_to_idx, df = add_index(df, 'pmp')
    return df, pmps, participants, mp_types, participant_to_idx, mp_type_to_idx, pmp_to_idx
