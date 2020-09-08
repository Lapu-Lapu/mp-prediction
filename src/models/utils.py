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


def load_data(fn='data/processed/processed_data.json'):
    df = pd.read_json('data/processed/processed_data_online.json')

    df['participant'] = df.participant.apply(str)
    df['pmp'] = df.apply(lambda row: ';'.join([row.participant, row.mp_type]), axis=1)
    pmps = df.pmp.unique()
    participants = df.participant.unique()
    mp_types = df.mp_type.unique()
    participant_to_idx = key_to_idx(participants)
    mp_type_to_idx = key_to_idx(mp_types)
    pmp_to_idx = key_to_idx(pmps)

    df['id_mptype'] = df.mp_type.apply(lambda x: mp_type_to_idx[x])
    df['id_participant'] = df.participant.apply(lambda x: participant_to_idx[x])
    df['id_pmp'] = df.pmp.apply(lambda x: pmp_to_idx[x])
    return df, pmps, participants, mp_types, participant_to_idx, mp_type_to_idx, pmp_to_idx
