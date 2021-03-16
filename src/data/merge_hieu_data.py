import pandas as pd
import numpy as np
from src.data.utils import open_pkls
from src.data.process_online_data import make_fn
from functools import partial


def add_modelscores(row, SCORES):
    try:
        row['MSE'] = SCORES[row.stimulus]['MSE']
    except KeyError:
        print(f"{row.stimulus} not in scores pickle.")
        row['MSE'] = np.nan
    return row


def add_id(d):
    d['id'] = make_fn(d['settings'])
    return d


def load_scores(fn='data/raw/online/scores/combined_errors.pkl'):
    scores = open_pkls(fn)
    s = map(add_id, scores)
    return {elem['id']: elem for elem in s}

if __name__ == '__main__':
    # with open('data/raw/online/contact_info/segments.json') as fo:
    #     contact_timings = json.load(fo)

    df = pd.read_pickle('/home/benjamin/temp/hieu/ba/processed_data')

    # load scores into global scope for load_data to be able to add model scores
    SCORES = load_scores()
    df = df.apply(partial(add_modelscores, SCORES=SCORES), axis=1)
    df.to_json('data/interim/hieu_merged.json')
