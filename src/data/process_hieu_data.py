import pandas as pd
import numpy as np
from src.data.process_online_data import add_param_columns
import json


if __name__ == '__main__':
    df = pd.read_json('data/interim/hieu_merged.json')
    df['y'] = 1 - df['p_score']

    df['model'] = df.model.apply(lambda x: x.lower())
    df = df.rename(columns={'model': 'mp_type', 'stimulus': 'fn'})
    df = df.apply(add_param_columns, axis=1)

    print('write json files...')
    df.to_json('data/processed/processed_hieu.json')
    print('done')
