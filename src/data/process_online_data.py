from glob import glob
import json
import pandas as pd
import os
from datetime import datetime
import json
import numpy as np
from functools import partial

from src.globs import model
from src.data.utils import open_pkls, _parse_params, trial_id, parse_filename


def get_mptype(s):
    if s == 'catchtrial':
        return s
    ser = parse_filename(s)
    if ser.model == 'vcgpdm':
        if ser['mode'] == 'ELBO':
            prefix = 'v'
        else:
            prefix = 'map'
        if ser.parts == 3:
            m = prefix + 'cgpdm'
        else:
            m = prefix + 'gpdm'
        return m
    else:
        return ser.model


def compute_partial_mse(errors):
    ds = errors['settings']['dataset']
    hold = errors['settings']['hold']
    for occ in ['pre', 'inter', 'post']:
        start, end = contact_timings[ds][occ][hold]
        continuation_err = errors['observed'][end:]
        mse = np.sum((continuation_err - errors['predicted'][end:])**2)
        errors[occ] = mse / len(continuation_err)
    errors['id'] = make_fn(errors['settings'])
    return errors


def make_fn(settings):
    VCGPDM = {
        'params': ['mode', 'parts', 'dyn', 'lvm'],
        'scores': ['MSE', 'ELBO', 'dyn_elbo', 'lvm_elbo']
    }
    MAPGPDM = {'params': ['mode', 'parts'], 'scores': ['MSE']}
    model['vcgpdm'] = VCGPDM
    m = settings['model']
    modelm = model[m]
    if m == 'vcgpdm':
        if settings['mode'] == 'MAP':
            # modelm = model['cgpdm']
            modelm = MAPGPDM
    ds = settings['dataset']
    hold = settings['hold']
    s = f'model({m})-dataset({ds})-'
    for p in modelm['params']:
        s += f'{p}({settings[p]})-'
    s += f'hold({hold})'
    return s


def get_condition(ds, hold, occ_start):
    # ds, hold = trial.dataset, trial.hold
    for occ in ['pre', 'post', 'inter']:
        if occ_start == contact_timings[ds][occ][hold][0]:
            return occ
    raise Exception("occ start not found!")


def get_ds_hold_from_catch(fn):
    tpl = fn.split('-training')
    ds = tpl[0].split('/')[-1]
    hold = int(tpl[1][0])
    return ds, hold


def add_partial_mse(row, score_data):
    if 'catch_trial' in row.test_stimulus.keys():
        fn = row.test_stimulus['fn_train']
        ds, hold = get_ds_hold_from_catch(fn)
    else:
        fn = row.test_stimulus['fn'].split('/')[-2]
        trial = parse_filename(fn)
        ds, hold = trial.dataset, trial.hold
    occ = get_condition(ds, hold, row.test_stimulus['occ_start'])
    row['occ_cond'] = occ
    row['hold'] = hold
    row['movement'] = ds
    try:
        partial_mse = score_data[fn][occ]
    except KeyError:
        partial_mse = None
    row['partial_mse'] = partial_mse
    return row


def is_correct_response(row):
    if row.order == 'model_first':
        return row.button_pressed == '1'
    elif row.order == 'training_first':
        return row.button_pressed == '0'
    else:
        raise Exception


def preprocess(Df, score_data):
    df = Df[Df.test_part == 'response']

    def get_fn(row):
        if 'fn' in row:
            return row['fn'].split('/')[-2]
        else:
            return 'catchtrial'

    df.drop(columns=['rt', 'stimulus'], inplace=True)
    df.loc[:, 'fn'] = df.loc[:, 'test_stimulus'].apply(get_fn)
    df.loc[:, 'order'] = df.apply(lambda x: x['test_stimulus']['order'],
                                  axis=1)
    df.loc[:, 'vp_correct'] = df.apply(is_correct_response, axis=1)
    df.loc[:, 'result'] = 1 - df.loc[:, 'vp_correct']
    df = df.apply(partial(add_partial_mse, score_data=score_data), axis=1)
    return df


def load_data(json_fps, score_data):
    Dfs = []
    for fp in json_fps:
        df_fn = 'data/interim/' + fp.split('/')[-1]
        if os.path.exists(df_fn):
            Df = pd.read_json(df_fn)
        else:
            with open(fp) as fo:
                d = json.load(fo)

            Df = pd.DataFrame(d)
            date = datetime.utcfromtimestamp(os.path.getmtime(fp))
            try:
                Df = preprocess(Df, score_data)
                try:
                    t, c = Df.shape
                except ValueError:
                    continue
                if t > 10 and c == 18:
                    Df.to_json(df_fn)
                else:
                    continue
                print('loading', fp, Df.shape, end=':')
            except AttributeError:
                print("AttributeError", fp)
                print(Df.shape)
                continue
            print(date.strftime("%m-%d- %H:%M"))
        Dfs.append(Df)
    return Dfs


def add_param_columns(row):
    s = row.fn
    ser = parse_filename(s)
    mp_type = get_mptype(s)
    # for p in io.model[mp_type]['params']:
    for p in model[mp_type]['params']:
        row[p] = ser[p]
    return row


def load_scores(fn='data/raw/online/scores/combined_errors.pkl'):
    scores = open_pkls(fn)
    s = map(compute_partial_mse, scores)
    return {elem['id']: elem for elem in s}


if __name__ == '__main__':
    with open('data/raw/online/contact_info/segments.json') as fo:
        contact_timings = json.load(fo)

    # load scores into global scope for load_data to be able to add model scores
    score_data = load_scores()

    # load and process all json files
    json_fps = glob('data/raw/online/*.json')
    Dfs = load_data(json_fps, score_data)
    for df in Dfs:
        try:
            print(df.iloc[0]['sona'], df.shape, df.result.mean())
            # s += len(df)
        except (KeyError, TypeError):
            pass
    df = pd.concat(
        [df for df in Dfs if not isinstance(df.columns, pd.Int64Index)],
        axis=0,
        ignore_index=True)
    df = df.dropna(axis=0, how='all')

    # add generally useful columns
    df.loc[:, 'mp_type'] = df.fn.apply(get_mptype)
    df = df.apply(add_param_columns, axis=1)
    df.loc[:, 'trial_id'] = df.apply(trial_id, axis=1)
    df['occluded_contact'] = df.apply(
        lambda row:
        (row.occ_cond == 'inter'
         if row.movement in ['pass-bottle', 'return-bottle'] else False),
        axis=1)

    # split catchtrials into separate dataframe
    catchtrials = df[df.mp_type == 'catchtrial']
    df = df[df.mp_type != 'catchtrial']
    for i in np.where(np.isnan(df.partial_mse))[0]:
        pass  # TODO: find error data file for 5,5 vgpdm pass-bottle-hold!
        # pprint(df.iloc[i])
    df = df[~df.trial_id.apply(lambda x: 'vgpdm_dyn5lvm5_pass-bottle-hold' in x
                               )]

    # add useful columns for non-catchtrial data
    df['model'] = df.apply(_parse_params, axis=1)
    df['X'] = (df.partial_mse - df.partial_mse.mean()) / (
        df.partial_mse.max() - df.partial_mse.mean())
    df['direction'] = df.movement.apply(
        lambda x: 'from_left'
        if x in ['pass-bottle', 'pass-bottle-hold'] else 'from_right')
    df['touch'] = df.movement.apply(
        lambda x: x in ['pass-bottle', 'return-bottle'])

    # process attention checks
    ghost_experiment = pd.read_csv(
        'data/raw/online/attention_info/test_dmp.csv')
    attcheck_set = set(ghost_experiment[ghost_experiment.ghosting].fn)
    attcheck_idx = df.fn.apply(lambda s: s in attcheck_set)
    attcheck = df[attcheck_idx]
    df['attention_check'] = attcheck_idx
    sbj_att = attcheck.groupby('subject').result.mean()
    bad_vps = set(sbj_att[sbj_att > 0.4].index)
    df_withattention = df
    df = df[df.subject.apply(lambda vp: not (vp in bad_vps))]
    df = df[~df['attention_check']]

    catchtrials = catchtrials[catchtrials.subject.apply(
        lambda vp: not (vp in bad_vps))]

    print('write json files...')
    df_withattention.to_json(
        'data/processed/processed_online_with_attentioncheck.json')
    catchtrials.to_json('data/processed/catchtrial_online.json')
    df.to_json('data/processed/processed_data_online.json')
    print('done')
