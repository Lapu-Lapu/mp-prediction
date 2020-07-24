# -*- coding: utf-8 -*-
"""

This script is used with the data of the VR-Psychophysics Experiment to:
    - clean the dataset from training trials
    - clean the dataset from catch trials, as well as creating a csv-file
        containing all catch trials for catch_analysis
    - create a dataset that can be read by the analysis script and the vizualisation script

"""
from __future__ import division, print_function
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import os
from src.globs import model
import pickle

import re

MP_PARAMS = ['numprim', 'npsi', 'dyn', 'lvm']
remap_std_modelname = {
    "C": "vcgpdm",
    "nC": "vgpdm",
    "cMAP": "mapcgpdm",
    "MAP": "mapgpdm",
    "tmp": "tmp",
    "dmp": "dmp"
}


def load_data(dirname) -> pd.DataFrame:
    fp = os.path.join(dirname, fn)
    files = [dirname + f for f in os.listdir(dirname) if f[-3:] == 'csv']
    df = load_csv(files[0])
    for i in range(1, len(files)):
        df = df.append(load_csv(files[i]), ignore_index=True)
    return df


def load_csv(fn):
    cols = (
        'trialnumber,block,first_seq_from,second_seq_from,' +
        'participant,result,correct_sequence,first_seq,second_seq,expPart,' +
        'part_input,date,expName,' + 'trialstart_time,' +
        'trialend_time,trialtime,answer_time').split(',')
    # time_at_videoonset,time_at_videoend,
    # time_at_fixation_cross_onset,time_at_key_response,
    df = pd.read_csv(fn, usecols=cols)

    # change trial index
    df = df[~np.isnan(df.trialnumber)]
    # df = df[~df.participant_key_response.map(pd.isnull)]

    # adapt to new experiment names
    df = df.rename(
        {
            "trials.thisN": 'n_trial',
            "trials.thisTrialN": "n_inblock"
        }, axis=1)

    # get not natural stimulusname
    df['generated'] = df.apply(get_artificial, axis=1)
    df['natural'] = df.apply(get_natural, axis=1)
    return df


def get_artificial(row: pd.Series):
    """
    Returns name of generated stimulus.
    """
    # when the correct sequence is 1, look for the string in the second sequence,
    # where the model name should be displayed
    inv = {'1': 'second_seq_from', '2': 'first_seq_from'}
    try:
        return row[inv[row.correct_sequence]]  #.split('.')[0]
    except KeyError:
        return 'catchtrial'


def get_natural(row: pd.Series):
    """
    Returns name of natural stimulus.
    """
    try:
        return row[row.correct_sequence]  #.split('.')[0]
    except KeyError:
        return 'catchtrial'


def add_model_param_columns(row: pd.Series):
    '''split up the "model" into model types and model parameters.

    It takes a row, and appends the information stored
    in the model-string to it. When used on a dataframe with apply,
    this produces a dataframe with new columns containing
    movement type, model type, and parameters.
    Parameters which are not existing for a given model type are
    filled with None.
    '''
    s = row.model
    lst = s.split('_')
    movement_type = lst[0]
    m = lst[1]
    if m[:3] in ['tmp', 'dmp']:
        m = m[:3]
    params = re.findall(r'\d+', s)
    modelname = remap_std_modelname[m]
    type_specifics = model[modelname]
    d = {'movement': movement_type, 'modeltype': modelname}
    for param in MP_PARAMS:
        if param in type_specifics['params']:
            if param == 'lvm':
                d[param] = int(params[1])
            else:
                d[param] = int(params[0])
        else:
            d[param] = None
    return pd.concat([row, pd.Series(d)])


def training_movement(m):
    """
    returns stimulus id, in order to find out, which starting position
    of the model-generated movement is used
    """
    if m.first_seq < 25:
        return m.second_seq
    elif m.second_seq <= 25:
        return m.first_seq
    else:
        raise Warning


def load_post_processed_data(rootdir='../VR_data/') -> pd.DataFrame:
    """Open all csv files in root directory."""
    df = load_data(rootdir)

    # remove training trials
    df = df[~df.expPart.str.contains("trainingtrials")]

    # then remove the expPart column, for its purpose is fulfilled
    df = df.drop('expPart', 1)
    '''   catchtrials Extraction  '''

    # catchtrials contain certain strings in the columns 'first_seq' and
    # "second_seq" they help to identify the catchtrials
    df_catch_1 = df[df['first_seq'].isin(['400ms', '700ms', '1000ms'])]
    df_catch_2 = df[df['second_seq'].isin(['400ms', '700ms', '1000ms'])]
    df_catch = pd.concat(pd.DataFrame(i) for i in (df_catch_1, df_catch_2))
    df_catch.to_csv('data/processed/catchtrial_vr.csv')

    # now remove catchtrials from df

    df = df[~df['first_seq'].isin(['400ms', '700ms', '1000ms'])]
    df = df[~df['second_seq'].isin(['400ms', '700ms', '1000ms'])]

    # then remove 'first_seq' and 'second_seq' column, for its purpose is
    # fulfulled

    # make values numeric
    df.first_seq = pd.to_numeric(df.first_seq,
                                 errors='coerce').astype(np.int64)
    df.second_seq = pd.to_numeric(df.second_seq,
                                  errors='coerce').astype(np.int64)

    df['stimulus_id'] = df.apply(training_movement, axis=1)

    df = df.drop(['generated', 'natural'], axis=1)
    ''' create a column with the model type '''
    # 1 == wrong answer (generated movement fooled the participant); 0 ==
    # correct answer (natural stimulus selected)
    #df = df.drop(["answer_time", "trialend_time",
    #    "correct_sequence", "part_input", "trials.thisRepN","trials.thisTrialN",
    #   "trials.thisN", "trials.thisIndex", "date", "expName" ], axis = 1)

    # aus "second_seq_from" und "first_seq_from" die column "model" bilden
    df['second_seq_from'] = df['second_seq_from'].str.replace('natpasslst', '')
    df['second_seq_from'] = df['second_seq_from'].str.replace('natholdlst', '')
    df['second_seq_from'] = df['second_seq_from'].str.replace('natretlst', '')

    df['first_seq_from'] = df['first_seq_from'].str.replace('natpasslst', '')
    df['first_seq_from'] = df['first_seq_from'].str.replace('natholdlst', '')
    df['first_seq_from'] = df['first_seq_from'].str.replace('natretlst', '')

    df['model'] = df[['first_seq_from',
                      'second_seq_from']].apply(lambda x: ''.join(x), axis=1)

    # # then remove 'first_seq_from' and 'second_seq_from' column, for its
    # # purpose is fulfulled
    # df = df.drop('first_seq_from', 1)
    # df = df.drop('second_seq_from', 1)

    df = df.apply(add_model_param_columns, axis=1)
    # drop 'model' column
    df = df.drop(['model'], axis=1)
    # rename a column to let old scripts read it
    df = df.rename(columns={'modeltype': 'mp_type'})

    PP = {
        "ret": "return-bottle",
        "hold": "pass-bottle-hold",
        "put": "pass-bottle"
    }
    df['movement'] = df['movement'].apply(lambda x: PP[x])
    return df


def get_mptype(d: dict) -> str:
    """from settings
    gets model name according to
    mp_perception naming convention
    """
    model = d['model']
    if model == 'vcgpdm':
        if d['parts'] == 1:
            model = "vgpdm"
        if d["mode"] == "MAP":
            model = d["mode"].lower() + model[1:]
    return model


def get_params(row: pd.Series) -> pd.Series:
    """make new columns for ideal obs params
    """
    ser = pd.Series(index=['dyn', 'lvm', 'npsi', 'numprim'], dtype=int)
    for param in model[row.mp_type]['params']:
        ser[param] = row['settings'][param]
    return pd.concat([row, ser])


def process_data_dict(D: dict) -> pd.DataFrame:
    training_data = pd.DataFrame(D)
    training_data = training_data.rename(
        {
            "WRAP_DYN": "dyn_warped_error",
            "WRAP_PATH": "path_warped_error",
            "dyn_ELBO": "dyn_elbo",
            "lvm_ELBO": "lvm_elbo"
        },
        axis=1)
    training_data['mp_type'] = training_data['settings'].apply(get_mptype)
    training_data['movement'] = training_data.settings.apply(
        lambda x: x['dataset'])
    training_data['hold'] = training_data.settings.apply(lambda x: x['hold'])
    training_data = training_data.apply(get_params, axis=1)
    training_data['training_id'] = training_data.apply(trial_id, axis=1)
    return training_data


def trial_id(row: pd.Series) -> str:
    """Create unique identifier for a trial.

    Arguments:
    row --  row of DataFrame containing all trial data

    Return string like: "tmp_numprim11_return-bottle_0"
    """
    param_str = _parse_params(row)
    s = (param_str + "_" + row["movement"] + "_" + str(row["hold"]))
    return s


def _parse_params(row: pd.Series) -> (str, str):
    m = row["mp_type"]
    param_str = m + "_"
    for prm_name in model[m]["params"]:
        if prm_name in ['mode']:
            continue
        param_str += prm_name
        param_str += str(int(row[prm_name]))
    return param_str


def movement_model_id(row):
    """
    Return string like: "tmp_numprim11_return-bottle"
    """
    return _parse_params(row) + "_" + row["movement"]


def open_pkls(fn: str) -> list:
    """Open pickled list of dictionaries containing model scores."""
    with open(fn, "rb") as f:
        D = pickle.load(f, fix_imports=True, encoding="latin1")
    return D


def remove_brackets(s):
    s = s.replace('(', '').replace(')', '')
    try:
        return int(s)
    except ValueError:
        return s


inparenthesis = re.compile('\([a-zA-Z0-9\-]*\)')
beforeparenthesis = re.compile('[a-zA-Z0-9\-]*\(')


def parse_filename(s: str) -> pd.Series:
    """
    Finds numbers in brackets, associates this value
    with a key in front of the bracket.
    Returns Series like {'model': 'vcgpdm',...}
    """
    A = map(remove_brackets, re.findall(inparenthesis, s))
    B = map(remove_brackets, re.findall(beforeparenthesis, s))
    return pd.Series({b.replace('-', ''): a for a, b in zip(A, B)})
