import numpy as np


def beta_std(s):
    p = s.sum() + 1
    q = len(s) - s.sum() + 1
    return np.sqrt(p * q / (p + q + 1) / (p + q)**2)


def center_data(d, key='partialMSE'):
    x = d[key] - d[key].mean()
    return x/x.max()
