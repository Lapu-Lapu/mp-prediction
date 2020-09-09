from src.models.utils import load_data
import pymc3 as pm
import numpy as np
import pickle


df, pmps, participants, mp_types, participant_to_idx, mp_type_to_idx, pmp_to_idx = load_data()


with pm.Model() as mult_mixed:
    a_bar = pm.Normal('a_bar', 2, 1)
    sigma_a = pm.Exponential('sigma_a', 1)
    b_bar = pm.Normal('b_bar', -5, 1.5)
    sigma_b = pm.Exponential('sigma_b', 1)
    # c_bar = pm.Normal('c_bar', 0, 1.5)
    sigma_c = pm.Exponential('sigma_c', 1)
    a = pm.Normal('a', a_bar, sigma_a, shape=len(participants))
    b = pm.Normal('b', b_bar, sigma_b, shape=len(mp_types))
    c = pm.Normal('c', 0, 1, shape=len(mp_types))
    # x = d['partialMSE'] - d['partialMSE'].mean()
    # x = x / x.max()
    x = df['partialMSE']
    p = 0.5 / (1 + np.exp(-(a[df.id_participant] + sigma_c * c[df.id_mptype] + b[df.id_mptype] * x)))
    # p = 0.5 / (1 + np.exp(-(a_bar + sigma_a * a[df.id_participant] + sigma_c * c[df.id_mptype] + b[df.id_mptype] * x)))
    # p = 0.5 / (1 + np.exp(-(a_bar + sigma_a* a[df.id_participant] + sigma_c*c[df.id_mptype] + (b_bar + sigma_b * b[df.id_mptype]) * x)))
    s = pm.Bernoulli('s', p=p, observed=df['result'])
    trace_multimixed = pm.sample(1000, tune=1000, init='adapt_diag')


with open("data/processed/multilevel_trace.pkl", "wb") as fh:
    pickle.dump((mult_mixed, trace_multimixed), fh)

import matplotlib.pyplot as plt

pps = pm.sample_prior_predictive(model=mult_mixed)
x = np.linspace(-0.5, 2)
a = pps['a']
b = pps['b']
for i in range(200):
    y = 0.5 / (1 + np.exp(-(a[i, 0] + b[i, 0] * x)))
    plt.plot(x, y, alpha=0.1)
plt.show()
