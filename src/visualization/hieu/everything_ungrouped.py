"""
Check if plot is the same as in Hieu's notebook
"""
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_json('data/processed/processed_hieu.json')
d = df.groupby('fn').mean()
d['mp_type'] = df.groupby('fn').first().mp_type
sns.scatterplot(data=d, x='MSE', y='y', hue='mp_type')
plt.xlim(0, 700)
plt.show()
