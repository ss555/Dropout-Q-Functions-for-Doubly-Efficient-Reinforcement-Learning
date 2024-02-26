'''
finds best episode and plots the time difference
'''
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import subprocess
from rlutils.utils import load_data#,is_integer

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

filedirs=sorted([int(f) for f in os.listdir('./logs') if is_integer(f)])
print(filedirs)
logpath=os.path.join('./logs',str(filedirs[-1]))
print(f'Loading data from {logpath}')
dfs, names = load_data(str(logpath))

diff_times = []
os.makedirs(os.path.join(logpath, 'figs_dt'), exist_ok=True)
for i, df in zip(names, dfs):
    ep_len = len(df['time'])
    if len(df['time']) < 90:
        print(f'len:{ep_len}-{i}')
    diff_times.append(df['time'].diff().mean())
    fig, ax = plt.subplots()
    dft_diff = df['time'].diff()
    idx_remove = dft_diff.nlargest(3).index
    dft_diff = dft_diff.drop(idx_remove)
    plt.plot(dft_diff)
    plt.title('Episode: ' + str(i))
    plt.xlabel('Step')
    plt.ylabel('Time')
    plt.savefig(os.path.join(logpath, 'figs_dt/episode_' + str(i) + '.png'))


print(np.array(diff_times).mean())