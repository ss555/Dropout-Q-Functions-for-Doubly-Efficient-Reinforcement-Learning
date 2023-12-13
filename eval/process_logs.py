
# logpath='/home/sardor/1-THESE/4-sample_code/00-current/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning/logs/105'
import os
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import subprocess
from rlutils.utils import load_data
import matplotlib
from rlutils.utils import plot_data_from_dirs_exp_linear
# matplotlib.use("Agg")
logpath = '../logs/152'
plot_all_episodes=False

#plot best episode
dfs, names = load_data(str(logpath))
rews=[sum(d['reward']) for d in dfs]
best_episode=names[np.argmax(rews)]
print(f'best episode : {best_episode}, reward : {np.max(rews)}')
df=dfs[np.argmax(rews)]
plot_data_from_dirs_exp_linear(df,os.path.join(logpath,'figs/episode_'+str(best_episode)+'.png'),show=True)

if plot_all_episodes:
    dfs, names = load_data(str(logpath))
    os.makedirs(os.path.join(logpath,'figs'),exist_ok=True)

    for i,df in zip(names, dfs):
        plot_data_from_dirs_exp_linear(df,os.path.join(logpath,'figs/episode_'+str(i)+'.png'))
    