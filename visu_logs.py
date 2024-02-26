'''
finds best episode and plots the obs/actions/rewards for all agents
'''
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import subprocess
from rlutils.utils import load_data,is_integer
from rlutils.utils import plot_data_from_dirs_exp_linear

# filedirs=sorted([int(f) for f in os.listdir('./logs') if is_integer(f)])#TAKE LAST of list of all dirs
# print(filedirs)
# logpath=os.path.join('./logs',str(filedirs[-1]))

# logpath='/home/sardor/1-THESE/4-sample_code/00-current/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning/logs/152'
logpath='/home/sardor/1-THESE/4-sample_code/00-current/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning/logs/182'
#load all processed data OR PROCESS
dfs, names = load_data(str(logpath))
os.makedirs(os.path.join(logpath,'figs'),exist_ok=True)
rews=[sum(df['reward']) for df in dfs]
best=np.argmax(rews)
print(f'max return : {names[best]} : {rews[best]}')


#best episode
plot_data_from_dirs_exp_linear(dfs[best],os.path.join(logpath,'figs/best_episode_'+str(best)+'.png'),show=True)
#plot all data
# sys.exit(0)

for i,df in zip(names, dfs):
    plot_data_from_dirs_exp_linear(df,os.path.join(logpath,'figs/episode_'+str(i)+'.png'))