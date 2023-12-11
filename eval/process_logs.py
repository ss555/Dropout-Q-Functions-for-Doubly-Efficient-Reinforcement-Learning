
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
matplotlib.use("Agg")
logpath = '../logs/136'

dfs, names = load_data(str(logpath))
os.makedirs(os.path.join(logpath,'figs'),exist_ok=True)
from rlutils.utils import plot_data_from_dirs_exp_linear
for i,df in zip(names, dfs):
    plot_data_from_dirs_exp_linear(df,os.path.join(logpath,'figs/episode_'+str(i)+'.png'))    
    