'''
determines the best frequency for the fish to move from csv logs-fish
perform fft to find the frequency with the highest rew
'''
from rlutils.envs import *
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GaussianPolicy
import torch
import gym
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import glob
from rlutils.env_wrappers import RlTrainer
from rlutils.utils import *
from rlutils.plot_utils import *

path= '../logs/152'
total_rews=[]
dfs=[]

for f in glob.glob(os.path.join(path, '*log_ep*')):
    df = pd.read_csv(f).dropna()
    total_rews.append(df['reward'].mean())
    dfs.append(df)
assert dfs!=[], f'path: {path}'
total_rews = np.array(total_rews)
idx_max = np.argmax(total_rews)
dfm = dfs[idx_max]
acts=dfm['action'].to_numpy()
obs=dfm['obs'].to_numpy()
rew=dfm['reward'].to_numpy()

# Apply this function to the desired column
df['action'] = df['action'].apply(convert_to_float)
acts = np.array([float(a.strip('[]')) for a in acts]).reshape(-1)
dt=df['time'].diff().mean()
power,sample_freq,freqs,pos_mask=fft_power(acts,dt=dt)
plot_fft(power,sample_freq,freqs,pos_mask,inset=False)
maxfreq=fft_plot_ampltitude(acts,dt)

#plot rew,obs,act
fig, ax = plt.subplots(2,1)
ax[0].plot(rew)
ax[1].plot(acts)
ax[0].set_title(f'best frequency is {maxfreq} with {sum(rew)}')
# plt.title(f'best frequency is {maxfreq} with {sum(rew)}')
plt.savefig(os.path.join(path, f"stats-rl.pdf"))
plt.show()
print(f'best frequency is {maxfreq}')

'''
#fft plot
# Define the sampling time and the frequency
sampling_time = 0.02  # 20 ms sampling time
frequency_1hz = 1  # 1 Hz frequency
frequency_5hz = 5  # 5 Hz frequency

# Create a time array
t = np.arange(0, 2, sampling_time)  # 2 seconds duration

# Generate the 1Hz and 5Hz sine waves
sin_1hz = np.sin(2 * np.pi * frequency_1hz * t)
sin_5hz = np.sin(2 * np.pi * frequency_5hz * t)

# Modulate the two sine waves
modulated_signal = sin_1hz + sin_5hz
power,sample_freq,freqs,pos_mask=fft_power(modulated_signal,dt=sampling_time)
plot_fft(power,sample_freq,freqs,pos_mask,inset=True)
'''