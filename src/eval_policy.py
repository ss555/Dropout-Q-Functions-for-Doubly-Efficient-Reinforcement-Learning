'''
python main.py -info drq -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -critic_updates_per_step 20 -method sac -target_entropy -1.0 -target_drop_rate 0.005 -layer_norm 1
droq-8400s, sac=8863s, redq= 2x time
python main.py -info sac -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10  -gpu_id 0 -critic_updates_per_step 20 -method sac -target_entropy -1.0
python main.py -info drq -env FishStationary-v0 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -critic_updates_per_step 20 -method sac -target_entropy 0 -target_drop_rate 0.005 -layer_norm 1
'''
import os
import sys
sys.path.insert(0, "./..")
import argparse
import datetime
import gym
from agent import SacAgent
#from IQNagent import IQNSacAgent
from rlutils.envs import *
from model import GaussianPolicy
import torch
import gym
from matplotlib import pyplot as plt
import numpy as np
from CONFIG import *

env_name='FishStationary-v0'
# env_name='FishMovingTargetSpeed-v0'
path='/home/install/Project/00-current/00-current/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning/logs/179/model/policy.pth'

if env_name=='FishStationary-v0':
    configs = FISH_STATIONARY_CONFIG
else:
    configs = FISH_MOVING_CONFIG

hidden_units=[256, 256]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make(env_name)

policy = GaussianPolicy(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            hidden_units=hidden_units).to(device)

policy.load_state_dict(torch.load(path))
done = False
r_arr, obs_arr, acts = [], [], []
i = 0
obs=env.reset()
while not done:
    act= policy.sample(torch.FloatTensor(obs).to(device))[-1].detach().item()
    obs, rew, done, _ = env.step(act)
    r_arr.append(rew)
    obs_arr.append(obs)
    acts.append(act)
    i += 1

#take the first 400 steps
obs_arr = np.array(obs_arr)
obs_arr = obs_arr[:400]
r_arr = r_arr[:400]
acts = acts[:400]


fig, ax = plt.subplots(3,1,figsize=(9,6),sharex=True)
# Define labels for each subplot
axis_labels = ['a', 'b', 'c']
titles = [r'$\dot{x} [a.u]$', r'$\alpha$ [a.u]', 'Control value [a.u]']
crop_val=400
for t, a, label, ly  in zip([obs_arr[:,1], obs_arr[:,2], acts], ax[:], axis_labels,titles):
    plt.locator_params(nbins=6)
    t=t[:crop_val]
    a.plot(t)
    if ly==titles[0]:
        a.axhline(y=1, color='r', linestyle='--')
    elif ly==titles[1]:
        a.axhline(y=0, color='r', linestyle='--')
    # Add the label to the plot
    a.text(0.015, 0.95, f'{label})', transform=a.transAxes, fontsize=14, fontweight='bold', va='top')
    a.set_ylabel(ly)
plt.xlabel('time steps')
os.makedirs('eval', exist_ok=True)
plt.savefig(f'./eval/{env_name}_eval.pdf')
plt.show()





