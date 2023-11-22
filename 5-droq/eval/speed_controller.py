'''
python main.py -info drq -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -critic_updates_per_step 20 -method sac -target_entropy -1.0 -target_drop_rate 0.005 -layer_norm 1
droq-8400s, sac=8863s, redq= 2x time
python main.py -info sac -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10  -gpu_id 0 -critic_updates_per_step 20 -method sac -target_entropy -1.0
python main.py -info drq -env FishStationary-v0 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -critic_updates_per_step 20 -method sac -target_entropy 0 -target_drop_rate 0.005 -layer_norm 1
'''
import os
import argparse
import datetime
import gym
from agent import SacAgent
from util.utilsTH import SparseRewardEnv
#from IQNagent import IQNSacAgent
import customenvs
customenvs.register_mbpo_environments()
from agent4profile import SacAgent4Profile
from rlutils.envs import *
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import GaussianPolicy
import torch
import gym
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy

EP_STEPS = 2000
path='/home/sardor/1-THESE/4-sample_code/00-current/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning/runs/droq/FishMovingTargetSpeed-v0_2023-11-08/model/policy.pth'

configs = {'num_steps': 100000,
    'batch_size': 256,
    'lr': 0.0003,
    'hidden_units': [256, 256],
    'memory_size': 1000000.0,
    'gamma': 0.99,
    'tau': 0.005,
    'entropy_tuning': True,
    'ent_coef': 0.2,
    'multi_step': 1,
    'per': 0,
    'alpha': 0.6,
    'beta': 0.4,
    'beta_annealing': 3e-07,
    'grad_clip': None,
    'critic_updates_per_step': 20,
    'start_steps': 5000,
    'log_interval': 10,
    'target_update_interval': 1,
    'eval_interval': 1000,
    'cuda': 0,
    'seed': 0,
    'eval_runs': 1,
    'huber': 0,
    'layer_norm': 1,
    'target_entropy': -1.0,
    'method': 'sac',
    'target_drop_rate': 0.005,
    'critic_update_delay': 1}

hidden_units=[256, 256]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
env = FishMovingTargetSpeed(EP_STEPS=3000) #gym.make('FishMovingTargetSpeed-v0')

policy = GaussianPolicy(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            hidden_units=hidden_units).to(device)

policy.load_state_dict(torch.load(path))
done = False
r_arr, obs_arr, acts, consigne = [], [], [], []
i = 0
obs=env.reset()
targets = [2,1.5,1,0.5,0]
targets = [1,1.5,2,0.5,0]
env.target = targets[0]
c=0
while not done:
    obs_m = deepcopy(obs)
    obs_m[1] = obs_m[1]+1-env.target
    act = policy.sample(torch.FloatTensor(obs_m).to(device))[-1].detach().item()
    obs, rew, done, _ = env.step(act)
    r_arr.append(rew)
    obs_arr.append(obs)
    acts.append(act)
    consigne.append(env.target)
    i += 1
    if i%300==0:
        c += 1
        env.target = targets[c%len(targets)]


#take the first 400 steps
obs_arr = np.array(obs_arr)
crop_end=800
obs_arr = obs_arr[:crop_end]
r_arr = r_arr[:crop_end]
acts = acts[:crop_end]


fig, ax = plt.subplots(3,1,figsize=(9,6),sharex=True)
axis_labels = ['a', 'b', 'c']
titles = [r'$\dot{x} [a.u]$', r'$\alpha$ [a.u]', 'Control value [a.u]']

crop_val=400
for t, a, label, ly  in zip([obs_arr[:,1], obs_arr[:,2], acts], ax[:], axis_labels,titles):
    plt.locator_params(nbins=6)
    a.plot(t)
    if ly==titles[0]:
        a.plot(consigne[:crop_end], color='r', linestyle='--')
    elif ly==titles[1]:
        a.axhline(y=0, color='r', linestyle='--')
    # Add the label to the plot
    a.text(0.015, 0.95, f'{label})', transform=a.transAxes, fontsize=14, fontweight='bold', va='top')
    a.set_ylabel(ly)

plt.xlabel('time steps')
plt.savefig(f'./FishMovingTargetSpeedDroq.pdf')
plt.show()

fig, ax = plt.subplots(3,1,sharex=True)
fig.suptitle(f'droq:{np.sum(r_arr)}')
ax[0].plot(acts)
ax[1].plot(obs_arr[:,2])
ax[2].plot(obs_arr[:,1])
plt.xlabel('time step')
plt.show()
