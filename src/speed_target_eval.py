'''
python main.py -info drq -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -critic_updates_per_step 20 -method sac -target_entropy -1.0 -target_drop_rate 0.005 -layer_norm 1
droq-8400s, sac=8863s, redq= 2x time
python main.py -info sac -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10  -gpu_id 0 -critic_updates_per_step 20 -method sac -target_entropy -1.0
python main.py -info drq -env FishStationary-v0 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -critic_updates_per_step 20 -method sac -target_entropy 0 -target_drop_rate 0.005 -layer_norm 1
'''
import os
import sys
sys.path.insert(0, "../")
import argparse
import datetime
import gym
from src.agent_async import SacAgentAsync
#from IQNagent import IQNSacAgent
from rlutils.envs import *
from src.model import GaussianPolicy
import torch
import gym
from matplotlib import pyplot as plt
import numpy as np
from src.CONFIG import *
import numpy as np
from copy import deepcopy
from rlutils.utils import config_paper
from rlutils.env_wrappers import FishMovingRenderWrapper, VideoRecorderWrapper
c = config_paper()
EP_STEPS = 768
# path='/home/sardor/1-THESE/4-sample_code/00-current/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning/runs/droq/FishMovingTargetSpeed-v0_2023-11-08/model/policy.pth'
path= '../eval/runs/FishMovingTargetSpeed-v0_2023-11-22/model/policy.pth'
path= '../eval/runs/FishMovingTargetSpeed-v0_2023-11-23/model/policy.pth'


hidden_units=[256, 256]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'using device: {device}')
env = FishMovingTargetSpeed(EP_STEPS=3000)
env = FishMovingRenderWrapper(env, renderWindow=True)
policy = GaussianPolicy(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            hidden_units=hidden_units).to(device)

policy.load_state_dict(torch.load(path))
done = False
r_arr, obs_arr, acts, consigne = [], [], [], []
i = 0
obs=env.reset()
env.state = obs
env.target = 1
c=0

obs = np.zeros_like(obs)
env.state = obs
while not done:
    act = policy.sample(torch.FloatTensor(obs).to(device))[-1].detach().item()
    obs, rew, done, _ = env.step(act)
    r_arr.append(rew)
    obs_arr.append(obs)
    acts.append(act)
    consigne.append(env.target)
    i += 1


#take the first 400 steps
obs_arr = np.array(obs_arr)
crop_end=800
obs_arr = obs_arr[:crop_end]
r_arr = r_arr[:crop_end]
acts = acts[:crop_end]


fig, ax = plt.subplots(3,1, figsize=(9,6), sharex=True)
axis_labels = ['a', 'b', 'c']
titles = [r'$\dot{x} [a.u]$', r'$\alpha$ [a.u]', 'Control value [a.u]']

for t, a, label, ly  in zip([obs_arr[:,0], obs_arr[:,1], acts], ax[:], axis_labels,titles):
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
plt.savefig(f'./FishMovingTargetSpeedFixedDroq.pdf')
plt.show()