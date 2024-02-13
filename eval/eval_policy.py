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
from src.agent import SacAgent
from rlutils.envs import *
from rlutils.env_wrappers import FishMovingRenderWrapper,RlTrainer
from src.model import GaussianPolicy
import torch
import gym
from matplotlib import pyplot as plt
import numpy as np
from src.CONFIG import *
from rlutils.plot_utils import fft_plot_ampltitude

# env_name='FishStationary-v0'
env_name='FishMoving-v0'
# env_name='FishMovingTargetSpeed-v0'#[x_dot, alpha, alpha_dot, x_dd, alpha_dd]

if env_name=='FishStationary-v0':
    configs = FISH_STATIONARY_CONFIG
    path = '../logs/179/model/policy.pth'
    path = '../logs/180/model/policy.pth'
    freq_arr_scan = np.arange(0.1, 2.5, 0.1)
elif env_name=='FishMoving-v0':
    freq_arr_scan = np.arange(0.1, 2.5, 0.1)
    configs = FISH_MOVING_CONFIG
    path = '/../eval/runs/FishMoving-v0_2023-06-19/model/policy.pth'
    path = '/home/sardor/1-THESE/4-sample_code/00-current/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning/eval/runs/FishMoving-v0_2023-06-19/model/policy.pth'
elif env_name=='FishMovingTargetSpeed-v0':
    freq_arr_scan = np.arange(0.1, 2.5, 0.1)
    configs = FISH_MOVING_CONFIG
    path = '../eval/runs/FishMovingTargetSpeed-v0_2023-11-23/model/policy.pth'

hidden_units=[256, 256]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make(env_name)

policy = GaussianPolicy(
            env.observation_space.shape[0],
            env.action_space.shape[0],
            hidden_units=hidden_units).to(device)
policy.load_state_dict(torch.load(path))

done = False
r_arr, obs_arr, acts, consigne = [], [], [], []
i = 0
obs=env.reset()
env.target = 1
c=0

obs = np.zeros_like(obs)
env.state = obs
while not done:
    act = policy.sample(torch.FloatTensor(obs).to(device))[-1].detach().item()
    obs, rew, done, _ = env.step(act)
    r_arr.append(rew)
    if env_name=='FishStationary-v0':
        obs_arr.append(np.append(obs,env.unwrapped.angleState[0]))
    else:
        obs_arr.append(obs)
    acts.append(act)
    consigne.append(env.target)
    i += 1

obs_arr, r_arr, acts = np.array(obs_arr), np.array(r_arr), np.array(acts)
dt = env.tau if hasattr(env, 'tau') else 0.02
maxfreq = fft_plot_ampltitude(np.array(acts),dt)
print(f'using sampling time: {dt}')
#freq forcing and plot the RL reward
trainer = RlTrainer(env=env)
trainer.mode_freq_forcing(freq_arr=freq_arr_scan, freq_rl=maxfreq, rew_rl=r_arr.sum(),
                          plt_save_name= f'./eval/{env_name}_eval_fscan.pdf',dt=dt)

#take the first 400 steps
obs_arr = np.array(obs_arr)
# obs_arr = obs_arr[:400]
# r_arr = r_arr[:400]
# acts = acts[:400]

if env_name=='FishStationary-v0':
    fig, ax = plt.subplots(3,1,figsize=(9,6),sharex=True)
    # Define labels for each subplot
    axis_labels = ['a', 'b', 'c']
    titles = [r'$\alpha$', r'$F_x$', 'Control value [a.u]']
    # titles = [r'$\dot{F}_y$', r'$F_x$', 'Control value [a.u]']
    crop_val=400
    for t, a, label, ly  in zip([obs_arr[:,-1], r_arr, acts], ax[:], axis_labels,titles):
        plt.locator_params(nbins=6)
        t=t[:crop_val]
        a.plot(t)
        a.text(0.015, 0.95, f'{label})', transform=a.transAxes, fontsize=14, fontweight='bold', va='top')
        a.set_ylabel(ly)
    plt.xlabel('time steps')
    plt.suptitle(f'FishStationary-v0-DROQ-rew:{sum(r_arr)}')
    os.makedirs('eval', exist_ok=True)
    plt.savefig(f'./eval/{env_name}_eval.pdf')
    plt.show()

elif env_name=='FishMoving-v0':
    fig, ax = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    # Define labels for each subplot
    axis_labels = ['a', 'b', 'c']
    titles = [r'$\dot{x} [a.u]$', r'$\alpha$ [a.u]', 'Control value [a.u]']
    crop_val = 400
    for t, a, label, ly in zip([obs_arr[:, 1], obs_arr[:, 2], acts], ax[:], axis_labels, titles):
        plt.locator_params(nbins=6)
        t = t[:crop_val]
        a.plot(t)
        if ly == titles[0]:
            a.axhline(y=1, color='r', linestyle='--')
        elif ly == titles[1]:
            a.axhline(y=0, color='r', linestyle='--')
        # Add the label to the plot
        a.text(0.015, 0.95, f'{label})', transform=a.transAxes, fontsize=14, fontweight='bold', va='top')
        a.set_ylabel(ly)
    plt.xlabel('time steps')
    plt.suptitle(f'{env_name}-DROQ-rew:{sum(r_arr)}')
    os.makedirs('eval', exist_ok=True)
    plt.savefig(f'./eval/{env_name}_eval.pdf')
    plt.show()

elif env_name=='FishMovingTargetSpeed-v0':
    fig, ax = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    # Define labels for each subplot
    axis_labels = ['a', 'b', 'c']
    titles = [r'$\dot{x} [a.u]$', r'$\alpha$ [a.u]', 'Control value [a.u]']
    crop_val = 400
    for t, a, label, ly in zip([obs_arr[:, 0], obs_arr[:, 1], acts], ax[:], axis_labels, titles):
        plt.locator_params(nbins=6)
        t = t[:crop_val]
        a.plot(t)
        if ly == titles[0]:
            a.axhline(y=1, color='r', linestyle='--')
        elif ly == titles[1]:
            a.axhline(y=0, color='r', linestyle='--')
        # Add the label to the plot
        a.text(0.015, 0.95, f'{label})', transform=a.transAxes, fontsize=14, fontweight='bold', va='top')
        a.set_ylabel(ly)
    plt.xlabel('time steps')
    plt.suptitle(f'{env_name}-DROQ-rew:{sum(r_arr)}')
    os.makedirs('eval', exist_ok=True)
    plt.savefig(f'./eval/{env_name}_eval.pdf')
    plt.show()