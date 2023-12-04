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
from rlutils.envs import *
from rlutils.utils import *
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname('./..'))
from model import GaussianPolicy
import torch
import gym
from matplotlib import pyplot as plt
import numpy as np
from copy import deepcopy
from rlutils.utils import config_paper
from agent import SacAgent

EP_STEPS = 768
# path='/home/sardor/1-THESE/4-sample_code/00-current/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning/runs/droq/FishMovingTargetSpeed-v0_2023-11-08/model/policy.pth'
path='./runs/FishMovingTargetSpeed-v0_2023-11-22/model/'

configs = {'num_steps': 100000,
               'batch_size': 1024,
               'lr': 0.0003,
               'hidden_units': [256, 256],
               'memory_size': 1000000.0,
               'gamma': 0.99,
               'tau': 0.005,
               'entropy_tuning': True,
               'ent_coef': 0.2,
               'multi_step': 1,
               'per': 0,#1,
               'alpha': 0.6,
               'beta': 0.4,
               'beta_annealing': 3e-07,
               'grad_clip': None,
               'critic_updates_per_step': 100,#20,
               'gradients_step': 768,#20,
                'eval_episodes_interval': 10,
               'start_steps': 0,
               'log_interval': 10,
               'target_update_interval': 1,
               'cuda': 0,
               'seed': 0,
               'eval_runs': 3,
               'huber': 0,
               'layer_norm': 1,
               'target_entropy': -1.0,
               'method': 'sac',
               'target_drop_rate': 0.005,
               'critic_update_delay': 1}

os.makedirs('./logs', exist_ok=True)
monitor_dir, _ = make_dir_exp(os.path.abspath(os.path.join(os.path.dirname(__file__), './logs')))
hidden_units=[256, 256]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'using device: {device}')
env = FishMovingTargetSpeed(EP_STEPS=3000)

agent=SacAgent(env=env, resume_training_path=path, log_dir=monitor_dir,**configs)
agent.run()

