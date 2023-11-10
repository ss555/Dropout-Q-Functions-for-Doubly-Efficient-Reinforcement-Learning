'''
python main.py -info drq -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0 -target_drop_rate 0.005 -layer_norm 1
droq-8400s, sac=8863s, redq= 2x time
python main.py -info sac -env Hopper-v2 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10  -gpu_id 0 -updates_per_step 20 -method sac -target_entropy -1.0
python main.py -info drq -env FishStationary-v0 -seed 0 -eval_every 1000 -frames 100000 -eval_runs 10 -gpu_id 0 -updates_per_step 20 -method sac -target_entropy 0 -target_drop_rate 0.005 -layer_norm 1

'''
import os
import argparse
import datetime
import gym
from agent import SacAgent
from util.utilsTH import SparseRewardEnv
# TODO remove IQN agent part
#from IQNagent import IQNSacAgent
import customenvs
customenvs.register_mbpo_environments()
from agent4profile import SacAgent4Profile
from rlutils.envs import *

path='/home/sardor/1-THESE/4-sample_code/00-current/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning/KUCodebase/code/runs/droq/FishMovingTargetSpeed-v0_2023-11-08/model/policy.pth'

# def run(path,env):
env=gym.make('FishMovingTargetSpeed-v0')
model = SacAgent4Profile(env, path, device='cpu')

