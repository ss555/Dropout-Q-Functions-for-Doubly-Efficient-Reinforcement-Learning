import glob
import os.path
import matplotlib.pyplot as plt
import numpy as np
from rlutils.utils import *
from rlutils.vision import *
import pickle
import traceback
from rlutils.plot_utils import config_paper
import os
import argparse
import datetime
from agent import SacAgent
from rlutils.utils import *
import socket
from rlutils.linear_expe import *
import multiprocessing as mp
from rlutils.envs import * #register_envs
from rlutils.utils import *

# register_envs()
videofile='/home/install/Project/00-current/00-current/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning/logs/105/ep-0-vid-.mp4'
frames=read_video_to_frames(videofile)
print(frames[0].shape)
len_episode=100
tau = 0.05
monitor_dir, _ = make_dir_exp(os.path.abspath(os.path.join(os.path.dirname(__file__), './logs')))  # '../docs/weightsParams/ppo.yml')
print(monitor_dir)
vid = cv2.VideoCapture(0)
_, obs = vid.read()
assert obs.any() != None

s=vid
env, params = make_red_yellow_env_speed_debug(vid, s, monitor_dir, len_episode=len_episode, tau=tau, discrete_actions=False, phi=40, sb3=False)
state_arr=[]
env.reset()

for obs in frames:
    assert len(obs.shape)==3, f'obs must be 3d: {obs.shape}'
    assert obs.shape[-1]==3, 'obs must be 3 channels'+str(obs.shape)
    state_arr.append(env.step(obs)[0])    
    # state_arr.append(env.process_obs(obs)[0])    

state_arr=np.array(state_arr)
print(state_arr.shape)
print(state_arr[0][0])
fig, ax = plt.subplots(4,1)
ax[0].plot(state_arr[:,-3])
ax[1].plot(state_arr[:,-2])
ax[2].plot(state_arr[:,-1])
ax[3].plot(state_arr[:,-5])
# plt.plot(state_arr[:,-3])
# # plt.plot(state_arr[:,-2])
# plt.plot(state_arr[:,-1])
# # plt.plot(state_arr[:,-0])
plt.show()

env.reset()
# env, params = make_red_yellow_env_speed(vid, s, monitor_dir, len_episode=len_episode, tau=tau, discrete_actions=False, phi=40, sb3=False)
# state_arr=[]
# # env = DummyconnectionEnv(s)
# for obs in frames:
#     env.calc_state(obs)
#     state_arr.append(env.unwrapped.state)

# state_arr=np.array(state_arr)
# plt.plot(state_arr[:,0])
# plt.show()