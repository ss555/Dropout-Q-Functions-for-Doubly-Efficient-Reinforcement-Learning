
import sys
import os
sys.path.append(os.path.abspath('./src/'))
from agent_async import SacAgentAsync
from CONFIG import *
from rlutils.utils import *
from rlutils.envs import *
import os
import gym
from rlutils.utils import *
import socket
from rlutils.linear_expe import make_red_yellow_env_speed, DummyconnectionEnv
import multiprocessing as mp
from rlutils.envs import * #register_envs
from rlutils.env_wrappers import LoggerWrap
from gym.wrappers import NormalizeObservation, NormalizeReward

def run():
    # env_name='FishMovingTargetSpeed-v0'
    # env_name='FishStationary-v0'
    # env_name='FishMovingTargetSpeedController-v0'
    env_name='FishMoving-v0'
    # env_name='FishMovingVisualServoContinousSparse-v0'

    env = gym.make(env_name)
    os.makedirs('./logs', exist_ok=True)
    monitor_dir, _ = make_dir_exp(os.path.abspath(os.path.join(os.path.dirname(__file__), './logs')))
    print(monitor_dir)
    # env = LoggerWrap(env, path=monitor_dir, pickle_images=False)
    env = TimeLimit(env, max_episode_steps=768)
    if env_name=='FishStationary-v0':
        configs = FISH_STATIONARY_CONFIG_test
    else:
        configs = FISH_MOVING_CONFIG

    try:
        env._max_episode_steps = env.wrapped_env._max_episode_steps
    except:
        env._max_episode_steps = configs['gradients_step']

    configs.update({'gradients_step': 2 * 768})
    agent = SacAgentAsync(env=env, log_dir=monitor_dir, **configs)
    configs.update({'env_name': env_name,'agent':agent.__class__.__name__})
    save_yaml_dict(configs, os.path.join(monitor_dir, 'configs.yaml'))
    try:
        agent.run()
    except:
        traceback.print_exc()


if __name__ == '__main__':
    run()