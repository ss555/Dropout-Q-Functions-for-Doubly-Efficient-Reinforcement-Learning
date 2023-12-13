import os
import argparse
import datetime
import gym
from agent import SacAgent
from agent_async import SacAgentAsync
from rlutils.envs import *
from rlutils.env_wrappers import LoggerWrap
from rlutils.utils import *
from rlutils.envs import *

def run():
    env_name='FishMovingTargetSpeed-v0'
    env_name='FishMovingTargetSpeedController-v0'
    # env_name='FishMoving-v0'
    # env_name='FishMovingVisualServoContinousSparse-v0'
    env_name='FishStationary-v0'
    env = gym.make(env_name)
    # env = FishMovingTargetSpeedController(EP_STEPS=768,random_target=True)
    os.makedirs('./logs', exist_ok=True)
    monitor_dir, _ = make_dir_exp(os.path.abspath(os.path.join(os.path.dirname(__file__), './logs')))
    print(monitor_dir)
    env = LoggerWrap(env, path=monitor_dir, pickle_images=False)
    env = TimeLimit(env, max_episode_steps=768)

    configs = {'num_steps': 200000,
               'batch_size': 256,
               'lr': 0.0003,
               'hidden_units': [256, 256],
               'memory_size': 1000000.0,
               'gamma': 0.99,
               'tau': 0.005,
               'entropy_tuning': True,
               'ent_coef': 0.2,
               'multi_step': 1,
               'per': 1,
               'alpha': 0.6,
               'beta': 0.4,
               'beta_annealing': 3e-07,
               'grad_clip': None,
               'critic_updates_per_step': 20,#20,
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
               'critic_update_delay': 1
               }    

    try:
        env._max_episode_steps = env.wrapped_env._max_episode_steps
    except:
        env._max_episode_steps = 768
    # agent = SacAgent(env=env, log_dir=monitor_dir, **configs)
    agent = SacAgentAsync(env=env, log_dir=monitor_dir, **configs)
    configs.update({'env_name': env_name,'agent':agent.__class__.__name__})
    save_yaml_dict(configs, os.path.join(monitor_dir, 'configs.yaml'))
    try:
        agent.run()
    except:
        traceback.print_exc()
        agent.save_buffer()


if __name__ == '__main__':
    run()