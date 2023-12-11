import os
import argparse
import datetime
import gym
from agent import SacAgent
from rlutils.envs import *
from rlutils.env_wrappers import LoggerWrap
from rlutils.utils import *
from rlutils.envs import *
from utils import grad_false, hard_update, soft_update, to_batch, update_params, RunningMeanStats
import torch.utils.tensorboard as tf

def run():
    len_episode = 128
    env = dummyEnv()

    os.makedirs('./logs', exist_ok=True)
    monitor_dir, _ = make_dir_exp(os.path.abspath(os.path.join(os.path.dirname(__file__), './logs')))
    print(monitor_dir)
    env = LoggerWrap(env, path=monitor_dir, pickle_images=False)
    env = TimeLimit(env, max_episode_steps=768)

    configs = {'num_steps': 100000,
    'batch_size': 256,#512,#256,
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
    'critic_updates_per_step': 20,#20,
    'eval_episodes_interval': 50,
    'gradients_step': len_episode,#20,
    'start_steps': 500,
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
    'log_dir': monitor_dir,
    'critic_update_delay': 1}


    resume_training_path='./logs/136/model'
    agent = SacAgent(env=env, resume_training_path=resume_training_path,**configs)
    agent.writer.add_text('metrics', json.dumps(configs, indent=4), 0)
    save_yaml_dict(configs, os.path.join(monitor_dir, 'configs.yaml'))
    dfs, names = load_data('./logs/138', drop_reset_observation=False)

    for df in dfs:
        next_state = None
        for i in range(1,len(df)):
            if next_state is None:
                array = list(filter(None, df['obs'][i - 1].replace('[', '').replace(']', '').replace('\n', '').split(' ')))
                state = [float(x) for x in array]
            else:
                state = next_state

            array = list(filter(None, df['obs'][i].replace('[', '').replace(']', '').replace('\n', '').split(' ')))
            next_state = [float(x) for x in array]
            action = df['action'][i]
            action = float(action.strip('[]'))
            reward = df['reward'][i]
            done = False
            masked_done = False
                    
        if agent.per:
            batch = to_batch(state, action, reward, next_state, masked_done,
                agent.device)
            with torch.no_grad():
                curr_q1, curr_q2 = agent.calc_current_q(*batch)
            target_q = agent.calc_target_q(*batch)
            error = (0.5 * torch.abs(curr_q1 - target_q) + 0.5 * torch.abs(curr_q2 - target_q)).item()
            # We need to give true done signal with addition to masked done
            # signal to calculate multi-step rewards.
            agent.memory.append(state, action, reward, next_state, masked_done, error, episode_done=done)
        else:
            agent.memory.append(state, action, reward, next_state, masked_done, episode_done=done)
        agent.learn()
        agent.episodes_num += 1
        agent.steps += 128
        print(f'offline episode {i} done')

if __name__ == '__main__':
    run()