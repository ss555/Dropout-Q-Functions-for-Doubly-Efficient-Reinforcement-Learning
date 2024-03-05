import os
import sys

sys.path.append(os.path.abspath('./src/'))
import argparse
import datetime
from agent_async import SacAgentAsync
from rlutils.env_wrappers import LoggerWrap
from rlutils.utils import *
from rlutils.envs import *
from src.utils import to_batch
from CONFIG import config_EXPE

def run():
    len_episode = 128
    env = dummyEnv()

    os.makedirs('./logs', exist_ok=True)
    monitor_dir, _ = make_dir_exp(os.path.abspath(os.path.join(os.path.dirname(__file__), './logs')))
    print(monitor_dir)
    env = LoggerWrap(env, path=monitor_dir, pickle_images=False)
    env = TimeLimit(env, max_episode_steps=768)

    configs = config_EXPE
    configs.update({'log_dir': monitor_dir})
    configs.update({'gradients_step': 2 * len_episode})


    resume_training_path= 'logs/logs_phi_30/152/model'
    resume_training_path=None
    agent = SacAgentAsync(env=env, resume_training_path=resume_training_path,**configs)
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
            batch = to_batch(state, action, reward, next_state, masked_done, agent.device)
            with torch.no_grad():
                curr_q1, curr_q2 = agent.calc_current_q(*batch)
            target_q = agent.calc_target_q(*batch)
            error = (0.5 * torch.abs(curr_q1 - target_q) + 0.5 * torch.abs(curr_q2 - target_q)).item()
            # We need to give true done signal with addition to masked done signal to calculate multi-step rewards.
            agent.memory.append(state, action, reward, next_state, masked_done, error, episode_done=done)
        else:
            agent.memory.append(state, action, reward, next_state, masked_done, episode_done=done)
        agent.learn()
        agent.episodes_num += 1
        agent.steps += 128
        print(f'offline episode {i} done')

if __name__ == '__main__':
    run()