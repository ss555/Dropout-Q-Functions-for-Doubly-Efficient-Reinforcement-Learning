import os
import argparse
import datetime
from agent import SacAgent
from rlutils.utils import *
import socket
from rlutils.linear_expe import make_red_yellow_env_speed, DummyconnectionEnv
import multiprocessing as mp
from rlutils.envs import * #register_envs
from rlutils.env_wrappers import LoggerWrap

tau = 0.05
len_episode = 128

def run():

    process = mp.Process(target=run_rpi)
    process.start()
    time.sleep(7)
    
    monitor_dir, _ = make_dir_exp('./logs')
    HOST = 'raspberrypi.local'  # '192.168.0.10'  # IP address of Raspberry Pi
    PORT = 8080  
    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_FPS, 30)
    print(vid.get(cv2.CAP_PROP_FPS))
    _, obs = vid.read()
    assert obs.any() != None
    # create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # connect to the server
    s.connect((HOST, PORT))
    # MAXIMIZE SPEED ENV
    env, params = make_red_yellow_env_speed(vid, s, monitor_dir, len_episode=len_episode, tau=tau, discrete_actions=False, phi=30, sb3=False)

    #UNCOMMENT FOR DUMMY ENV
    # dummy ENV
    # env = DummyconnectionEnv(vid, s, monitor_dir)
    # env = LoggerWrap(env, path=monitor_dir, pickle_images=False)
    # env = TimeLimit(env, max_episode_steps=len_episode)

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
    'critic_updates_per_step': 20,#20,
    'eval_episodes_interval': 50,
    'gradients_step': 128,#20,
    'start_steps': 0,
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
    try:
        agent = SacAgent(env=env, **configs)
        agent.run()
    except:
        traceback.print_exc()
    finally:
        s.close()
        vid.release()
        cv2.destroyAllWindows()
        


if __name__ == '__main__':
    run()
