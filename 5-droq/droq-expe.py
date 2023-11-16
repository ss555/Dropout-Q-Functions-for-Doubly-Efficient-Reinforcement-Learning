import os
import argparse
import datetime
from agent import SacAgent
from rlutils.utils import *
import socket
from rlutils.linear_expe import make_red_yellow_env_speed
import multiprocessing as mp
from rlutils.envs import * #register_envs
from rlutils.utils import *
# register_envs()

tau = 0.05

def run():

    set_high_priority()
    monitor_dir, _ = make_dir_exp(os.path.abspath(os.path.join(os.path.dirname(__file__), './logs')))  # '../docs/weightsParams/ppo.yml')
    print(monitor_dir)

    process = mp.Process(target=run_rpi)
    process.start()
    time.sleep(4)
    
    HOST = 'raspberrypi.local'  # '192.168.0.10'  # IP address of Raspberry Pi
    PORT = 8080  # same arbitrary port as on server
    vid = cv2.VideoCapture(0)
    _, obs = vid.read()
    assert obs.any() != None
    # create a socket object
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # connect to the server
    s.connect((HOST, PORT))
    # MAXIMIZE SPEED ENV
    # env, params = make_red_yellow_env_speed(vid, s, monitor_dir, len_episode=90, tau=tau, discrete_actions=False, phi=30, sb3=False)

    # dummy ENV
    env = DummyconnectionEnv(s)

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

    label = "fish_exp-speed-" + str(datetime.now()).split(" ")[0]
    # label = args.env + "_" + str(datetime.datetime.now())
    log_dir = os.path.join('runs', label)

    agent = SacAgent(env=env, log_dir=log_dir, **configs)
    agent.run()


if __name__ == '__main__':
    run()
