import os
import sys
sys.path.append(os.path.abspath('./src/'))
import argparse
import datetime
from agent_async import SacAgentAsync
from rlutils.utils import *
import socket
from rlutils.linear_expe import make_red_yellow_env_speed, DummyconnectionEnv
import multiprocessing as mp
from rlutils.envs import * #register_envs
from rlutils.env_wrappers import LoggerWrap
from CONFIGS import config_EXPE

tau = 0.05
len_episode = 128
PHI= 40

def run():
    set_high_priority()
    process = mp.Process(target=run_rpi_linear_learning)
    process.start()
    time.sleep(10)
    
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
    env, params = make_red_yellow_env_speed(vid, s, monitor_dir, len_episode=len_episode, tau=tau, discrete_actions=False, phi=PHI, sb3=False)

    #UNCOMMENT FOR DUMMY ENV
    # dummy ENV
    # env = DummyconnectionEnv(vid, s, monitor_dir)
    # env = LoggerWrap(env, path=monitor_dir, pickle_images=False)
    # env = TimeLimit(env, max_episode_steps=len_episode)

    configs=config_EXPE
    configs.update({'log_dir': monitor_dir})
    configs.update({'gradients_step': 2 * len_episode})
    resume_training_path='./logs/152/model'#None#'./logs-fish/138/model'#105
    configs.update({'resume_training_path': resume_training_path})
    save_yaml_dict(configs, os.path.join(monitor_dir, 'configs.yaml'))
    
    try:
        agent = SacAgentAsync(env=env, **configs)
        agent.run()
    except:
        traceback.print_exc()
    finally:
        agent.save_models(prefix='final_')
        agent.save_buffer()
        s.close()
        vid.release()
        cv2.destroyAllWindows()
        
if __name__ == '__main__':
    run()
