'''
process the learning visual-servo-continous ->offline RL
'''
import gym
from rlutils.utils import *  # plot_html_eps_from_inference_experiment
from rlutils.vision import *  # plot_html_eps_from_inference_experiment
import re
from cs285.infrastructure.utils import *
from dev.modelTrain import *
from gym.wrappers import TimeLimit
import datetime
from cs285.agents.mb_agent import MBAgent, EnsembleDynamicsModel
import cv2

env = gym.make('FishMoving-v0')
args_dict_default['ac_dim'] = env.action_space.shape[0]
args_dict_default['ob_dim'] = env.observation_space.shape[0]
args_dict_default['dropout'] = 0.05
args_dict_default['layer_norm'] = True

trainer = EnsembleDynamicsModel(env, args_dict_default)
dataset = trainer.prepare_dat_sim(num_steps=10000)
print(len(dataset))
trainer.learn()

model_list=[[64,2]]
args, params = parse_args()

data_list = []
# TRAIN DIFFERENT MODELS and compare
model_list = [[256, 2]]
for i, mparams in enumerate(model_list):
    params['size'] = mparams[0]
    params['n_layers'] = mparams[1]
    # PARAM sweep
    params['exp_name'] = str(datadir)
    params['save_params'] = True
    params['dropout'] = 0.05
    params['layer_norm'] = True
    create_log_dir(args, params)

    #############CHOSE ENV with same characteristics as data
    # params['env_name'] = 'FishMoving-v0'
    ###################
    ### RUN TRAINING
    ###################
    trainer = MB_Trainer(params)
    trainer.fit_model(dataset)
    ##v2
    trainer=EnsembleDynamicsModel(params)
    trainer.learn(dataset)
    data_list.append([mparams[0], mparams[1], trainer.ini_score])

pd.DataFrame(data_list, columns=['nodes', 'n_layers', 'score']).to_csv(f'./model_fit{datetime.now().strftime("%Y%m%d-%H%M%S")}.csv', index=False)