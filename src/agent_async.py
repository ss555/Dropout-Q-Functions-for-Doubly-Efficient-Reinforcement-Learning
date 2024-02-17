import os

import dill
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from rltorch.memory import MultiStepMemory, PrioritizedMemory
from model import TwinnedQNetwork, GaussianPolicy
from src.utils import grad_false, hard_update, soft_update, to_batch, update_params, RunningMeanStats
from collections import deque
import itertools
import traceback
import math
import pickle as pkl
import threading
import time
import json
from cs285.infrastructure.utils import *

class SacAgentAsync:
    def __init__(self, env, log_dir, num_steps=3000000, batch_size=256,
                 lr=0.0003, hidden_units=[256, 256], memory_size=1e6,
                 gamma=0.99, tau=0.005, entropy_tuning=True, ent_coef=0.2,
                 multi_step=1, per=False, alpha=0.6, beta=0.4,
                 beta_annealing=0.0001, grad_clip=None, critic_updates_per_step=1,
                 start_steps=10000, log_interval=10, target_update_interval=1,
                 scheduler=None, cuda=0, seed=0,
                 eval_runs=1, huber=0, layer_norm=0,
                 method=None, target_entropy=None, target_drop_rate=0.0, critic_update_delay=1, save_model_interval=1,
                 gradients_step=1, eval_episodes_interval=20,resume_training_path=None):

        self.env = env
        self.episodes = 0
        self.gradients_step= gradients_step
        self.save_model_interval = save_model_interval

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        torch.backends.cudnn.deterministic = True  # It harms a performance.
        torch.backends.cudnn.benchmark = False

        self.method = method
        self.critic_update_delay = critic_update_delay
        self.target_drop_rate = target_drop_rate
        self.device = torch.device("cuda:" + str(cuda) if torch.cuda.is_available() else "cpu")

        # policy
        self.policy = GaussianPolicy(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            hidden_units=hidden_units).to(self.device)

        # Q functions
        kwargs_q = {"num_inputs": self.env.observation_space.shape[0],
                    "num_actions": self.env.action_space.shape[0],
                    "hidden_units": hidden_units,
                    "layer_norm": layer_norm,
                    "drop_rate": self.target_drop_rate}
        self.critic = TwinnedQNetwork(**kwargs_q).to(self.device)
        self.critic_target = TwinnedQNetwork(**kwargs_q).to(self.device)
        if self.target_drop_rate <= 0.0:
            self.critic_target = self.critic_target.eval()
        # copy parameters of the learning network to the target network
        hard_update(self.critic_target, self.critic)
        # disable gradient calculations of the target network
        grad_false(self.critic_target)

        # optimizer
        self.policy_optim = Adam(self.policy.parameters(), lr=lr)
        if self.method == "redq":
            for i in range(self.critic.N):
                setattr(self, "q"+str(i)+"_optim",
                        Adam(getattr(self.critic, "Q"+str(i)).parameters(), lr=lr))
        else:
            self.q1_optim = Adam(self.critic.Q1.parameters(), lr=lr)
            self.q2_optim = Adam(self.critic.Q2.parameters(), lr=lr)

        if entropy_tuning:
            if not (target_entropy is None):
                self.target_entropy = torch.prod(torch.Tensor([target_entropy]).to(self.device)).item()
            else:
                # Target entropy is -|A|.
                self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            # We optimize log(alpha), instead of alpha.
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.scheduler=scheduler
            if self.scheduler == "linear":
                # cosine annealing
                self.alpha_optim = Adam([self.log_alpha], lr=lr)
                self.alpha_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.alpha_optim, num_steps, eta_min=0, last_epoch=-1)
            else:
                self.alpha_optim = Adam([self.log_alpha], lr=lr)

        else:
            # fixed alpha
            self.alpha = torch.tensor(ent_coef).to(self.device)
        if per:
            # replay memory with prioritied experience replay
            # See https://github.com/ku2482/rltorch/blob/master/rltorch/memory
            self.memory = PrioritizedMemory(
                memory_size, self.env.observation_space.shape,
                self.env.action_space.shape, self.device, gamma, multi_step,
                alpha=alpha, beta=beta, beta_annealing=beta_annealing)
        else:
            # replay memory without prioritied experience replay
            # See https://github.com/ku2482/rltorch/blob/master/rltorch/memory
            self.memory = MultiStepMemory(
                memory_size, self.env.observation_space.shape,
                self.env.action_space.shape, self.device, gamma, multi_step)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_rewards = RunningMeanStats(log_interval)
        self.steps = 0
        self.learning_steps = 0
        self.episodes_num = 0
        self.eval_episodes_interval = eval_episodes_interval#20
        self.num_steps = num_steps
        self.tau = tau
        self.per = per
        self.batch_size = batch_size
        self.start_steps = start_steps
        self.gamma_n = gamma ** multi_step
        self.entropy_tuning = entropy_tuning
        self.grad_clip = grad_clip
        self.critic_updates_per_step = critic_updates_per_step
        self.log_interval = log_interval
        self.target_update_interval = target_update_interval
        self.eval_runs = eval_runs
        self.huber = huber
        self.multi_step = multi_step
        self.start_time = time.time()

        if resume_training_path is not None:
            self.load_models(resume_training_path)
    def save_buffer(self):
        print('saving replay buffer')
        # Use numpy.savez_compressed to save the data
        # np.savez_compressed(os.path.join(self.model_dir, 'buffer.npz'), self.memory)
        with open(os.path.join(self.model_dir, 'buffer.pkl'), "wb") as file_handler:
            pkl.dump(self.memory, file_handler, protocol=pkl.HIGHEST_PROTOCOL)

    def load_buffer(self, resume_path):
        with open('buffer.pkl', 'wb') as file_handler:
            pkl.dump(self.memory, file_handler, protocol=pkl.HIGHEST_PROTOCOL)

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                print(f'Finishing at {self.steps} steps> {self.num_steps}, episode {self.episodes}')
                break

    def is_update(self):
        return len(self.memory) > self.batch_size and self.steps >= self.start_steps

    def act(self, state):
        if self.start_steps > self.steps:
            action = self.env.action_space.sample()
        else:
            action = self.explore(state)
        return action

    def explore(self, state):
        # act with randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def exploit(self, state):
        # act without randomness
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, _, action = self.policy.sample(state)
        return action.cpu().numpy().reshape(-1)

    def calc_current_q(self, states, actions, rewards, next_states, dones):
        curr_q1, curr_q2 = self.critic(states, actions)
        return curr_q1, curr_q2

    def calc_target_q(self, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_entropies, _ = self.policy.sample(next_states)
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            if self.method == "sac" or self.method == "redq":
                next_q = torch.min(next_q1, next_q2) + self.alpha * next_entropies
            elif self.method == "duvn":
                next_q = next_q1 + self.alpha * next_entropies # discard q2
            elif self.method == "monosac":
                next_q2, _ = self.critic_target(next_states, next_actions)
                next_q = torch.min(next_q1, next_q2) + self.alpha * next_entropies
            else:
                raise NotImplementedError()
        # rescale rewards by num step TH20210705
        target_q = (rewards / (self.multi_step * 1.0)) + (1.0 - dones) * self.gamma_n * next_q
        return target_q

    def interact(self):
        self.episode_reward = 0.
        self.episode_steps = 0
        done = False
        state = self.env.reset()
        while not done:
            action = self.act(state)
            next_state, reward, done, info = self.env.step(action)
            self.steps += 1
            self.episode_steps += 1
            self.episode_reward += reward

            # ignore done if the agent reach time horizons
            # (set done=True only when the agent fails)
            if hasattr(self.env,'_max_episode_steps') and self.episode_steps >= self.env._max_episode_steps:
                masked_done = False
            elif hasattr(self.env,'N_STEPS') and self.episode_steps >= self.env.N_STEPS:
                masked_done = False
            elif ("TimeLimit.truncated" in info.keys() and info["TimeLimit.truncated"]): # TODO check this
                masked_done = False
            else:
                masked_done = done

            if self.per:
                batch = to_batch(
                    state, action, reward, next_state, masked_done,
                    self.device)
                with torch.no_grad():
                    curr_q1, curr_q2 = self.calc_current_q(*batch)
                target_q = self.calc_target_q(*batch)
                error = (0.5 * torch.abs(curr_q1 - target_q) + 0.5 * torch.abs(curr_q2 - target_q)).item()
                # signal to calculate multi-step rewards.
                self.memory.append(state, action, reward, next_state, masked_done, error, episode_done=done)
            else:
                # signal to calculate multi-step rewards.
                self.memory.append(state, action, reward, next_state, masked_done, episode_done=done)
            state = next_state

    def train_episode(self):
        self.episodes += 1
        
        x = threading.Thread(target=self.interact)
        x.start()

        if self.is_update():
            self.learn()
        x.join()
        if self.save_model_interval!=-1 and self.episodes%self.save_model_interval==0:
            self.save_models()
        self.episodes_num += 1
        if self.episodes_num % self.eval_episodes_interval == 0:
            self.evaluate()

        # We log running mean of training rewards.
        self.train_rewards.append(self.episode_reward)
        self.writer.add_scalar('reward/train', self.train_rewards.get(), self.steps)

        print(f'episode: {self.episodes}  '
              f'episode steps: {self.episode_steps}  '
              f'reward: {self.episode_reward}')
        
        # if self.episodes%35==0:
        #     self.reset_sac_last_layer()
        
    def reset_sac_last_layer(self):
        self.policy.reset_last_layer()
        self.critic.reset_last_layer()
        self.critic_target.reset_last_layer()
        # hard_update(self.critic_target, self.critic)

    def learn(self):
        self.learning_steps += 1
        for _ in range(self.gradients_step):
            # critic update
            if (self.learning_steps - 1) % self.critic_update_delay == 0:
                for _ in range(self.critic_updates_per_step):
                    if self.per:
                        # batch with indices and priority weights
                        batch, indices, weights = self.memory.sample(self.batch_size)
                    else:
                        batch = self.memory.sample(self.batch_size)
                        # set priority weights to 1 when we don't use PER.
                        weights = 1.

                    q1_loss, q2_loss, errors, mean_q1, mean_q2 = self.calc_critic_loss(batch, weights)

                    update_params(self.q1_optim, self.critic.Q1, q1_loss, self.grad_clip)
                    update_params(self.q2_optim, self.critic.Q2, q2_loss, self.grad_clip)

                    if self.learning_steps % self.target_update_interval == 0:
                        soft_update(self.critic_target, self.critic, self.tau)

                    if self.per:
                        # update priority weights
                        self.memory.update_priority(indices, errors.cpu().numpy())
            # policy and alpha update
            if self.per:
                batch, indices, weights = self.memory.sample(self.batch_size)
            else:
                batch = self.memory.sample(self.batch_size)
                weights = 1.

            policy_loss, entropies = self.calc_policy_loss(batch, weights) # added by tH 20210705
            update_params(self.policy_optim, self.policy, policy_loss, self.grad_clip)

            if self.entropy_tuning:
                entropy_loss = self.calc_entropy_loss(entropies, weights)
                update_params(self.alpha_optim, None, entropy_loss)
                self.alpha = self.log_alpha.exp()

        self.writer.add_scalar('mean-q/q1', mean_q1, self.steps)
        self.writer.add_scalar('mean-q/q2', mean_q2, self.steps)

        self.writer.add_scalar('loss/critic_1', q1_loss, self.steps)
        self.writer.add_scalar('loss/critic_2', q2_loss, self.steps)

        self.writer.add_scalar('policy_loss/train', policy_loss.item(), self.steps)
        self.writer.add_scalar('entropy/train', entropies.mean().item(), self.steps)
        self.writer.add_scalar('time', time.time() - self.start_time, self.steps)

    def calc_critic_loss(self, batch, weights):
        assert self.method == "sac" or self.method == "duvn" or self.method == "monosac", "This method is only for sac or duvn or monosac method"

        curr_q1, curr_q2 = self.calc_current_q(*batch)
        target_q = self.calc_target_q(*batch)

        # TD errors for updating priority weights
        errors = torch.abs(curr_q1.detach() - target_q)
        # We log means of Q to monitor training.
        mean_q1 = curr_q1.detach().mean().item()
        mean_q2 = curr_q2.detach().mean().item()

        q1_loss = torch.mean((curr_q1 - target_q).pow(2) * weights)
        q2_loss = torch.mean((curr_q2 - target_q).pow(2) * weights)
        return q1_loss, q2_loss, errors, mean_q1, mean_q2

    def calc_policy_loss(self, batch, weights):
        states, actions, rewards, next_states, dones = batch
        # We re-sample actions to calculate expectations of Q.
        sampled_action, entropy, _ = self.policy.sample(states)
        # expectations of Q with clipped double Q technique
        q1, q2 = self.critic(states, sampled_action)
        if (self.method == "duvn") or (self.method == "monosac"):
            q2 = q1 # discard q2
        if self.target_drop_rate > 0.0:
            q = 0.5 * (q1 + q2)
        else:
            q = torch.min(q1, q2)
        # Policy objective is maximization of (Q + alpha * entropy) with priority weights.
        policy_loss = torch.mean((- q - self.alpha * entropy) * weights)
        return policy_loss, entropy

    def calc_entropy_loss(self, entropy, weights):
        entropy_loss = -torch.mean(self.log_alpha * (self.target_entropy - entropy).detach() * weights)
        return entropy_loss

    def evaluate(self):
        episodes = self.eval_runs
        returns = np.zeros((episodes,), dtype=np.float32)

        # for return bias estimation TH
        sar_buf = [[] for _ in range(episodes) ] # episodes x (satte, action , reward)

        for i in range(episodes):
            state = self.env.reset()
            episode_reward = 0.
            done = False
            while not done:
                action = self.exploit(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
                # MCE store all (state, action, reward) TH 20210723
                sar_buf[i].append([state, action, reward])
            returns[i] = episode_reward

        mean_return = np.mean(returns)

        # calculate mean / std return bias. TH 20210801
        # - calculate MCE future discounted return (in backward)
        mc_discounted_return = [deque() for _ in range(episodes) ]
        for i in range(episodes):
            for re_tran in reversed(sar_buf[i]):
                if len(mc_discounted_return[i]) > 0:
                    mcret = re_tran[2] + self.gamma_n * mc_discounted_return[i][0]
                else:
                    mcret = re_tran[2]
                mc_discounted_return[i].appendleft(mcret)
        # - calculate normalized MCE return by averaging all MCE returns
        norm_coef = np.mean(list(itertools.chain.from_iterable(mc_discounted_return)))
        norm_coef = math.fabs(norm_coef) + 0.000001
        # - estimate return for all state action, and normalized score
        norm_scores = [[] for _ in range(episodes)]
        for i in range(episodes):
            # calculate normalized score
            states = np.array(sar_buf[i], dtype="object")[:, 0].tolist()
            actions = np.array(sar_buf[i], dtype="object")[:, 1].tolist()
            with torch.no_grad():
                state = torch.FloatTensor(states).to(self.device)
                action = torch.FloatTensor(actions).to(self.device)
                if self.method == "redq":
                    q = self.critic.averageQ(state, action)
                else:
                    q1, q2 = self.critic(state, action)
                    q = 0.5 * (q1 + q2)
                qs = q.to('cpu').numpy()
            for j in range(len(sar_buf[i])):
                score = (qs[j][0] - mc_discounted_return[i][j]) / norm_coef
                norm_scores[i].append(score)
        # calculate std
        flatten_norm_score = list(itertools.chain.from_iterable(norm_scores))
        mean_norm_score = np.mean(flatten_norm_score)
        std_norm_score = np.std(flatten_norm_score)
        print("mean norm score " + str(mean_norm_score))
        print("std norm score " + str(std_norm_score))

        self.writer.add_scalar(
            'reward/test', mean_return, self.steps)
        print('-' * 60)
        print(f'Num steps: {self.steps:<5}  '
              f'reward: {mean_return:<5.1f}')
        print('-' * 60)
        #
        with open(self.log_dir + "/" + "reward.csv", "a") as f:
            f.write(str(self.steps) + "," + str(mean_return) + ",\n")
            f.flush()

        with open(self.log_dir + "/" + "avrbias.csv", "a") as f:
            f.write(str(self.steps) + "," + str(mean_norm_score) + ",\n")
            f.flush()

        with open(self.log_dir + "/" + "stdbias.csv", "a") as f:
            f.write(str(self.steps) + "," + str(std_norm_score) + ",\n")
            f.flush()


    def save_models(self,prefix=''):
        print(f'saving models at {self.model_dir}')
        import dill
        data = self.__dict__.copy()
        exclude_keys = ['env', 'writer', 'summary_dir']
        for k in data.keys():
            if not dill.pickles(data[k]):
                exclude_keys.append(k)

        for k in exclude_keys:
            if k in data.keys():
                del data[k]
        # # Saving the dictionary to a file
        with open(os.path.join(self.model_dir, 'data.pkl'), "wb") as file_handler:
            pkl.dump(data, file_handler, protocol=pkl.HIGHEST_PROTOCOL)

        # Create a dictionary to hold all the model states and optimizer states
        # state_dicts = {
        #     'policy_state_dict': self.policy.state_dict(),
        #     'critic_state_dict': self.critic.state_dict(),
        #     'critic_target_state_dict': self.critic_target.state_dict(),
        #     'alpha_optim_state_dict': self.alpha_optim.state_dict(),
        #     'alpha_state_dict': self.alpha.to('cpu').detach().numpy(),
        #     'q1_optim_state_dict': self.q1_optim.state_dict(),
        #     'q2_optim_state_dict': self.q2_optim.state_dict(),
        # }
        #
        # # Save the combined state dictionaries to a single file
        # torch.save(state_dicts, os.path.join(self.model_dir, f'{prefix}all_models.pth'))
        self.save_buffer()

    def load_models(self, path):
        print(f'loading models from {path}')
        exclude=['steps','episodes','num_steps','env','writer','summary_dir']
        with open(os.path.join(path, 'data.pkl'), "rb") as file_handler:
            data = pkl.load(file_handler)
            for k in exclude:
                if k in data:
                    del data[k]
            self.__dict__.update(data)
        self.load_buffer(path)
        # Load the saved state dictionaries
        # state_dicts = torch.load(os.path.join(resume_training_path, 'all_models.pth'))
        # # Restore the states for each model component and optimizer
        # self.policy.load_state_dict(state_dicts['policy_state_dict'])
        # self.critic.load_state_dict(state_dicts['critic_state_dict'])
        # self.critic_target.load_state_dict(state_dicts['critic_target_state_dict'])
        # self.alpha_optim.load_state_dict(state_dicts['alpha_optim_state_dict'])
        # self.q1_optim.load_state_dict(state_dicts['q1_optim_state_dict'])
        # self.q2_optim.load_state_dict(state_dicts['q2_optim_state_dict'])
        #
        # # For the `alpha` tensor, you would directly replace it since it's a tensor, not a state dict
        # self.alpha = torch.tensor(state_dicts['alpha_state_dict'], dtype=torch.float32)
        # self.alpha = self.alpha.to(self.device)
    def __del__(self):
        self.writer.close()
        self.env.close()