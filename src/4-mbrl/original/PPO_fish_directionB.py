'''
gae
compare sb3
wrap model-learn
ten vs np
'''
import time
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import gym
import numpy as np
import os
from rlutils.envs import *
from cs285.infrastructure.pytorch_util import *
from cs285.infrastructure.logger import Logger
from rlutils.utils import make_dir_exp
from rlutils.cnn_utils import weight_init
from rlutils.VAE_model import Encoder, std_config_vae
from collections import deque, OrderedDict
from gym.wrappers.time_limit import TimeLimit
from timeit import default_timer as timer
# from sta

torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'using {device}')
LAYER_SIZE=64

class Memory:
    # structure to store data for each update
    def __init__(self, max_size, state_dim, action_dim, mode='torch'):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.mode = mode
        if mode=='torch':
            self.actions = torch.zeros((self.max_size, self.action_dim), dtype=torch.float32).to(device)
            self.states = torch.zeros((self.max_size, self.state_dim), dtype=torch.float32).to(device)
            self.logprobs = torch.zeros((self.max_size,), dtype=torch.float32).to(device)
            self.rewards = np.zeros(shape=(self.max_size,), dtype=np.float32)
            self.is_terminals = np.zeros(shape=(self.max_size,), dtype=np.float32)
        else:
            self.actions = np.zeros(shape=(self.max_size, self.action_dim), dtype=np.float32)
            self.states = np.zeros(shape=(self.max_size, self.state_dim), dtype=np.float32)
            self.logprobs = np.zeros(shape=(self.max_size,), dtype=np.float32)
            self.rewards = np.zeros(shape=(self.max_size,), dtype=np.float32)
            self.is_terminals = np.zeros(shape=(self.max_size, ), dtype=np.float32)
        self.pos=0

    def clear_memory(self):
        if self.mode=='torch':
            self.actions = torch.zeros((self.max_size, self.action_dim), dtype=torch.float32).to(device)
            self.states = torch.zeros((self.max_size, self.state_dim), dtype=torch.float32).to(device)
            self.logprobs = torch.zeros((self.max_size,), dtype=torch.float32).to(device)
            self.rewards = np.zeros(shape=(self.max_size,), dtype=np.float32)
            self.is_terminals = np.zeros(shape=(self.max_size,), dtype=np.float32)
        else:
            self.actions = np.zeros(shape=(self.max_size, self.action_dim), dtype=np.float32)
            self.states = np.zeros(shape=(self.max_size, self.state_dim), dtype=np.float32)
            self.logprobs = np.zeros(shape=(self.max_size,), dtype=np.float32)
            self.rewards = np.zeros(shape=(self.max_size,), dtype=np.float32)
            self.is_terminals = np.zeros(shape=(self.max_size,), dtype=np.float32)
        self.pos = 0


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std,discrete=False):
        super(ActorCritic, self).__init__()
        self.actor = build_mlp(state_dim, action_dim, n_layers=2, size=LAYER_SIZE, activation='tanh', output_activation='tanh').to(device)
        self.critic = build_mlp(state_dim, action_dim, n_layers=2, size=LAYER_SIZE, activation='tanh', output_activation='tanh').to(device)
        # self.critic = build_mlp(state_dim, action_dim, n_layers=2, size=LAYER_SIZE, activation='tanh').to(device)
        self.actor.apply(weight_init)
        self.critic.apply(weight_init)
        self.action_var = torch.full((action_dim,), action_std * action_std).to(device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory=None,add_memory=True):
        # operations per decision time step
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)
        # dist = DiagGaussianDistribution(action_mean, cov_mat)
        action = dist.sample()

        if add_memory:
            action_logprob = dist.log_prob(action)
            memory.states[memory.pos] = state
            memory.actions[memory.pos] = action
            memory.logprobs[memory.pos] = action_logprob
        return action.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state).reshape(-1)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, action_std, lr, gamma, n_epochs, eps_clip, params):
        self.logger = Logger(params['logdir'])
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.n_epochs = n_epochs
        self.policy = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.policy_old = ActorCritic(state_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.vf_loss = params["vf_coef"]
        self.gae_lambda = params["gae_lambda"]
        self.max_grad_norm = params["max_grad_norm"]
        self.normalize_advantage = params["normalize_advantage"]
        self.params = params
        self.itr = 0

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def play_episodes(self, env, n_eps=1):
        obs, acts, rews, dones = [], [], [], []
        self.policy.actor.eval()
        for i in range(n_eps):
            state = env.reset()
            obs.append(state)
            done = False
            while not done:
                action = self.policy.act(torch.FloatTensor(state.reshape(1, -1)).to(device),add_memory=False).cpu().data.numpy().flatten()
                acts.append(action)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                rews.append(reward)

        self.policy.actor.train()
        print(f'eval rew: {np.array(rews).sum()/n_eps}')
        self.logger.log_scalar(np.array(rews).sum()/n_eps, 'eval_rew', self.itr)

    def _get_samples(self, memory, indices):
        '''
        # Create a TensorDataset
        dataset = TensorDataset(features, labels)

        # Create a DataLoader
        # You can adjust the batch size and whether to shuffle the data or not
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        '''
        return self.advantages[indices], self.returns[indices], memory.logprobs[indices], memory.states[indices], memory.actions[indices]

    def get_minibatch(self, memory, batch_size):
        indices = np.random.choice(memory.max_size, batch_size, replace=False)
        start_idx = 0
        while start_idx < self.params["max_timesteps"]:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size
        return memory


    def update(self, memory):
        self.itr += 1

        # convert list to tensor:
        old_states = memory.states
        old_actions = memory.actions

        # Optimize policy for K epochs:
        for _ in range(self.n_epochs):
            #m mini-batch loop
            # Evaluating old actions and values :
            self.logprobs, self.state_values, self.dist_entropy = self.policy.evaluate(old_states, old_actions)
            # Convert to numpy
            last_values = self.policy.critic(memory.states[-1]).detach().cpu().numpy().flatten()

            #calculate gae (advantage)
            last_gae_lam = 0
            state_values_np = self.state_values.detach().cpu().numpy()
            self.advantages = np.zeros_like(memory.rewards)
            for step in reversed(range(len(memory.states))):
                if step == len(memory.states) - 1:
                    next_non_terminal = 1.0 - memory.is_terminals[step]
                    next_values = last_values
                else:
                    next_non_terminal = 1.0 - memory.is_terminals[step]
                    next_values = state_values_np[step + 1]

                delta = memory.rewards[step] + self.gamma * next_values * next_non_terminal - state_values_np[step]
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
                self.advantages[step] = last_gae_lam

            if self.normalize_advantage and len(self.advantages) > 1:
                self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

            self.returns = self.advantages + state_values_np
            self.advantages, self.returns = torch.FloatTensor(self.advantages).to(device),torch.FloatTensor(self.returns).to(device)
            indices = np.random.permutation(memory.max_size)
            start_idx = 0

            while start_idx < self.params["max_timesteps"]:

                advantages_b, returns_b, old_logprobs_b, observations_b, actions_b = self._get_samples(memory, indices[start_idx : start_idx + self.params["batch_size"]])
                state_values_b, logprobs_b, dist_entropy_b = self.policy.evaluate(observations_b, actions_b)
                dist_entropy_b=-torch.mean(dist_entropy_b)
                start_idx += self.params["batch_size"]

                ratios = torch.exp(logprobs_b - old_logprobs_b.detach())
                surr1 = ratios * advantages_b
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_b
                assert surr1.shape == surr2.shape, "surr1.shape != surr2.shape"
                try:
                    assert returns_b.shape == state_values_b.shape, f"returns.shape:{returns_b.shape} != state_values_b.shape:{state_values_b.shape}"
                except:
                    print(f"returns.shape:{returns_b.shape} != state_values_b.shape:{state_values_b.shape}")
                # value_loss = nn.functional.mse_loss(state_values_b, returns_b)
                value_loss = self.MseLoss(state_values_b, returns_b)

                policy_loss = -torch.min(surr1, surr2).mean()
                loss = policy_loss + self.params['vf_coef'] * value_loss - self.params['ent_coef'] * dist_entropy_b

                # take gradient step
                self.optimizer.zero_grad()
                loss.backward()
                # loss.mean().backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

            # Copy new weights into old policy:
            self.policy_old.load_state_dict(self.policy.state_dict())
        # decide what to log
        logs = OrderedDict()
        logs['loss'] = loss.mean().item()
        logs['return'] = sum(memory.rewards)
        logs['entropy'] = self.dist_entropy.mean().item()
        logs['value_Loss'] = value_loss.mean().item()
        logs['policy_loss'] = policy_loss.mean().item()
        logs['value'] = self.state_values.mean().item()
        logs['ratio'] = ratios.mean().item()
        logs['advantage'] = self.advantages.mean().item()
        # perform the logging
        for key, value in logs.items():
            self.logger.log_scalar(value, key, self.itr)
        self.logger.flush()


def main(k):
    path, _ = make_dir_exp('../logs-fish')
    torch.manual_seed(0)
    np.random.seed(0)

    log_interval = 10  # print avg reward in the interval
    params={}
    params["max_episodes"] = 150  # max training episodes
    params["max_timesteps"] = 768  # max timesteps in one episode
    params["batch_size"] = 128  # max timesteps in one episode
    params["vf_coef"] = 0.451 #0.42280131196534576  # max timesteps in one episode
    params['logdir'] = path
    params['max_grad_norm'] = 0.8
    params['ent_coef'] = 0.0876493187514028
    params['normalize_advantage'] = True
    action_std = 0.5  # constant std for action distribution (Multivariate Normal)
    params["n_epochs"] = 40  # update policy for K epochs
    params["eps_clip"] = 0.1  # clip parameter for PPO clip_range
    params["gamma"] = 0.98  # discount factor
    params["gae_lambda"] = 1 #0.95  # discount factor for GAE
    lr = 1.7045770678150675e-05  # parameters for Adam optimizer
    random_seed = None
    #############################################
    # creating environment
    env_name='FishStationary-v0'
    # env_name='FishMoving-v0'
    env = gym.make(env_name)  # fish.FishEvasionEnv(dt = 0.1)

    # get observation and action dimensions from the environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)

    memory = Memory(params["max_timesteps"], env.observation_space.shape[0], env.action_space.shape[0])
    ppo = PPO(state_dim, action_dim, action_std, lr, params["gamma"], params["n_epochs"], params["eps_clip"], params=params)
    # ------------------------------------------------------------------
    # start training from an existing policy
    # ppo.policy_old.load_state_dict(torch.load('./direction_policy/PPO_{}_{:06d}.pth'.format(env_name,4380),map_location=device))
    # ppo.policy.load_state_dict(torch.load('./direction_policy/PPO_{}_{:06d}.pth'.format(env_name,4380),map_location=device))
    # ------------------------------------------------------------------
    # logging variables
    running_reward = 0
    avg_length = 0
    time_step = 0

    # training loop
    for i_episode in range(1, params["max_episodes"] + 1):
        state = env.reset()
        for t in range(params["max_timesteps"]):
            time_step += 1
            action = ppo.select_action(state, memory)
            # ------------------------------------------------------------------
            s=time.time()
            state, reward, done, info = env.step(action)
            if time.time()-s>0.01:
                print(time.time()-s)
                continue

            # Storing reward and is_terminals:
            memory.rewards[memory.pos]=reward
            if info!={}:
                memory.is_terminals[memory.pos] = done * (1-info['TimeLimit.truncated'])
            else:
                memory.is_terminals[memory.pos] = done
            memory.pos += 1
            running_reward += reward

            # break if episode ends
            if done:
                ppo.update(memory)
                memory.clear_memory()
                time_step = 0
                break

        avg_length += t
        ppo.logger.log_scalar(running_reward, 'train/reward', ppo.itr)
        # save every 50 episodes
        if i_episode % 50 == 0:
            torch.save(ppo.policy.state_dict(), path + '/PPO_{}_direction{:06d}.pth'.format(env_name, i_episode))
        # ------------------------------------------------------------------ logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length / log_interval)
            running_reward = ((running_reward / log_interval))
            ppo.play_episodes(n_eps=1, env=env)
            print('Episode {} \t Avg length: {} \t Avg reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0

if __name__ == '__main__':
    main(0)

