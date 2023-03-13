import copy
from collections import deque
from tqdm.notebook import tqdm
import gym
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
#from moviepy.editor import ImageSequenceClip
from rlgtsrc.project2 import utils
import os
import numpy as np

# https://blog.gofynd.com/building-a-deep-q-network-in-pytorch-fa1086aa5435

class DQN_Agent:

    def __init__(self, seed, lr, sync_freq, exp_replay_size, neuralNet):
        torch.manual_seed(seed)
        self.traningModel = copy.deepcopy(neuralNet)
        self.target_net = copy.deepcopy(self.traningModel)
        self.traningModel  # .cuda()
        self.target_net  # .cuda()
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.traningModel.parameters(), lr=lr)

        self.network_sync_freq = sync_freq
        self.network_sync_counter = 0
        self.gamma = torch.tensor(0.95).float()  # .cuda()
        self.experience_replay = deque(maxlen=exp_replay_size)
        return

    def get_action(self, state, action_space_len, epsilon):
        # We do not require gradient at this point, because this function will be used either
        # during experience collection or during inference
        with torch.no_grad():
            # Qp = self.traningModel(torch.from_numpy(state).float().cuda())
            Qp = self.traningModel(torch.from_numpy(state).float())
        Q, A = torch.max(Qp, axis=0)
        A = A if torch.rand(1, ).item() > epsilon else torch.randint(0, action_space_len, (1,))
        return A

    def get_q_next(self, state):
        with torch.no_grad():
            qp = self.target_net(state)
        q, _ = torch.max(qp, axis=1)
        return q

    def collect_experience(self, experience):
        self.experience_replay.append(experience)
        return

    def sample_from_experience(self, sample_size):
        if (len(self.experience_replay) < sample_size):
            sample_size = len(self.experience_replay)
        sample = random.sample(self.experience_replay, sample_size)
        s = torch.tensor([exp[0] for exp in sample]).float()
        a = torch.tensor([exp[1] for exp in sample]).float()
        rn = torch.tensor([exp[2] for exp in sample]).float()
        sn = torch.tensor([exp[3] for exp in sample]).float()
        return s, a, rn, sn

    def train(self, batch_size):
        s, a, rn, sn = self.sample_from_experience(sample_size=batch_size)
        if (self.network_sync_counter == self.network_sync_freq):
            self.target_net.load_state_dict(self.traningModel.state_dict())
            self.network_sync_counter = 0

        # predict expected return of current state using main network
        # qp = self.traningModel(s.cuda())
        qp = self.traningModel(s)
        pred_return, _ = torch.max(qp, axis=1)

        # get target return using target network
        # q_next = self.get_q_next(sn.cuda())

        q_next = self.get_q_next(sn)
        # target_return = rn.cuda() + self.gamma * q_next
        target_return = rn + self.gamma * q_next

        loss = self.loss_fn(pred_return, target_return)
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)

        for param in self.traningModel.parameters():
            param.grad.data.clamp_(-1.0, 1.0)

        self.optimizer.step()

        self.network_sync_counter += 1
        return loss.item()



def main():
    env = gym.make('CartPole-v0')
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    exp_replay_size = 256

    layer_sizes = [input_dim, 64, output_dim]

    def build_nn(layer_sizes):
        assert len(layer_sizes) > 1
        layers = []
        for index in range(len(layer_sizes) - 1):
            linear = nn.Linear(layer_sizes[index], layer_sizes[index + 1])
            act = nn.Tanh() if index < len(layer_sizes) - 2 else nn.Identity()
            layers += (linear, act)
        return nn.Sequential(*layers)

    neuralNet = build_nn(layer_sizes)

    agent = DQN_Agent(seed=1423, lr=1e-3, sync_freq=5, exp_replay_size=exp_replay_size, neuralNet=neuralNet)

    # initiliaze experiance replay
    index = 0
    for i in range(exp_replay_size):
        obs = env.reset()
        done = False
        while (done != True):
            A = agent.get_action(obs, env.action_space.n, epsilon=1)
            obs_next, reward, done, _ = env.step(A.item())
            agent.collect_experience([obs, A.item(), reward, obs_next])
            obs = obs_next
            index += 1
            if (index > exp_replay_size):
                break

    # Main training loop
    losses_list, reward_list, episode_len_list, epsilon_list = [], [], [], []
    index = 128
    episodes = 10000
    epsilon = 1

    for i in tqdm(range(episodes)):
        obs, done, losses, ep_len, rew = env.reset(), False, 0, 0, 0
        while (done != True):
            ep_len += 1
            A = agent.get_action(obs, env.action_space.n, epsilon)
            obs_next, reward, done, _ = env.step(A.item())
            agent.collect_experience([obs, A.item(), reward, obs_next])

            obs = obs_next
            rew += reward
            index += 1

            if (index > 128):
                index = 0
                for j in range(4):
                    loss = agent.train(batch_size=16)
                    losses += loss
        if epsilon > 0.05:
            epsilon -= (1 / 5000)

        losses_list.append(losses / ep_len), reward_list.append(rew), episode_len_list.append(
            ep_len), epsilon_list.append(epsilon)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    mean_reward = np.mean(reward_list)
    ax.plot(reward_list)
    ax.set_xlabel("Episode", size=16)
    ax.set_ylabel("Total reward per episode", size=16)
    ax.axhline(mean_reward, linestyle='--', color="black", label="Mean reward per episode {:2.0f}".format(mean_reward))
    ax.legend(fontsize=16)
    ax.set_title("Reward across {:2.0f} episodes".format(episodes),
                 size=16)
    return fig, ax



# Another example
# https://github.com/hubbs5/rl_blog/blob/master/q_learning/deep/ddqn.py
# Double DQN for playing OpenAI Gym Environments. For full writeup, visit:
# https://www.datahubbs.com/deep-q-learning-101/

import numpy as np
import sys
import os
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import gym
import torch
from torch import nn
from collections import namedtuple, deque, OrderedDict
from copy import copy, deepcopy
import pandas as pd
import time
import shutil


def main(argv):
    args = parse_arguments()
    if args.gpu is None or args.gpu == False:
        args.gpu = 'cpu'
    else:
        args.gpu = 'cuda'

    # Initialize environment
    env = gym.make(args.env)

    # Initialize DQNetwork
    dqn = QNetwork(env=env,
                   n_hidden_layers=args.hl,
                   n_hidden_nodes=args.hn,
                   learning_rate=args.lr,
                   bias=args.bias,
                   tau=args.tau,
                   device=args.gpu)
    # Initialize DQNAgent
    agent = DQNAgent(env, dqn,
                     memory_size=args.memorySize,
                     burn_in=args.burnIn,
                     reward_threshold=args.threshold,
                     path=args.path)
    print(agent.network)
    print(agent.target_network)
    # Train agent
    start_time = time.time()

    agent.train(epsilon=args.epsStart,
                gamma=args.gamma,
                max_episodes=args.maxEps,
                batch_size=args.batch,
                update_freq=args.updateFreq,
                network_sync_frequency=args.netSyncFreq)
    end_time = time.time()
    # Save results
    if agent.success:
        agent.save_results(args)
        if args.plot:
            agent.plot_rewards()
    else:
        shutil.rmtree(agent.path)

    x = end_time - start_time
    hours, remainder = divmod(x, 3600)
    minutes, seconds = divmod(remainder, 60)
    print("Peak mean reward: {:.2f}".format(
        max(agent.mean_training_rewards)))
    print("Training Time: {:02}:{:02}:{:02}\n".format(
        int(hours), int(minutes), int(seconds)))


class experienceReplayBuffer:

    def __init__(self, memory_size=50000, burn_in=10000):
        self.memory_size = memory_size
        self.burn_in = burn_in
        self.Buffer = namedtuple('Buffer',
                                 field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        samples = np.random.choice(len(self.replay_memory), batch_size,
                                   replace=False)
        # Use asterisk operator to unpack deque
        batch = zip(*[self.replay_memory[i] for i in samples])
        return batch

    def append(self, state, action, reward, done, next_state):
        self.replay_memory.append(
            self.Buffer(state, action, reward, done, next_state))

    def burn_in_capacity(self):
        return len(self.replay_memory) / self.burn_in

    def capacity(self):
        return len(self.replay_memory) / self.memory_size

class DQNAgent:

    def __init__(self, env, network, memory_size=50000,
                 batch_size=32, burn_in=10000, reward_threshold=None,
                 path=None, *args, **kwargs):

        self.env = env
        self.env_name = env.spec.id
        self.network = network
        self.target_network = deepcopy(network)
        self.tau = network.tau
        self.batch_size = batch_size
        self.window = 100
        if reward_threshold is None:
            self.reward_threshold = 195 if 'CartPole' in self.env_name \
                else 300
        else:
            self.reward_threshold = reward_threshold
        self.path = path
        self.timestamp = time.strftime('%Y%m%d_%H%M')
        self.initialize(memory_size, burn_in)

    # Implement DQN training algorithm
    def train(self, epsilon=0.05, gamma=0.99, max_episodes=10000,
              batch_size=32, network_sync_frequency=5000, update_freq=4):
        self.gamma = gamma
        self.epsilon = epsilon
        # Populate replay buffer
        while self.buffer.burn_in_capacity() < 1:
            done = self.take_step(mode='explore')
            if done:
                self.s_0 = self.env.reset()

        ep = 0
        training = True
        while training:
            self.s_0 = self.env.reset()
            self.rewards = 0
            done = False
            while done == False:
                done = self.take_step(mode='train')
                # Update network
                if self.step_count % update_freq == 0:
                    self.update()
                # Sync networks
                if self.step_count % network_sync_frequency == 0:
                    self.target_network.load_state_dict(
                        self.network.state_dict())

                if done:
                    ep += 1
                    self.training_rewards.append(self.rewards)
                    mean_rewards = np.mean(
                        self.training_rewards[-self.window:])
                    self.training_loss.append(np.mean(self.update_loss))
                    self.update_loss = []
                    self.mean_training_rewards.append(mean_rewards)
                    print("\rEpisode {:d} Mean Rewards {:.2f}\t\t".format(
                        ep, mean_rewards), end="")

                    if ep >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        self.success = True
                        print('\nEnvironment solved in {} steps!'.format(
                            self.step_count))
                        break

    def take_step(self, mode='train'):
        if mode == 'explore':
            action = self.env.action_space.sample()
        else:
            s_0 = np.ravel(self.state_buffer)
            action = self.network.get_action(s_0, epsilon=self.epsilon)
            self.step_count += 1
        s_1, r, done, _ = self.env.step(action)
        self.rewards += r
        self.state_buffer.append(self.s_0.copy())
        self.next_state_buffer.append(s_1.copy())
        self.buffer.append(deepcopy(self.state_buffer), action, r, done,
                           deepcopy(self.next_state_buffer))
        self.s_0 = s_1.copy()
        return done

    def calculate_loss(self, batch):
        states, actions, rewards, dones, next_states = [i for i in batch]
        rewards_t = torch.FloatTensor(rewards).to(device=self.network.device).reshape(-1, 1)
        actions_t = torch.LongTensor(np.array(actions)).to(
            device=self.network.device).reshape(-1, 1)
        dones_t = torch.ByteTensor(dones).to(device=self.network.device)

        #################################################################
        # DDQN Update
        next_actions = torch.max(self.network.get_qvals(next_states), dim=-1)[1]
        next_actions_t = torch.LongTensor(next_actions).reshape(-1, 1).to(
            device=self.network.device)
        target_qvals = self.target_network.get_qvals(next_states)
        qvals_next = torch.gather(target_qvals, 1, next_actions_t).detach()
        #################################################################
        qvals_next[dones_t] = 0  # Zero-out terminal states
        expected_qvals = self.gamma * qvals_next + rewards_t
        loss = nn.MSELoss()(qvals, expected_qvals)
        return loss

    def update(self):
        self.network.optimizer.zero_grad()
        batch = self.buffer.sample_batch(batch_size=self.batch_size)
        loss = self.calculate_loss(batch)
        loss.backward()
        self.network.optimizer.step()
        if self.network.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())

    def initialize(self, memory_size, burn_in):
        self.buffer = experienceReplayBuffer(memory_size, burn_in)
        self.training_rewards = []
        self.training_loss = []
        self.update_loss = []
        self.mean_training_rewards = []
        self.rewards = 0
        self.step_count = 0
        self.s_0 = self.env.reset()
        self.state_buffer = deque(maxlen=self.tau)
        self.next_state_buffer = deque(maxlen=self.tau)
        [self.state_buffer.append(np.zeros(self.s_0.size))
         for i in range(self.tau)]
        [self.next_state_buffer.append(np.zeros(self.s_0.size))
         for i in range(self.tau)]
        self.state_buffer.append(self.s_0)
        self.success = False
        if self.path is None:
            self.path = os.path.join(os.getcwd(),
                                     self.env_name, self.timestamp)
        os.makedirs(self.path, exist_ok=True)

    def plot_rewards(self):
        plt.figure(figsize=(12, 8))
        plt.plot(self.training_rewards, label='Rewards')
        plt.plot(self.mean_training_rewards, label='Mean Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.ylim([0, np.round(self.reward_threshold) * 1.05])
        plt.savefig(os.path.join(self.path, 'rewards.png'))
        plt.show()

    def save_results(self, args):
        weights_path = os.path.join(self.path, 'dqn_weights.pt')
        torch.save(self.network.state_dict(), weights_path)
        # Save rewards
        rewards = pd.DataFrame(self.training_rewards, columns=['reward'])
        rewards.insert(0, 'episode', rewards.index.values)
        rewards.to_csv(os.path.join(self.path, 'rewards.txt'))
        # Save model parameters
        file = open(os.path.join(self.path, 'parameters.txt'), 'w')
        file.writelines('rewards')
        [file.writelines('\n' + str(k) + ',' + str(v))
         for k, v in vars(args).items()]
        file.close()


class QNetwork(nn.Module):

    def __init__(self, env, learning_rate=1e-3, n_hidden_layers=4,
                 n_hidden_nodes=256, bias=True, activation_function='relu',
                 tau=1, device='cpu', *args, **kwargs):
        super(QNetwork, self).__init__()
        self.device = device
        self.actions = np.arange(env.action_space.n)
        self.tau = tau
        n_inputs = env.observation_space.shape[0] * tau
        self.n_inputs = n_inputs
        n_outputs = env.action_space.n

        activation_function = activation_function.lower()
        if activation_function == 'relu':
            act_func = nn.ReLU()
        elif activation_function == 'tanh':
            act_func = nn.Tanh()
        elif activation_function == 'elu':
            act_func = nn.ELU()
        elif activation_function == 'sigmoid':
            act_func = nn.Sigmoid()
        elif activation_function == 'selu':
            act_func = nn.SELU()

        # Build a network dependent on the hidden layer and node parameters
        layers = OrderedDict()
        n_layers = 2 * (n_hidden_layers - 1)
        for i in range(n_layers + 1):
            if n_hidden_layers == 0:
                layers[str(i)] = nn.Linear(
                    n_inputs,
                    n_outputs,
                    bias=bias)
            elif i == n_layers:
                layers[str(i)] = nn.Linear(
                    n_hidden_nodes,
                    n_outputs,
                    bias=bias)
            elif i % 2 == 0 and i == 0:
                layers[str(i)] = nn.Linear(
                    n_inputs,
                    n_hidden_nodes,
                    bias=bias)
            elif i % 2 == 0 and i < n_layers - 1:
                layers[str(i)] = nn.Linear(
                    n_hidden_nodes,
                    n_hidden_nodes,
                    bias=bias)
            else:
                layers[str(i)] = act_func

        self.network = nn.Sequential(layers)

        # Set device for GPU's
        if self.device == 'cuda':
            self.network.cuda()

        self.optimizer = torch.optim.Adam(self.parameters(),
                                          lr=learning_rate)

    def get_action(self, state, epsilon=0.05):
        if np.random.random() < epsilon:
            action = np.random.choice(self.actions)
        else:
            action = self.greedy_action(state)
        return action

    def greedy_action(self, state):
        qvals = self.get_qvals(state)
        return torch.max(qvals, dim=-1)[1].item()

    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
        state_t = torch.FloatTensor(state).to(device=self.device)
        return self.network(state_t)