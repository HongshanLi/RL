import gym
import torch
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

learning_rate = 0.01
gamma = 0.99

env = gym.make('CartPole-v1')
env.seed(1); torch.manual_seed(1)


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.l1= nn.Linear(self.state_space, 128, bias=False)
        self.dropout = nn.Dropout(p=0.6)
        self.l2 = nn.Linear(128, self.action_space, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = gamma

        # episode policy and reward history
        self.policy_history = [] 
        self.reward_episode = []
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        x = self.l1(x)
        x = self.dropout(x)
        x = self.l2(x)
        x = self.softmax(x)
        return x

policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

def select_action(state):
    # select an action (0 or 1) by running policy model 
    # and choosing based on the probabilities in state
    state = torch.from_numpy(state).type(torch.FloatTensor)
    state = policy(Variable(state))
    c = Categorical(state)
    action = c.sample()

    # add log probability of our chosen action to our history
    policy.policy_history.append(c.log_prob(action))
    return action

def update_policy(episode):
    R = 0
    rewards = []
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0, R)
    
    if episode % 50 == 0:
        pass

    # scale rewards
    rewards = torch.FloatTensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    # loss
    policy.policy_history = torch.stack(policy.policy_history) 
    loss = torch.mul(policy.policy_history, Variable(rewards)).mul(-1)
    loss = torch.sum(loss, -1)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # save and initialize episode history counters
    policy.loss_history.append(loss.item())
    policy.reward_history.append(np.sum(policy.reward_episode))
    policy.policy_history = [] 
    policy.reward_episode = []
    return


def main(episode):
    running_reward = 10
    for episode in range(episodes):
        state = env.reset()
        done = False
                    
        for time in range(1000):
            action = select_action(state)

            state, reward, done, _ = env.step(action.item())
            # save reward
            policy.reward_episode.append(reward)
            if done:
                break
        # used to determine when the environment is solved
        running_reward = (running_reward * 0.99)  + (time * 0.01)
        update_policy(episode)

        if episode % 50 == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(episode, time, running_reward))

        if running_reward > env.spec.reward_threshold:
            print('Solved! Running reward is now {} and the last episode runs to {} time steps'.format(running_reward, time))
            break
    return

episodes = 1000
main(episodes)
