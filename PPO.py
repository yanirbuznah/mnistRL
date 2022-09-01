import time
from collections import deque
from typing import Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.distributions.categorical import Categorical

from Agent import Agent, ReplayMemory
from main import epsilon_annealing, get_env_dim
from mnist_env import Mnist_env
from ppo_parses import FLAGS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Transition:
    def __init__(self, current_state, action, reward, probs, vals, done):
        self.state = current_state
        self.action = action
        self.reward = reward
        self.probs = probs
        self.vals = vals
        self.done = done

    def get_all(self):
        return self.state, self.action, self.reward, self.probs, self.vals, self.done

    def __repr__(self):
        return f"""
    ====== Transition ======
    > state:\t{self.state}
    > action:\t{self.action}
    > reward:\t{self.reward}
    > probs:\t{self.probs}
    > vals:\t{self.vals}
    > done:\t{self.done} 
    """


class PPOMemory(ReplayMemory):
    def __init__(self, batch_size):
        super().__init__(batch_size)
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size

    def get_batches(self):
        num_of_states = len(self.states)

        # arange indices in batces
        batches_starts = np.arange(0, num_of_states, self.batch_size)

        indices = np.arange(num_of_states)
        np.random.shuffle(indices)
        batches = [indices[batch_start:batch_start + self.batch_size] for batch_start in batches_starts]

        return np.array(self.states), np.array(self.actions), np.array(self.probs), \
               np.array(self.vals), np.array(self.rewards), np.array(self.dones), batches

    # TODO: make sure it's fine
    def push(self, transition: Transition):
        self.states.append(transition.state)
        self.probs.append(transition.probs)
        self.vals.append(transition.vals)
        self.actions.append(transition.action)
        self.rewards.append(transition.reward)
        self.dones.append(transition.done)

    def reset(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []


class ActorModel(nn.Module):
    def __init__(self, input_shape, actions_space, lr):
        super(ActorModel, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, actions_space),
            # maybe remove this softmax
            nn.Softmax(dim=-1)
        )
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        return dist

    def save_model(self, file='actor_model'):
        torch.save(self.state_dict(), file)

    def load_model(self, file='actor_model'):
        self.load_state_dict(torch.load(file))


class CriticModel(nn.Module):
    def __init__(self, input_shape, actions_space, lr):
        super(CriticModel, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=0.001)

    def forward(self, state):
        return self.critic(state)

    def save_model(self, file='critic_model'):
        torch.save(self.state_dict(), file)

    def load_model(self, file='critic_model'):
        self.load_state_dict(torch.load(file))


class PPOAgent:
    def __init__(self, input_shape, actions_space, args, env):
        self.batch_size = args.batch_size

        self.env = env
        self.gamma = args.gamma
        self.policy_clip = args.policy_clip
        self.epochs = args.epochs
        self.gae_lambda = args.gae_lambda
        self.actions_space = actions_space
        self.actor = ActorModel(input_shape=input_shape, actions_space=self.actions_space, lr=args.lr)
        self.critic = CriticModel(input_shape=input_shape, actions_space=self.actions_space, lr=args.lr)
        self.action_std = args.std

        self.action_var = torch.full((8,), args.std * args.std).to(device)


        self.memory = PPOMemory(args.batch_size)

    def remember(self, transition):
        self.memory.push(transition)

    def save_models(self):
        print("... saving models ...")
        self.actor.save_model()
        self.critic.save_model()
        torch.save(self.action_var, 'var')

    def load_models(self):
        print("... loading models ...")
        self.actor.load_model()
        self.critic.load_model()
        torch.load(self.action_var, 'var')
        self.action_action_var = torch.load('var')
        # print(self.action_var)

    def set_action_std(self):
        new_action_std = max(0.1, self.action_std * 0.99)
        self.action_var = torch.full((self.actions_space,), new_action_std * new_action_std).to(device)

    def get_action(self, obs: np.ndarray):
        state = torch.tensor(obs, dtype=torch.float).to(device)
        #state = torch.tensor([obs], dtype=torch.float).to(device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.epochs):
            states_, actions_, old_probs_, vals, rewards, dones, batches = self.memory.get_batches()
            advantages = np.zeros(len(rewards), dtype=float)
            advantages = torch.tensor(advantages).to(device)
            for t in range(len(rewards) - 1):
                discount = 1
                advantage_t = 0
                for k in range(t, len(rewards) - 1):
                    advantage_t += discount * (rewards[k] + self.gamma * vals[k + 1] * (1 - int(dones[k])) - vals[k])
                    discount *= self.gamma * self.gae_lambda
                advantages[t] = advantage_t
            advantages = torch.tensor(advantages).to(device)

            vals = torch.tensor(vals).to(device)
            for batch in batches:
                states = torch.tensor(states_[batch], dtype=torch.float).to(device)
                # print(f"batch:{batch} {old_probs}")
                old_probs = torch.tensor(old_probs_[batch]).to(device)
                actions = torch.tensor(actions_[batch]).to(device)

                dist = self.actor(states)
                critic_value = self.critic(states)
                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                weighted_probs = advantages[batch] * prob_ratio
                weights_clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * \
                                        advantages[batch]

                actor_loss = -torch.min(weighted_probs, weights_clipped_probs).mean()

                returns = advantages[batch] + vals[batch]
                critic_loss = ((returns - critic_value) ** 2).mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.reset()


