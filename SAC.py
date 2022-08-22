import random
from typing import List

import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
# set device
from torch.distributions import Normal

from Agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Transition:
  def __init__(self, current_state: np.ndarray, action: int, reward:int, next_state:np.ndarray, done:bool):
    self.current_state = current_state
    self.action = action
    self.reward = reward
    self.next_state = next_state
    self.done = done

  def unpack(self):
    return tuple([self.current_state, self.action, self.reward, self.next_state, self.done])

  def __repr__(self):
    return f"""
    ====== Transition ======
    > state:\t{self.current_state}
    > action:\t{self.action}
    > reward:\t{self.reward}
    > next_state:\t{self.next_state}
    > done:\t{self.done} 
    """



class Memory:
  def __init__(self, input_shape,action_shape,max_len=1e6):
    self.capacity = int(max_len)
    self.memory = []
    self.index = 0


  def push(self, transition: Transition):
    # if len(self.memory) < self.capacity:
    #     self.memory.append(None)
    # self.memory[self.index] = transition.unpack()
    # self.index = (self.index + 1) % self.capacity
    self.memory.append(transition.unpack())
    self.memory = self.memory[-self.capacity:]
    # index = self.counter % self.max_len
    # self.states[index] = transition.current_state
    # self.next_states[index] = transition.next_state
    # self.actions[index] = transition.action
    # self.rewards[index] = transition.reward
    # self.dones[index] = transition.done
    # self.counter += 1

  def sample(self, batch_size):

    batch = random.sample(self.memory,batch_size)
    state, action, reward, next_state, done = map(np.stack, zip(*batch))
    return state, action, reward, next_state, done


  def __len__(self):
    return len(self.memory)

  def reset(self):
    self.memory = []


# GaussianPolicy
class ActorModel(nn.Module):
    def __init__(self, input_shape, actions_space, lr, max_action, name='actor', noise=1e-6, min_clamp=-20,
                 max_clamp=2):
        super(ActorModel, self).__init__()
        self.l1 = nn.Linear(input_shape, 256)
        self.l2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, actions_space)
        self.sigma = nn.Linear(256, actions_space)
        self.to(device)
        self.action_scale = torch.ones(actions_space).to(device)
        self.bias = torch.zeros(actions_space).to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.apply(weights_init)
        self.noise = noise
        self.name = name
        self.max_action = max_action
        self.min_clamp = min_clamp
        self.max_clamp = max_clamp

    def forward(self, state):
        probs = F.relu(self.l2(F.relu(self.l1(state))))
        mu = self.mu(probs)
        sigma = torch.clamp(self.sigma(probs), min=self.min_clamp, max=self.max_clamp)
        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.forward(state)
        dist = Normal(mu, sigma.exp())
        actions = dist.rsample() if reparameterize else dist.sample()
        y_t = torch.tanh(actions)
        action = y_t * self.action_scale + self.bias
        log_probs = dist.log_prob(actions)
        log_probs -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs, mu

    def save_model(self, ):
        torch.save(self.state_dict(), self.name)

    def load_model(self):
        self.load_state_dict(torch.load(self.name))


# QNetwork
class CriticModel(nn.Module):
    def __init__(self, input_shape, actions_space, beta, name):
        super(CriticModel, self).__init__()
        self.model1 = nn.Sequential(
            nn.Linear(input_shape + actions_space, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )
        self.model2 = nn.Sequential(
            nn.Linear(input_shape + actions_space, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )
        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.name = name

    def forward(self, state, action):
        return self.model1(torch.cat([state, action], dim=1)), self.model2(torch.cat([state, action], dim=1))

    def save_model(self, ):
        torch.save(self.state_dict(), self.name)

    def load_model(self):
        self.load_state_dict(torch.load(self.name))


class QModel(nn.Module):
    def __init__(self, input_shape, actions_space, beta, name):
        super(QModel, self).__init__()
        self.model11 = nn.Linear(input_shape + actions_space, 256)
        self.model12 = nn.Linear(256, 256)
        self.model13 = nn.Linear(256, 1)

        self.model21 = nn.Linear(input_shape + actions_space, 256)
        self.model22 = nn.Linear(256, 256)
        self.model23 = nn.Linear(256, 1)

        self.to(device)
        self.optimizer = optim.Adam(self.parameters(), lr=beta) if name == 'critic' else None
        self.name = name

    def forward(self, state, action):
        x1 = F.relu(self.model11(torch.cat([state, action], dim=1)))
        x1 = F.relu(self.model12(x1))
        x1 = self.model13(x1)

        x2 = F.relu(self.model21(torch.cat([state, action], dim=1)))
        x2 = F.relu(self.model22(x2))
        x2 = self.model23(x2)

        return x1, x2

    def save_model(self, ):
        torch.save(self.state_dict(), self.name)

    def load_model(self):
        self.load_state_dict(torch.load(self.name))


# Initialize Policy weights
def weights_init(x):
    if isinstance(x, nn.Linear):
        torch.nn.init.xavier_uniform_(x.weight, gain=1)
        torch.nn.init.constant_(x.bias, 0)




class SacAgent(Agent):
    def __init__(self, args, input_space, actions_space, env=None):
        super().__init__(name="SacAgent")
        self.input_dim = input_space.shape
        self.action_dim = actions_space.shape
        self.memory = Memory(self.input_dim[0], self.action_dim[0], max_len=args.buffer_size)

        self.batch_size = args.batch_size
        self.scale = args.reward_scale
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        self.initial_models(args)

    def initial_models(self, args):

        self.critic = CriticModel(self.input_dim[0], self.action_dim[0], args.beta, name='critic')
        self.critic_target = CriticModel(self.input_dim[0], self.action_dim[0], args.beta, name='critic_target')
        self.critic_target.load_state_dict(self.critic.state_dict())

        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.action_dim).to(device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=args.beta)

        self.actor = ActorModel(self.input_dim[0], self.action_dim[0], args.beta, max_action=1.)


    def choose_action(self, obs, eval=False):
        state = torch.Tensor(np.array([obs])).to(device)
        action, log_probs, mu = self.actor.sample_normal(state)
        if eval:
            return mu.cpu().detach().numpy()[0]
        return action.cpu().detach().numpy()[0]

    def remember(self, transition):
        self.memory.push(transition)


    def update_target(self ):
        for (value, target_value) in zip(self.critic.parameters(), self.critic_target.parameters()):
            # print(target_value.shape)
            target_value.data.copy_(target_value * (1 - self.tau) + value * self.tau)

    def save_models(self):
        self.actor.save_model()
        self.critic.save_model()
        self.critic_target.save_model()

    def train(self):
        if len(self.memory) < 1e4:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.tensor(states, dtype=torch.float).to(device)
        actions = torch.tensor(actions, dtype=torch.float).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device).unsqueeze(-1)
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)
        dones = torch.tensor(dones).to(device).unsqueeze(-1)

        with torch.no_grad():
            next_state_action, next_state_log_probs, next_state_mu = self.actor.sample_normal(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_state_action)
            minq_next_target = torch.min(q1_target, q2_target) - self.alpha * next_state_log_probs
            next_q_value = rewards + ((1 - dones) * self.gamma * (minq_next_target))

        q1_policy, q2_policy = self.critic(states, actions)
        q1_loss = F.mse_loss(q1_policy, next_q_value)
        q2_loss = F.mse_loss(q2_policy, next_q_value)
        critic_loss = q1_loss + q2_loss

        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        actions_from_policy, log_probs, _ = self.actor.sample_normal(states)
        q1_probs, q2_probs = self.critic(states, actions_from_policy)
        minq_probs = torch.min(q1_probs, q2_probs)
        actor_loss = ((self.alpha * log_probs) - minq_probs).mean()

        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

        else:
            alpha_loss = torch.tensor(0.).to(device)


        self.update_target()

#
# def validate(agent, env_, k=100):
#     score_history = np.full(100, 315)
#     for i in range(k):
#         env = wrap_env(env_)
#         obs = env.reset()
#         done = False
#         score = 0
#
#         while not done:
#             action = agent.choose_action(obs, eval=True)
#             obs, reward, done, info = env.step(action)
#             reward = max(-1, reward)
#             score += reward
#             reward *= args.reward_scale
#             obs = obs_
#         score_history[i] = score
#         if np.mean(score_history) < 300:  # for speed up
#             return np.mean(score_history[:i + 1])
#
#     return np.mean(score_history)
#
#
