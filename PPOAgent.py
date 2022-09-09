import os
import time
from collections import deque
from itertools import count
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torch.distributions.categorical import  Categorical

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
        return tuple(self.state, self.action, self.reward, self.probs, self.vals, self.done)

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
            nn.Linear(input_shape, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, actions_space),
            nn.Softmax(dim = -1)
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
            nn.Linear(input_shape, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
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


class PPOAgent(Agent):
    def __init__(self, input_shape, actions_space, args,env):
        super().__init__(name="PPOAgent")
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

        # TODO: optimizers

        self.memory = PPOMemory(args.batch_size)

    def remember(self, transition):
        self.memory.push(transition)

    def save_models(self):
        self.actor.save_model()
        self.critic.save_model()
        torch.save(self.action_var, 'var')

    def load_models(self):
        self.actor.load_model()
        self.critic.load_model()
        self.action_var = torch.load('var')
        # print(self.action_var)

    def set_action_std(self):
        new_action_std = max(0.1, self.action_std * 0.99)
        self.action_var = torch.full((self.actions_space,), new_action_std * new_action_std).to(device)

    def get_action(self, obs: np.ndarray,mask,must_guess=False):
        state = torch.tensor(obs, dtype=torch.float).to(device)
        # state = torch.tensor([obs], dtype=torch.float).to(device)

        dist = self.actor(state)
        dist.probs *= mask
        value = self.critic(state)
        action = torch.tensor(self.actions_space - 1,device = device) if must_guess else dist.sample()

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
            advantages = advantages.to(device)

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







def play_episode(env,
                 agent: PPOAgent,
                 replay_memory: ReplayMemory,
                 eps: float,
                 train_guesser=True,
                 train_dqn=True) -> Tuple[int,int]:
    """Play an epsiode and train
    Args:
        env (gym.Env): gym environment (CartPole-v0)
        agent (Agent): agent will train and get action
        replay_memory (ReplayMemory): trajectory is saved here
        eps (float): 𝜺-greedy for exploration

    Returns:
        Tuple[int,int]: reward earned in this episode, and steps
    """
    obs = env.reset(train_guesser=train_guesser)
    done = False
    total_reward = 0
    mask = env.reset_mask()
    # agent.set_action_std()
    t = 0
    while not done:
        must_guess =  t + 1 == FLAGS.episode_length
        action, prob, val =  agent.get_action(obs,mask,must_guess)

        obs_, reward, done, info,true_y = env.step(action)
        mask[action] = 0

        total_reward += reward

        transition = Transition(obs, action, reward, prob, val, done)
        agent.remember(transition)


        obs = obs_
        t += 1

        if t == FLAGS.episode_length:
            break


    return total_reward, t


env = Mnist_env(flags=FLAGS,
                device=device)
clear_threshold = 1.

# define agent
input_dim, output_dim = get_env_dim(env)
agent = PPOAgent(input_dim, output_dim,FLAGS,env)

env.guesser.to(device=device)

def main():
    """ Main """

    # # delete model files from previous runs
    # if os.path.exists(FLAGS.save_dir):
    #     env.guesser, agent.dqn = load_networks(i_episode='best')
    #     # shutil.rmtree(FLAGS.save_dir)




    # store best result
    best_val_acc = 0

    # counter of validation trials with no improvement, to determine when to stop training
    val_trials_without_improvement = 0

    # set up trainees for first cycle
    train_guesser = False
    train_dqn = True

    rewards = deque(maxlen=100)
    steps = deque(maxlen=100)

    replay_memory = PPOMemory(batch_size= FLAGS.batch_size)
    start_time = time.time()
    print(f"start_time: {start_time}")

    learn_iters, avg_score, n_steps = 0, 0, 0
    for i in range(1000000):

        # determint whether gesser or dqn is trained
        if i % (2 * FLAGS.ep_per_trainee) == FLAGS.ep_per_trainee:
            train_dqn = False
            train_guesser = True
        if i % (2 * FLAGS.ep_per_trainee) == 0:
            train_dqn = True
            train_guesser = False

        # set exploration epsilon
        eps = epsilon_annealing(i, FLAGS.max_episode, FLAGS.min_eps)

        # play an episode
        r, t = play_episode(env,
                            agent,
                            replay_memory,
                            eps,
                            train_dqn=train_dqn,
                            train_guesser=train_guesser)
        n_steps += 1
        # store rewards and episode length
        rewards.append(r)
        steps.append(t)

        # print results to console
        print(f"\r[Episode: {i:5}], Steps: {t}, Avg steps: {np.mean(steps):1.3f},"
              f" Reward: {r:1.3f}, Avg reward: {np.mean(rewards):1.3f}, 𝜺-greedy: {eps:5.2f}",end='')


        # check if environment is solved
        if len(rewards) == rewards.maxlen:
            if np.mean(rewards) >= clear_threshold:
                print("Environment solved in {} episodes with {:1.3f}".format(i, np.mean(rewards)))
                print(f"elapsed time: {time.time() - start_time} seconds")
                break

        if n_steps % FLAGS.learning_cycles == 0:
            agent.learn()
            learn_iters += 1
            best_val_acc = val(i,best_val_acc)

        # if i % FLAGS.val_interval == 0:
        #     # compute performance on validation set
        #     new_best_val_acc = val(i_episode=i,
        #                            best_val_acc=best_val_acc)
        #
        #     # update best result on validation set and counter
        #     if new_best_val_acc > best_val_acc:
        #         best_val_acc = new_best_val_acc
        #         val_trials_without_improvement = 0
        #         print(f"elapsed time: { round(time.time() - start_time)} seconds")
        #     else:
        #         val_trials_without_improvement += 1
        #
        # if val_trials_without_improvement == int(FLAGS.val_trials_wo_im / 2):
        #     env.guesser, agent.dqn = load_networks(i_episode='best')
        #
        # # check whether to stop training
        # if val_trials_without_improvement == FLAGS.val_trials_wo_im:
        #     print('Did not achieve val acc improvement for {} trials, training is done.'.format(FLAGS.val_trials_wo_im))
        #     print(f"elapsed time: {time.time() - start_time} seconds")
        #     break

def val(i_episode: int,
        best_val_acc: float) -> float:
    """ Compute performance on validation set and save current models """

    print('\nRunning validation')
    y_hat_val = np.zeros(len(env.y_val))

    for i in range(len(env.X_val)):  # count(1)

        ep_reward = 0
        state = env.reset(mode='val',
                          patient=i,
                          train_guesser=False)
        mask = env.reset_mask()

        # run episode
        for t in range(FLAGS.episode_length):
            must_guess = t == FLAGS.episode_length - 1
            # select action from policy
            action, _ , __ = agent.get_action(state, mask=mask,must_guess=must_guess)
            mask[action] = 0

            # take the action
            state, reward, done, guess,_ = env.step(action, mode='val')

            if guess != -1:
                y_hat_val[i] = torch.argmax(env.probs).item()

            ep_reward += reward

            if done:
                break

    confmat = confusion_matrix(env.y_val, y_hat_val)
    acc = np.sum(np.diag(confmat)) / len(env.y_val)
    # save_networks(i_episode, acc)

    if acc > best_val_acc:
        print(f'New best acc acheievd, saving best model {acc:0.4}')
        # save_networks(i_episode, acc)
        # save_networks(i_episode='best')
        return acc
    else:
        print(f'\rlast validation accuracy: {acc:0.4}')
        return best_val_acc


main()