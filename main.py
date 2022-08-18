"""
Created on Wed Nov 13 21:20:04 2019

Code is based on the official tutorial in
https://gym.openai.com/evaluations/eval_onwKGm96QkO9tJwdX7L0Gw/
"""
import os
import shutil
import torch
import torch.nn
import numpy as np
from collections import deque
from typing import List, Tuple
from sklearn.metrics import confusion_matrix

from agent import Agent, Transition, ReplayMemory
from dqn import DQNAgent
from ddqnagent import DDQNAgent

from mnist_env import Mnist_env
from mnist_env import Guesser
import utils
from parses import FLAGS

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def train_helper(agent: Agent,
                 minibatch: List[Transition],
                 gamma: float) -> float:
    """Prepare minibatch and train them
    Args:
        agent (Agent): Agent has `train(Q_pred, Q_true)` method
        minibatch (List[Transition]): Minibatch of `Transition`
        gamma (float): Discount rate of Q_target
    Returns:
        float: Loss value
    """
    states = np.vstack([x.state for x in minibatch])
    actions = np.array([x.action for x in minibatch])
    rewards = np.array([x.reward for x in minibatch])
    next_states = np.vstack([x.next_state for x in minibatch])
    done = np.array([x.done for x in minibatch])

    Q_predict = agent.get_Q(states)
    Q_target = Q_predict.clone().cpu().data.numpy()
    max_actions = np.argmax(agent.get_Q(next_states).cpu().data.numpy(), axis=1)
    Q_target[np.arange(len(Q_target)), actions] = rewards + gamma * agent.get_target_Q(next_states)[
        np.arange(len(Q_target)), max_actions].data.numpy() * ~done
    Q_target = agent._to_variable(Q_target).to(device=device)

    return agent.train(Q_predict, Q_target)


def play_episode(env,
                 agent: Agent,
                 replay_memory: ReplayMemory,
                 eps: float,
                 batch_size: int,
                 train_guesser=True,
                 train_dqn=True) -> int:
    """Play an epsiode and train
    Args:
        env (gym.Env): gym environment (CartPole-v0)
        agent (Agent): agent will train and get action
        replay_memory (ReplayMemory): trajectory is saved here
        eps (float): 𝜺-greedy for exploration
        batch_size (int): batch size
    Returns:
        int: reward earned in this episode
    """
    s = env.reset(train_guesser=train_guesser)
    done = False
    total_reward = 0
    mask = env.reset_mask()

    t = 0
    while not done:

        a = agent.get_action(s, eps, mask)
        s2, r, done, info = env.step(a)
        mask[a] = 0

        total_reward += r

        replay_memory.push(s, a, r, s2, done)

        if len(replay_memory) > batch_size:

            if train_dqn:
                minibatch = replay_memory.pop(batch_size)
                train_helper(agent, minibatch, FLAGS.gamma)

        s = s2
        t += 1

        if t == FLAGS.episode_length:
            break

    if train_dqn:
        agent.update_learning_rate()

    return total_reward, t


def get_env_dim(env) -> Tuple[int, int]:
    """Returns input_dim & output_dim
    Args:
        env (gym.Env): gym Environment
    Returns:
        int: input_dim
        int: output_dim
    """
    input_dim = env.n_questions
    output_dim = env.n_questions + 1

    return input_dim, output_dim


def epsilon_annealing(episode: int, max_episode: int, min_eps: float) -> float:
    """Returns 𝜺-greedy
    1.0---|\
          | \
          |  \
    min_e +---+------->
              |
              max_episode
    Args:
        epsiode (int): Current episode (0<= episode)
        max_episode (int): After max episode, 𝜺 will be `min_eps`
        min_eps (float): 𝜺 will never go below this value
    Returns:
        float: 𝜺 value
    """

    slope = (min_eps - 1.0) / max_episode
    return max(slope * episode + 1.0, min_eps)


# define envurinment and agent (needed for main and test)
env = Mnist_env(flags=FLAGS,
                device=device,
                load_pretrained_guesser=False
                )
clear_threshold = 1.

# define agent
input_dim, output_dim = get_env_dim(env)
agent = DDQNAgent(input_dim,
                  output_dim,
                  FLAGS.hidden_dim,
                  env = env)

agent.dqn.to(device=device)
env.guesser.to(device=device)


def save_networks(i_episode: int,
                  val_acc=None) -> None:
    """ A method to save parameters of guesser and dqn """
    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    if i_episode == 'best':
        guesser_filename = 'best_guesser.pth'
        dqn_filename = 'best_dqn.pth'
    else:
        guesser_filename = '{}_{}_{:1.3f}.pth'.format(i_episode, 'guesser', val_acc)
        dqn_filename = '{}_{}_{:1.3f}.pth'.format(i_episode, 'dqn', val_acc)

    guesser_save_path = os.path.join(FLAGS.save_dir, guesser_filename)
    dqn_save_path = os.path.join(FLAGS.save_dir, dqn_filename)

    # save guesser
    if os.path.exists(guesser_save_path):
        os.remove(guesser_save_path)
    torch.save(env.guesser.cpu().state_dict(), guesser_save_path + '~')
    env.guesser.to(device=device)
    os.rename(guesser_save_path + '~', guesser_save_path)

    # save dqn
    if os.path.exists(dqn_save_path):
        os.remove(dqn_save_path)
    torch.save(agent.dqn.cpu().state_dict(), dqn_save_path + '~')
    agent.dqn.to(device=device)
    os.rename(dqn_save_path + '~', dqn_save_path)


def load_networks(i_episode: int,
                  val_acc=None) -> None:
    """ A method to load parameters of guesser and dqn """
    if i_episode == 'best':
        guesser_filename = 'best_guesser.pth'
        dqn_filename = 'best_dqn.pth'
    else:
        guesser_filename = '{}_{}_{:1.3f}.pth'.format(i_episode, 'guesser', val_acc)
        dqn_filename = '{}_{}_{:1.3f}.pth'.format(i_episode, 'dqn', val_acc)

    guesser_load_path = os.path.join(FLAGS.save_dir, guesser_filename)
    dqn_load_path = os.path.join(FLAGS.save_dir, dqn_filename)

    # load guesser
    guesser = Guesser(state_dim=env.n_questions,
                      hidden_dim=FLAGS.g_hidden_dim,
                      lr=FLAGS.lr,
                      min_lr=FLAGS.min_lr,
                      weight_decay=FLAGS.g_weight_decay,
                      decay_step_size=FLAGS.decay_step_size,
                      lr_decay_factor=FLAGS.lr_decay_factor)

    guesser_state_dict = torch.load(guesser_load_path)
    guesser.load_state_dict(guesser_state_dict)
    guesser.to(device=device)

    # load sqn
    dqn = DQNAgent(input_dim, output_dim, FLAGS.hidden_dim)
    dqn_state_dict = torch.load(dqn_load_path)
    dqn.load_state_dict(dqn_state_dict)
    dqn.to(device=device)

    return guesser, dqn


def main():
    """ Main """

    # delete model files from previous runs
    if os.path.exists(FLAGS.save_dir):
        env.guesser, agent.dqn = load_networks(i_episode='best')
        # shutil.rmtree(FLAGS.save_dir)

    # store best result
    best_val_acc = 0

    # counter of validation trials with no improvement, to determine when to stop training
    val_trials_without_improvement = 0

    # set up trainees for first cycle
    train_guesser = False
    train_dqn = True

    rewards = deque(maxlen=100)
    steps = deque(maxlen=100)

    replay_memory = ReplayMemory(FLAGS.capacity)

    for i in range(100000):

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
                            FLAGS.batch_size,
                            train_dqn=train_dqn,
                            train_guesser=train_guesser)

        # store rewards and episode length
        rewards.append(r)
        steps.append(t)

        # print results to console
        print(
            "[Episode: {:5}], Steps: {}, Avg steps: {:1.3f}, Reward: {:1.3f}, Avg reward: {:1.3f}, 𝜺-greedy: {:5.2f}".format(
                i, t, np.mean(steps), r, np.mean(rewards), eps))

        # check if environment is solved
        if len(rewards) == rewards.maxlen:
            if np.mean(rewards) >= clear_threshold:
                print("Environment solved in {} episodes with {:1.3f}".format(i, np.mean(rewards)))
                break

        if i % FLAGS.val_interval == 0:
            # compute performance on validation set
            new_best_val_acc = val(i_episode=i,
                                   best_val_acc=best_val_acc)

            # update best result on validation set and counter
            if new_best_val_acc > best_val_acc:
                best_val_acc = new_best_val_acc
                val_trials_without_improvement = 0
            else:
                val_trials_without_improvement += 1

        if val_trials_without_improvement == int(FLAGS.val_trials_wo_im / 2):
            env.guesser, agent.dqn = load_networks(i_episode='best')

        # check whether to stop training
        # if val_trials_without_improvement == FLAGS.val_trials_wo_im:
        #    print('Did not achieve val acc improvement for {} trials, training is done.'.format(FLAGS.val_trials_wo_im))
        #    break

        if i % FLAGS.n_update_target_dqn == 0:
            agent.update_target_dqn()


def val(i_episode: int,
        best_val_acc: float) -> float:
    """ Compute performance on validation set and save current models """

    print('Running validation')
    y_hat_val = np.zeros(len(env.y_val))

    for i in range(len(env.X_val)):  # count(1)

        ep_reward = 0
        state = env.reset(mode='val',
                          patient=i,
                          train_guesser=False)
        mask = env.reset_mask()

        # run episode
        for t in range(FLAGS.episode_length):

            # select action from policy
            action = agent.get_action(state, eps=0, mask=mask)
            mask[action] = 0

            # take the action
            state, reward, done, guess = env.step(action, mode='val')

            if guess != -1:
                y_hat_val[i] = torch.argmax(env.probs).item()

            ep_reward += reward

            if done:
                break

    confmat = confusion_matrix(env.y_val, y_hat_val)
    acc = np.sum(np.diag(confmat)) / len(env.y_val)
    # save_networks(i_episode, acc)

    if acc > best_val_acc:
        print('New best acc acheievd, saving best model')
        save_networks(i_episode, acc)
        save_networks(i_episode='best')

        return acc

    else:
        return best_val_acc


def test():
    """ Computes performance nad test data """

    print('Loading best networks')
    env.guesser, agent.dqn = load_networks(i_episode='best')
    # env.guesser, agent.dqn = load_networks(i_episode='best', avg_reward = )

    # predict outcome on test data
    y_hat_test = np.zeros(len(env.y_test))
    y_hat_test_prob = np.zeros(len(env.y_test))

    print('Computing predictions of test data')
    n_test = len(env.X_test)
    for i in range(n_test):

        if i % 1000 == 0:
            print('{} / {}'.format(i, n_test))

        state = env.reset(mode='test',
                          patient=i,
                          train_guesser=False)
        mask = env.reset_mask()

        # run episode
        for t in range(FLAGS.episode_length):

            # select action from policy
            action = agent.get_action(state, eps=0, mask=mask)
            mask[action] = 0

            # take the action
            state, reward, done, guess = env.step(action, mode='test')

            if guess != -1:
                y_hat_test_prob[i] = torch.argmax(env.probs).item()

            if done:
                break
        y_hat_test[i] = guess

    C = confusion_matrix(env.y_test, y_hat_test)
    print('confusion matrix: ')
    print(C)

    acc = np.sum(np.diag(C)) / len(env.y_test)

    print('Test accuracy: ', np.round(acc, 3))


def view_images(nun_images=10, save=True):
    print('Loading best networks')
    env.guesser, agent.dqn = load_networks(i_episode='best')

    # delete model files from previous runs
    if os.path.exists(FLAGS.masked_images_dir):
        shutil.rmtree(FLAGS.masked_images_dir)

    if save:
        if not os.path.exists(FLAGS.masked_images_dir):
            os.makedirs(FLAGS.masked_images_dir)

    for i in range(nun_images):
        patient = np.random.randint(len(env.y_test))
        state = env.reset(mode='test',
                          patient=patient,
                          train_guesser=False)
        mask = env.reset_mask()

        actions = []
        for t in range(FLAGS.episode_length):

            # select action from policy
            action = agent.get_action(state, eps=0, mask=mask)
            mask[action] = 0
            actions += [action]
            # take the action
            state, reward, done, guess = env.step(action, mode='test')

            if done:
                break

        image = (env.X_test[patient] + 1.) / 2.
        for j in range(len(actions)):
            if actions[j] < 28 * 28:
                image[actions[j]] = -1
        utils.plot_mnist_digit(digit=image,
                               guess=guess,
                               true_label=env.y_test[patient],
                               num_steps=t,
                               save=save,
                               fig_num=i,
                               save_dir=FLAGS.masked_images_dir,
                               actions=actions)


if __name__ == '__main__':
    main()
    test()
    view_images(10)

