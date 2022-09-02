# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:49:57 2019

@author: urixs

Environment for MNIST

"""
from gym.spaces import Discrete, Box

from Guesser import Guesser

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:49:57 2019

@author: urixs

Environment for MNIST

"""
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif

import gym
import torch

import utils

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Mnist_env(gym.Env):
    """ Questionnaire Environment class
       Args:
           case (int): which data to use
           oversample (Boolean): whether to oversample the small class
           load_pretrained_guesser (Boolean): whether to load a pretrained guesser
       """

    def __init__(self,
                 flags,
                 device,
                 oversample=True,
                 load_pretrained_guesser=True):

        case = flags.case
        episode_length = flags.episode_length
        self.device = device

        # Load data
        self.n_questions = 28 * 28
        self.X_train, self.X_test, self.y_train, self.y_test = utils.load_mnist(case=case)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train,
                                                                              self.y_train,
                                                                              test_size=0.017)

        # Load / compute mutual information of each pixel with target
        mi = utils.load_mi_scores()
        if mi is None:
            print('Computing mutual information of each pixel with target')
            mi = mutual_info_classif(self.X_train, self.y_train)
            np.save('./mnist/mi.npy', mi)
        scores = np.append(mi, .1)
        self.action_probs = scores / np.sum(scores)

        self.guesser = Guesser(state_dim=2 * self.n_questions,
                               hidden_dim=flags.g_hidden_dim,
                               lr=flags.lr,
                               min_lr=flags.min_lr,
                               weight_decay=flags.g_weight_decay,
                               decay_step_size=12500,
                               lr_decay_factor=0.1).to(device)

        self.episode_length = episode_length

        # Load pre-trained guesser network, if needed
        if load_pretrained_guesser:
            save_dir = './pretrained_mnist_guesser_models'
            guesser_filename = 'best_guesser.pth'
            guesser_load_path = os.path.join(save_dir, guesser_filename)
            if os.path.exists(guesser_load_path):
                print('Loading pre-trained guesser')
                guesser_state_dict = torch.load(guesser_load_path)
                self.guesser.load_state_dict(guesser_state_dict)
        self.guesser.predict = False
        print('Initialized questionnaire environment')

        # Reset environment
        self.action_space = Discrete(n = self.n_questions +1)
        self.reward_range = tuple([-1.,1.])
        self.observation_space = Box(-np.ones(2 * self.n_questions),np.ones(2 * self.n_questions))

    def reset(self,
              mode='training',
              patient=0,
              train_guesser=True):
        """
        Args: mode: training / val / test
              patient (int): index of patient
              train_guesser (Boolean): flag indicating whether to train guesser network in this episode

        Selects a patient (random for training, or pre-defined for val and test) ,
        Resets the state to contain the basic information,
        Resets 'done' flag to false,
        Resets 'train_guesser' flag
        """

        self.state = np.concatenate([np.zeros(self.n_questions), np.zeros(self.n_questions)])

        if mode == 'training':
            self.patient = np.random.randint(self.X_train.shape[0])
        else:
            self.patient = patient

        self.done = False
        self.s = np.array(self.state)
        self.time = 0
        if mode == 'training':
            self.train_guesser = train_guesser
        else:
            self.train_guesser = False
        return self.s

    def reset_mask(self):
        """ A method that resets the mask that is applied
        to the q values, so that questions that were already
        asked will not be asked again.
        """
        mask = torch.ones(self.n_questions + 1)
        mask = mask.to(device=self.device)

        return mask

    def step(self,
             action,
             mode='training'):
        """ State update mechanism """

        # update state
        next_state = self.update_state(action, mode)
        self.state = np.array(next_state)
        self.s = np.array(self.state)

        # compute reward
        self.reward = self.compute_reward(mode)

        self.time += 1
        if self.time == self.episode_length:
            self.terminate_episode()

        return self.s, self.reward, self.done, self.guess, self.true_y

    # Update 'done' flag when episode terminates
    def terminate_episode(self):
        self.done = True

    def update_state(self, action, mode):
        next_state = np.array(self.state)
        guesser_input = self.guesser._to_variable(self.state.reshape(-1, 2 * self.n_questions)).to(device)
        self.guesser.train(mode=False)
        self.logits, self.probs = self.guesser(guesser_input)
        self.guess = torch.argmax(self.probs.squeeze()).item()
        if mode == 'training':
            # store probability of true outcome for reward calculation
            self.correct_prob = self.probs.squeeze()[self.y_train[self.patient]].item()  # - torch.max(self.probs).item()

        if action < self.n_questions:  # Not making a guess
            if mode == 'training':
                next_state[action] = self.X_train[self.patient, action]
            elif mode == 'val':
                next_state[action] = self.X_val[self.patient, action]
            elif mode == 'test':
                next_state[action] = self.X_test[self.patient, action]
            next_state[action + self.n_questions] += 1.
            guesser_input = self.guesser._to_variable(next_state.reshape(-1, 2 * self.n_questions)).to(device)
            self.logits, self.probs = self.guesser(guesser_input)
            self.guess = torch.argmax(self.probs.squeeze()).item()
            if mode == 'training':
                # store probability of true outcome for reward calculation
                self.correct_prob = self.probs.squeeze()[self.y_train[self.patient]].item() - self.correct_prob  # - torch.max(self.probs).item()

            self.done = False


        else:  # Making a guess
            # run guesser, and store guess and outcome probability

            self.terminate_episode()

        return next_state

    def compute_reward(self, mode):
        """ Compute the reward """

        if mode == 'test':
            return None
        if mode == 'training':
            self.true_y = self.y_train[self.patient]

        # if self.guess == -1:  # no guess was made
        #     return .01 * np.random.rand()
        # else:
        reward = self.correct_prob
        if self.train_guesser:
            # train guesser
            self.guesser.optimizer.zero_grad()
            y = torch.Tensor([self.true_y]).long()
            y = y.to(device=self.device)
            self.guesser.train(mode=True)
            self.guesser.loss = self.guesser.criterion(self.logits, y)
            self.guesser.loss.backward()
            self.guesser.optimizer.step()
            # update learning rate
            self.guesser.update_learning_rate()

        return reward
