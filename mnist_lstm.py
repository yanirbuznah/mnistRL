# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:49:57 2019

@author: urixs

Environment for questionnaire

"""
import os

import numpy as np
from gym.spaces import Discrete, Box
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split
import gym
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.nn.functional as F

import utils

#
# class Guesser(nn.Module):
#     """
#     implements a net that guesses the outcome given the state
#     """
#
#     """
#     implements LSTM state update mechanism
#     """
#
#     def __init__(self,
#                  embedding_dim,
#                  state_dim,
#                  n_questions,
#                  num_classes=10,
#                  lr=1e-4,
#                  min_lr=1e-6,
#                  weight_decay=0.,
#                  decay_step_size=12500,
#                  lr_decay_factor=0.1,
#                  device=torch.device("cpu")):
#         super(Guesser, self).__init__()
#
#         self.device = device
#
#         self.embedding_dim = embedding_dim
#         self.state_dim = state_dim
#         self.min_lr = min_lr
#
#         # question embedding, we add one a "dummy question" for cases when guess is made at first step
#         self.q_emb = nn.Embedding(num_embeddings=n_questions + 1,
#                                   embedding_dim=self.embedding_dim)
#
#         input_dim = 2 * self.embedding_dim
#         self.lstm = nn.LSTMCell(input_size=input_dim, hidden_size=self.state_dim)
#         self.affine = nn.Linear(self.state_dim, 2)
#
#         self.initial_c = nn.Parameter(torch.randn(1, self.state_dim), requires_grad=True).to(device=self.device)
#         self.initial_h = nn.Parameter(torch.randn(1, self.state_dim), requires_grad=True).to(device=self.device)
#
#         self.reset_states()
#
#         self.criterion = nn.CrossEntropyLoss()
#
#         self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
#
#     def forward(self, question, answer):
#         ind = torch.LongTensor([question]).to(device=self.device)
#         question_embedding = self.q_emb(ind)
#         answer_vec = torch.unsqueeze(torch.ones(self.embedding_dim) * answer, 0)
#         question_embedding = question_embedding.to(device=self.device)
#         answer_vec = answer_vec.to(device=self.device)
#         x = torch.cat((question_embedding,
#                        answer_vec), dim=-1)
#         self.lstm_h, self.ls3tm_c = self.lstm(x, (self.lstm_h, self.lstm_c))
#         logits = self.affine(self.lstm_h)
#         probs = F.softmax(logits, dim=1)
#         return self.lstm_h, logits, probs
#
#     def reset_states(self):
#         self.lstm_h = (torch.zeros(1, self.state_dim) + self.initial_h).to(device=self.device)
#         self.lstm_c = (torch.zeros(1, self.state_dim) + self.initial_c).to(device=self.device)
#
#     def _to_variable(self, x: np.ndarray) -> torch.Tensor:
#         """torch.Variable syntax helper
#         Args:
#             x (np.ndarray): 2-D tensor of shape (n, input_dim)
#         Returns:
#             torch.Tensor: torch variable
#         """
#         return torch.autograd.Variable(torch.Tensor(x))
#
from Guesser import Guesser


class LSTM(nn.Module):
    def __init__(self,embedding_dim,state_dim):
        super().__init__()
        self.state_dim = state_dim
        self.lstm = nn.LSTMCell(input_size=embedding_dim, hidden_size=state_dim)
        self.reset_states()


    def forward(self, x):
        self.lstm_h, self.lstm_c = self.lstm(x, (self.lstm_h, self.lstm_c))
        return self.lstm_h,self.lstm_c

    def pred(self,x):
        self.eval()
        return self.lstm(x, (self.lstm_h, self.lstm_c))

    def reset_states(self):
        self.lstm_h = torch.zeros(1, self.state_dim)
        self.lstm_c = torch.zeros(1, self.state_dim)

class EnvNet(nn.Module):
    def __init__(self,input_dim,embedding_dim,state_dim,output_dim,mlp_hidden_dim = 256):
        super().__init__()
        # self.embedding = nn.Embedding(num_embeddings=input_dim,
        #                               embedding_dim=embedding_dim)
        self.lstm = LSTM(input_dim,state_dim)
        self.mlp = Guesser(state_dim=state_dim,hidden_dim=mlp_hidden_dim,
                           num_classes=output_dim)

        self.optimizer = torch.optim.Adam(self.parameters())

        self.criterion = nn.CrossEntropyLoss()


    def forward(self,x):
        self.train()
        # embedding = self.embedding(x)
        h,_ = self.lstm(x)
        logits,probs = self.mlp(h)
        return logits,probs

    def pred(self,x):
        self.eval()
        # embedding = self.embedding(x)
        h,_ = self.lstm.pred(x)
        logits,probs = self.mlp(h)
        return logits,probs

    def reset(self):
        self.lstm.reset_states()


    def _to_variable(self, x: np.ndarray) -> torch.Tensor:
        """torch.Variable syntax helper
        Args:
            x (np.ndarray): 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: torch variable
        """
        return torch.autograd.Variable(torch.Tensor(x))

    def save_network(self,i_episode, save_dir,acc=None):
        """ A function that saves the gesser params"""

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if i_episode == 'best':
            guesser_filename = 'best_guesser.pth'
        else:
            guesser_filename = '{}_{}_{:1.3f}.pth'.format(i_episode, 'guesser', acc)

        guesser_save_path = os.path.join(save_dir, guesser_filename)

        # save guesser
        if os.path.exists(guesser_save_path):
            os.remove(guesser_save_path)
        torch.save(self.state_dict(), guesser_save_path + '~')

        os.rename(guesser_save_path + '~', guesser_save_path)

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
                 load_pretrained_guesser=False):

        case = flags.case
        episode_length = flags.episode_length
        self.device = device


        # Load data
        self.n_questions = 28 * 28
        # Reset environment
        self.action_space = Discrete(n=self.n_questions + 1)
        self.reward_range = tuple([-1., 1.])
        self.observation_space = Box(-np.ones(self.n_questions), np.ones(self.n_questions))

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

        self.net = EnvNet(self.n_questions , flags.embedding, flags.state_dim,output_dim=10,mlp_hidden_dim=flags.g_hidden_dim)
        self.episode_length = episode_length

        # Load pre-trained guesser network, if needed
        if load_pretrained_guesser:
            save_dir = './pretrained_mnist_guesser_models'
            guesser_filename = 'best_guesser.pth'
            guesser_load_path = os.path.join(save_dir, guesser_filename)
            if os.path.exists(guesser_load_path):
                print('Loading pre-trained guesser')
                guesser_state_dict = torch.load(guesser_load_path)
                self.net.load_state_dict(guesser_state_dict)
        self.net.predict = False
        print('Initialized questionnaire environment')


        self.lstm_loss = None

        print('Initialized LSTM-mnist environment')

        # Reset environment

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

        # Reset state
        self.net.reset()


        self.state = self.net.lstm.lstm_h.data.cpu().numpy()

        if mode == 'training':
            self.patient = np.random.randint(self.X_train.shape[0])
        else:
            self.patient = patient

        self.raw_state = torch.zeros((1,self.X_train[0].shape[0]))

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

        '''        
        # compute reward
        self.reward = self.compute_reward(mode)
        '''
        self.time += 1
        if self.time == self.episode_length:
            self.terminate_episode()

        # compute reward
        self.reward = self.compute_reward(mode)

        return self.s, self.reward, self.done, self.guess, self.true_y

    # Update 'done' flag when episode terminates
    def terminate_episode(self):
        self.done = True

    def update_state(self, action, mode):
        next_state = np.array(self.state)
        # input = self.net._to_variable(self.state.reshape(-1, 2 * self.n_questions)).to(self.device)
        #
        # self.logits, self.probs = self.net(self.raw_state)
        # self.guess = torch.argmax(self.probs.squeeze()).item()
        # if mode == 'training':
        #     # store probability of true outcome for reward calculation
        #     self.correct_prob = self.probs.squeeze()[
        #         self.y_train[self.patient]].item()  # - torch.max(self.probs).item()

        if action < self.n_questions:  # Not making a guess
            if mode == 'training':
                self.raw_state[0][action] = self.X_train[self.patient, action]
            elif mode == 'val':
                self.raw_state[0][action] = self.X_val[self.patient, action]
            elif mode == 'test':
                self.raw_state[0][action] = self.X_test[self.patient, action]

            self.guess = -1
        else:
        # input = self.net._to_variable(next_state.reshape(-1, 2 * self.n_questions)).to(self.device)
            self.logits, self.probs = self.net.pred(self.raw_state)
            self.guess = torch.argmax(self.probs.squeeze()).item()
            if mode == 'training':
                # store probability of true outcome for reward calculation
                self.correct_prob = self.probs.squeeze()[self.y_train[
                    self.patient]].item() #- self.correct_prob  # - torch.max(self.probs).item()

                self.done = True

        return next_state

    def compute_reward(self, mode):
        """ Compute the reward """

        if mode == 'test':
            return None
        if mode == 'training':
            self.true_y = self.y_train[self.patient]

        if self.guess == -1:  # no guess was made
            return .01 * np.random.rand()
        else:
            reward = self.correct_prob
        if self.train_guesser and self.done:
            # train guesser
            self.net.optimizer.zero_grad()
            y = torch.Tensor([self.true_y]).long()
            y = y.to(device=self.device)
            self.net.train(mode=True)
            loss = self.net.criterion(self.logits, y)

            loss.backward()
            self.net.optimizer.step()
            # update learning rate
            # self.guesser.update_learning_rate()

        return reward
