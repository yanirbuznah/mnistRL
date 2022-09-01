import random
from collections import namedtuple
from typing import List

import numpy as np
import torch


class Agent:

    def __init__(self, name):
        self.name = name


    def _to_variable(self, x: np.ndarray) -> torch.Tensor:
        """torch.Variable syntax helper
        Args:
            x (np.ndarray): 2-D tensor of shape (n, input_dim)
        Returns:
            torch.Tensor: torch variable
        """
        return torch.autograd.Variable(torch.Tensor(x))

    def get_action(self, states: np.ndarray,
                   eps: float,
                   mask: np.ndarray):
        """Returns an action
        Args:
            states (np.ndarray): 2-D tensor of shape (n, input_dim)
            eps (float): 𝜺-greedy for exploration
            mask (np.ndarray) zeroes out q values for questions that were already asked, so they will not be chosen again
        Returns:
            int: action index
        """
        raise NotImplementedError("get_action() is not implemented")

    def get_Q(self, states: np.ndarray) -> torch.FloatTensor:
        """Returns `Q-value`
        Args:
            states (np.ndarray): 2-D Tensor of shape (n, input_dim)
        Returns:
            torch.FloatTensor: 2-D Tensor of shape (n, output_dim)
        """
        raise NotImplementedError("get_Q() not implemented")

    # virtual methods
    def train(self,minibatch):
        raise NotImplementedError("train not implemented")


Transition = namedtuple("Transition",
                        field_names=["state", "action", "reward", "next_state", "done","y_true"])


class ReplayMemory(object):

    def __init__(self, capacity: int = 1000000, batch_size:int=64) -> None:
        """Replay memory class
        Args:
            capacity (int): Max size of this memory
        """
        self.capacity = capacity
        self.cursor = 0
        self.memory = []
        self.batch_size = batch_size

    def push(self,
             state: np.ndarray,
             action: int,
             reward: int,
             next_state: np.ndarray,
             done: bool,
             true_label) -> None:
        """Creates `Transition` and insert
        Args:
            state (np.ndarray): 1-D tensor of shape (input_dim,)
            action (int): action index (0 <= action < output_dim)
            reward (int): reward value
            next_state (np.ndarray): 1-D tensor of shape (input_dim,)
            done (bool): whether this state was last step
        """
        if len(self) < self.capacity:
            self.memory.append(None)

        self.memory[self.cursor] = Transition(state,
                                              action, reward, next_state, done,true_label)
        self.cursor = (self.cursor + 1) % self.capacity

    def pop(self, batch_size: int = None) -> List[Transition]:
        """Returns a minibatch of `Transition` randomly
        Args:
            batch_size (int): Size of mini-bach
        Returns:
            List[Transition]: Minibatch of `Transition`
        """
        batch_size = self.batch_size if batch_size is None else batch_size
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        """Returns the length """
        return len(self.memory)
