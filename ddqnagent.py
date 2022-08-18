import numpy as np
import torch
from torch.optim import lr_scheduler

from dqn import DQNAgent
from parses import FLAGS
from agent import Agent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def lambda_rule(i_episode) -> float:
    """ stepwise learning rate calculator """
    exponent = int(np.floor((i_episode + 1) / FLAGS.decay_step_size))
    return np.power(FLAGS.lr_decay_factor, exponent)


class DDQNAgent(Agent):

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 env) -> None:
        super().__init__(name="DDQNAgent")
        """Agent class that choose action and train
        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            hidden_dim (int): hidden dimension
        """

        self.env = env
        self.dqn = DQNAgent(input_dim, output_dim, hidden_dim)
        self.target_dqn = DQNAgent(input_dim, output_dim, hidden_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.loss_fn = torch.nn.MSELoss()
        self.optim = torch.optim.Adam(self.dqn.parameters(),
                                      lr=FLAGS.lr,
                                      weight_decay=FLAGS.weight_decay)

        self.scheduler = lr_scheduler.LambdaLR(self.optim,
                                               lr_lambda=lambda_rule)

        self.update_target_dqn()

    def update_target_dqn(self):

        # hard copy model parameters to target model parameters
        for param, target_param in zip(self.dqn.parameters(), self.target_dqn.parameters()):
            target_param.data.copy_(param.data)



    def get_action(self, states: np.ndarray,
                   eps: float,
                   mask: np.ndarray) -> int:
        """Returns an action
        Args:
            states (np.ndarray): 2-D tensor of shape (n, input_dim)
            eps (float): 𝜺-greedy for exploration
            mask (np.ndarray) zeroes out q values for questions that were already asked, so they will not be chosen again
        Returns:
            int: action index
        """
        if np.random.rand() < eps:
            r = np.random.rand()
            if r < .2:
                return np.random.choice(self.output_dim)
            elif r < .6:
                return np.random.choice(self.output_dim, p=self.env.action_probs)
            else:
                return self.output_dim - 1
        else:
            self.dqn.train(mode=False)
            scores = self.get_Q(states)
            _, argmax = torch.max(scores.data * mask, 1)
            return int(argmax.item())

    def get_Q(self, states: np.ndarray) -> torch.FloatTensor:
        """Returns `Q-value`
        Args:
            states (np.ndarray): 2-D Tensor of shape (n, input_dim)
        Returns:
            torch.FloatTensor: 2-D Tensor of shape (n, output_dim)
        """
        states = self._to_variable(states.reshape(-1, self.input_dim))
        states = states.to(device=device)
        self.dqn.train(mode=False)
        return self.dqn(states)

    def get_target_Q(self, states: np.ndarray) -> torch.FloatTensor:
        """Returns `Q-value`
        Args:
            states (np.ndarray): 2-D Tensor of shape (n, input_dim)
        Returns:
            torch.FloatTensor: 2-D Tensor of shape (n, output_dim)
        """
        states = self._to_variable(states.reshape(-1, self.input_dim))
        self.target_dqn.train(mode=False)
        return self.target_dqn(states)

    def train(self, Q_pred: torch.FloatTensor, Q_true: torch.FloatTensor) -> float:
        """Computes `loss` and backpropagation
        Args:
            Q_pred (torch.FloatTensor): Predicted value by the network,
                2-D Tensor of shape(n, output_dim)
            Q_true (torch.FloatTensor): Target value obtained from the game,
                2-D Tensor of shape(n, output_dim)
        Returns:
            float: loss value
        """
        self.dqn.train(mode=True)
        self.optim.zero_grad()
        loss = self.loss_fn(Q_pred, Q_true)
        loss.backward()
        self.optim.step()

        return loss

    def update_learning_rate(self):
        """ Learning rate updater """

        self.scheduler.step()
        lr = self.optim.param_groups[0]['lr']
        if lr < FLAGS.min_lr:
            self.optim.param_groups[0]['lr'] = FLAGS.min_lr
            lr = self.optim.param_groups[0]['lr']
        print('DQN learning rate = %.7f' % lr)