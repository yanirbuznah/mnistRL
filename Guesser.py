#TODO:Change
import os

import numpy as np
import torch
from torch import nn
from torch.optim import lr_scheduler

from guesser_parses import FLAGS
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Guesser(nn.Module):
    """
    implements a net that guesses the outcome given the state
    """

    def __init__(self,
                 state_dim,
                 hidden_dim,
                 num_classes=10,
                 lr=1e-3,
                 min_lr=1e-6,
                 weight_decay=0.,
                 decay_step_size=12500,
                 lr_decay_factor=0.1,
                 pretrain = False):
        super(Guesser, self).__init__()

        self.state_dim = state_dim
        self.min_lr = min_lr
        self.pretrain = pretrain
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.PReLU(),
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU(),
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.PReLU(),
        )

        # output layer
        self.logits = nn.Linear(hidden_dim, num_classes)

        self.softmax = nn.Softmax(dim=1)

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.parameters())

        self.lambda_rule = lambda x: np.power(lr_decay_factor, int(np.floor((x + 1) / decay_step_size)))

        self.scheduler = lr_scheduler.LambdaLR(self.optimizer,
                                               lr_lambda=self.lambda_rule)

        self.p = 1.

    def forward(self, x):
        x = x.view(-1,self.state_dim)
        if self.pretrain and self.p > 0.:
            x = self.modified_dropout(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        logits = self.logits(x)
        probs = self.softmax(logits)

        return logits, probs

    def modified_dropout(self, x):

        mask = nn.Dropout(self.p)(torch.ones((x.shape[0],x.shape[1]//2))) * (1-self.p)
        mask.to(device)
        x = torch.concat((x[:,:mask.shape[1]]*mask, x[:,mask.shape[1]:]*mask),dim=1)
        return x


    def update_learning_rate(self):
        """ Learning rate updater """

        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        if lr < self.min_lr:
            self.optimizer.param_groups[0]['lr'] = self.min_lr
            lr = self.optimizer.param_groups[0]['lr']
        # print('Guesser learning rate = %.7f' % lr)

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