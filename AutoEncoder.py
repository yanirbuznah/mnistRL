import os

import torch
import torch.nn as nn

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 64
# learning rate

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),  # compress to 3 features which can be visualized in plt
        )

    def forward(self, x):
        x = torch.flatten(x,start_dim=1)
        return self.encoder(x)

    def save_network(self, save_dir, name):
        """ A function that saves the gesser params"""

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, name+'_encoder')

        # save guesser
        if os.path.exists(save_path):
            os.remove(save_path)
        torch.save(self.state_dict(), save_path)

    def load_networks(self, filename):
        self.load_state_dict(torch.load(filename))


class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )

    def forward(self, x):
        return self.decoder(x)
    def save_network(self, save_dir, name):
        """ A function that saves the gesser params"""

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, name+'_decoder')

        # save guesser
        if os.path.exists(save_path):
            os.remove(save_path)
        torch.save(self.state_dict(), save_path)

    def load_networks(self, filename):
        self.load_state_dict(torch.load(filename))

class AutoEncoder(nn.Module):
    def __init__(self, bottleneck_dim=100, input_dim=28 * 28):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, bottleneck_dim),  # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid(),  # compress to a range (0, 1)
        )
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def forward_encoder(self, x):
        return self.encoder(x)

    def train_autoencoder(self, train_loader, loss_func=nn.MSELoss()):
        for epoch in range(EPOCH):
            for step, (x,_) in enumerate(train_loader):
                b_x = x.view(-1, 28*28).to(device)   # batch x, shape (batch, 28*28)
                b_y = x.view(-1, 28*28).to(device)  # batch y, shape (batch, 28*28)

                encoded, decoded = self(b_x)

                loss = loss_func(decoded, b_y)      # mean square error
                self.optimizer.zero_grad()               # clear gradients for this training step
                loss.backward()                     # backpropagation, compute gradients
                self.optimizer.step()                    # apply gradients

                if step % 100 == 0:
                    print(
                        f'\rAutoEncoder - Epoch: {epoch} | Step: {step}\{len(train_loader)} | Train loss: {loss.data.cpu().numpy():.4f}',
                        end='')

        print('\n')

    def save_network(self, save_dir, name):
        """ A function that saves the gesser params"""

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        save_path = os.path.join(save_dir, name)

        # save guesser
        if os.path.exists(save_path):
            os.remove(save_path)
        torch.save(self.state_dict(), save_path)
        # self.encoder.save_network(save_dir,name)
        # self.decoder.save_network(save_dir,name)

    def load_networks(self, filename):
        self.load_state_dict(torch.load(filename))
