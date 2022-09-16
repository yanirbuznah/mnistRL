import torch
import torch.nn as nn
import numpy as np

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1
BATCH_SIZE = 64
# learning rate

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class AutoEncoder(nn.Module):
    def __init__(self,output_dim = 50):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),   # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

    def forward_encoder(self, x):
        return self.encoder(x)

    def train_autoencoder(self,train_loader,loss_func = nn.MSELoss()):
        for epoch in range(EPOCH):
            for step, (x,) in enumerate(train_loader):
                b_x = x.view(-1, 28*28).to(device)   # batch x, shape (batch, 28*28)
                b_y = x.view(-1, 28*28).to(device)  # batch y, shape (batch, 28*28)

                encoded, decoded = self(b_x)

                loss = loss_func(decoded, b_y)      # mean square error
                self.optimizer.zero_grad()               # clear gradients for this training step
                loss.backward()                     # backpropagation, compute gradients
                self.optimizer.step()                    # apply gradients

                if step % 100 == 0:
                    print(f'\rAutoEncoder - Epoch: {epoch} | Step: {step}\{len(train_loader)} | Train loss: {loss.data.numpy():.4f}', end='')

        print('\n')