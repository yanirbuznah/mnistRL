"""
1. import pytorch
2. train mnist classifier
"""

import torch
from torch.autograd import Variable
from torchvision import datasets, transforms

# Training settings
from Guesser import Guesser
from guesser_parses import FLAGS

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(epoch, guesser, train_loader):
    guesser.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # data =  data.view(-1,784)
        data, target = torch.cat((data, torch.ones_like(data)), dim=1).to(device), target.to(device)
        # print(data.shape)
        guesser.optimizer.zero_grad()
        logits, probs = guesser(data)
        loss = guesser.criterion(logits, target)
        loss.backward()
        guesser.optimizer.step()
        # if batch_idx % 10 == 0:
        print(
            f'\rTrain Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]'
            f'\tLoss: {loss.item():.6f}', end='')


def test(guesser, test_loader):
    guesser.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = torch.cat((data, torch.ones_like(data)), dim=1).to(device), target.to(device)
        logits, probs = guesser(data)
        # sum up batch loss
        test_loss += guesser.criterion(logits, target).item()
        # get the index of the max
        pred = probs.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print(
        f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
    return correct / len(test_loader.dataset)


def fit(guesser, train_loader, test_loader):
    best_val_acc = 0
    for epoch in range(1, 10):
        train(epoch, guesser, train_loader)
        acc = test(guesser, test_loader)
        if acc > best_val_acc:
            best_val_acc = acc
            guesser.save_network(epoch,FLAGS.save_dir, acc)
            guesser.save_network(i_episode='best' ,save_dir=FLAGS.save_dir)


def main():
    batch_size = 64

    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./data/',
                                   train=True,
                                   transform=transforms.ToTensor(),
                                   download=True)

    test_dataset = datasets.MNIST(root='./data/',
                                  train=False,
                                  transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # Load data and randomly split to train, validation and test sets
    n_questions = 28 * 28

    # Initialize guesser
    guesser = Guesser(state_dim=2 * n_questions, hidden_dim=FLAGS.hidden_dim, pretrain=True)

    guesser.to(device=device)
    fit(guesser, train_loader, test_loader)


if __name__ == '__main__':
    main()
