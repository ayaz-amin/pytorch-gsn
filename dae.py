from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import torch.distributions as dist
from torchvision import datasets, transforms
from torchvision.utils import save_image

NUM_WALKBACKS = 100
BATCH_SIZE = 32
ITERATIONS = 10

torch.manual_seed(1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True)


class DAE(nn.Module):
    def __init__(self):
        super(DAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 392)
        self.fc3 = nn.Linear(392, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        x = x + torch.randn_like(x)
        h1 = F.relu(self.fc1(x))
        return self.fc2(h1)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        x = x.view(-1, 784)
        z = self.encode(x)
        return self.decode(z)


model = DAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = F.binary_cross_entropy(recon_batch, data.view(-1, 784), reduction="sum")
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon_batch = model(data)
            loss = F.binary_cross_entropy(recon_batch, data.view(-1, 784), reduction="sum")
            test_loss += loss
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                      recon_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
                save_image(comparison.cpu(),
                         'results_dae/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    sample = torch.randn(64, 784).to(device)
    for epoch in range(1, ITERATIONS + 1):
        train(epoch)
        test(epoch)
        with torch.no_grad():
            sample_x = model(sample)
            for _ in range(NUM_WALKBACKS):
                sample_x = model(sample_x)
            sample_x = sample_x.cpu()
            save_image(sample_x.view(64, 1, 28, 28),
                       'results_dae/sample_' + str(epoch) + '.png')
