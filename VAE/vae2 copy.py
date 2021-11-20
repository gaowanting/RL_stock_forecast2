import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import os
import pandas as pd
from stock_DataLoader import StockDataset
from torch.utils.data import DataLoader



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(300, 100)
        self.fc21 = nn.Linear(100, 20)
        self.fc22 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, 100)
        self.fc4 = nn.Linear(100, 300)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        # standardization
        std = logvar.mul(1.0e-5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        # return F.sigmoid()
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    print('recon_x',recon_x)
    print('x',x)
    breakpoint()
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # print('BCE',BCE)
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    # print('KLD_element',KLD_element)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # print('KLD',KLD)
    # KL divergence
    breakpoint()
    return BCE + KLD


if __name__ == "__main__":
    num_epochs = 100
    batch_size = 12
    learning_rate = 1e-3

    # dataloader
    df = pd.read_csv(r".\data\STOCK\data.csv")
    train_data = StockDataset(df)
    dataloader = DataLoader(train_data,batch_size=1, shuffle=True)

    # load model
    model = VAE()
    if torch.cuda.is_available():
        model.cuda()
    reconstruction_function = nn.MSELoss(size_average=False)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # train
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(dataloader): 
            data = data.view(data.size(0), -1)
            data = Variable(data)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            #  DEBUG!!! loss_function
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            # train_loss += loss.data[0]
            train_loss += loss.data.item()
            optimizer.step()
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(dataloader.dataset),
                    100. * batch_idx / len(dataloader),
                    # loss.data[0] / len(img)))
                    loss.data.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(dataloader.dataset)))
        # if epoch % 10 == 0:
        #     save = recon_batch.cpu().data
        #     save_image(save, './vae_img/image_{}.png'.format(epoch))
        torch.save(model.state_dict(), './vae_stock.pth')

# RuntimeError: mat1 and mat2 shapes cannot be multiplied (80x15 and 4x100)
# RuntimeError: expected scalar type Double but found Float