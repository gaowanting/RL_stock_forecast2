import torch
from torch.nn.functional import embedding
from vae2 import VAE
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable

img_transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# dataloader
dataset = MNIST('VAE\data', transform=img_transform, download=False)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

embedding = []
for batch_idx, data in enumerate(dataloader):
    print(batch_idx, len(data))
    img, _ = data
    img = img.view(img.size(0), -1)
    img = Variable(img)
    if torch.cuda.is_available():
        img = img.cuda()
    model = VAE()
    if torch.cuda.is_available():
        model.cuda()
    model.load_state_dict(torch.load(r'VAE\\vae.pth'))
    mu, logvar = model.encode(img)
    z = model.reparametrize(mu, logvar)
    embedding.append(z)


print(embedding)
