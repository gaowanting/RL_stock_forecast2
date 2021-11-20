import torch
import torchvision
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2

train_dataset = datasets.MNIST(root = './VAE/data', train = True,
                               transform = transforms.ToTensor(), download = False)
# test_dataset = datasets.MNIST(root = 'data/', train = False,
#                                transform = transforms.ToTensor(), download = False)
for i in enumerate(train_dataset):
    breakpoint()


train_loader = DataLoader(dataset = train_dataset, batch_size = 100, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size= 100, shuffle = True)

images, lables = next(iter(train_loader))
img = torchvision.utils.make_grid(images, nrow = 10)
img = img.numpy().transpose(1, 2, 0)
cv2.imshow('img', img)
cv2.waitKey(0)
