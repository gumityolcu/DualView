from torchvision.datasets.mnist import MNIST as origMNIST, FashionMNIST as origFash
from torchvision.transforms import ToTensor
from datasets import MNIST, FashionMNIST, CIFAR
import torch


#ds=MNIST(root="~/Documents/Code/Datasets", split="train", download=True)
#ds=origMNIST(root="~/Documents/Code/Datasets", train=True, download=True)
#ds=origFash(root="~/Documents/Code/Datasets", train=True, download=True)
#ds=FashionMNIST(root="~/Documents/Code/Datasets", split="train", download=True)
ds=CIFAR(root="~/Documents/Code/Datasets", split="train", download=True)
sample=ds[0][0]
num_data=len(ds)*sample.shape[1]*sample.shape[2]
mean=torch.zeros((3,))
trans=ToTensor()
mean2=ds.data.sum(axis=0).sum(axis=0).sum(axis=0)/(num_data*255)
print(mean2)
for i in ds:
    img=i[0]
    #mean=mean+trans(img).sum()
    mean=mean+img.sum(axis=1).sum(axis=1)

mean=mean/num_data
print(mean)
std=torch.zeros((3,))

for i in ds:
    img=i[0]
    std=std+torch.sum((img-mean[:,None,None])**2,axis=1).sum(axis=1)
    pass
    #std=std+torch.sum((trans(img)-mean)**2)
std=std/num_data
print([f.data for f in torch.sqrt(std)])
