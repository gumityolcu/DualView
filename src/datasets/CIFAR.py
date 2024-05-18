from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch
import os
class CIFAR(CIFAR10):
    default_class_groups = [[i] for i in range(10)]
    name='CIFAR'
    default_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703233, 0.24348505, 0.26158768))
    ])
    inverse_transform = transforms.Compose([transforms.Normalize(mean=(0., 0., 0.),
                                                                 std=(
                                                                 1. / 0.24703233, 1. / 0.24348505, 1. / 0.26158768)),
                                            transforms.Normalize(mean=(-0.49139968, -0.48215827, -0.44653124),
                                                                 std=(1., 1., 1.)),
                                            ])
    class_labels=["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    def __init__(
            self,
            root,
            split="train",
            transform=None,
            inv_transform=None,
            target_transform=None,
            download=False,
            validation_size=2000
    ):
        if transform is None:
            transform=CIFAR.default_transform
        if inv_transform is None:
            inv_transform=CIFAR.inverse_transform
        train=(split=="train")
        super().__init__(root,train,transform,target_transform,download)
        self.split=split
        self.inverse_transform=inv_transform #MUST HAVE THIS FOR MARK DATASET TO WORK
        self.classes=[i for i in range(10)]
        N = super(CIFAR, self).__len__()

        if not train:
            if (os.path.isfile("datasets/CIFAR_val_ids") and os.path.isfile("datasets/CIFAR_test_ids")):
                self.val_ids=torch.load("datasets/CIFAR_val_ids")
                self.test_ids=torch.load("datasets/CIFAR_test_ids")
            else:
                torch.manual_seed(42)  # THIS SHOULD NOT BE CHANGED BETWEEN TRAIN TIME AND TEST TIME
                perm = torch.randperm(N)
                self.val_ids = torch.tensor([i for i in perm[:validation_size]])
                self.test_ids = torch.tensor([i for i in perm[validation_size:]])
                torch.save(self.val_ids, 'datasets/CIFAR_val_ids')
                torch.save(self.test_ids, 'datasets/CIFAR_test_ids')

            print("Validation ids:")
            print(self.val_ids)
            print("Test ids:")
            print(self.test_ids)
            self.test_targets=self.targets.clone().detach()[self.test_ids]


    def __getitem__(self, item):
        if self.split=="train":
            id=item
        elif self.split=="val":
            id=self.val_ids[item]
        else:
            id=self.test_ids[item]

        x,y=super().__getitem__(id)
        return x,y


    def __len__(self):
        if self.split=="train":
            return super(CIFAR, self).__len__()
        elif self.split=="val":
            return len(self.val_ids)
        else:
            return len(self.test_ids)
