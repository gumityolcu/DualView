from torchvision.datasets import MNIST as tvMNIST
from torchvision import transforms
import torch
import os



class MNIST(tvMNIST):
    default_class_groups = [[i] for i in range(10)]
    class_labels=[i for i in range(10)]
    name='MNIST'
    default_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    inverse_transform = transforms.Compose([transforms.Normalize(mean=(0.,),
                                                                 std=(1 / 0.3081,)),
                                            transforms.Normalize(mean=(-0.1307,),
                                                                 std=(1.,)),
                                            ])
    def __init__(
            self,
            root,
            split="train",
            transform=None,
            inv_transform=None,
            target_transform=None,
            download=False,
            validation_size=0
    ):
        if transform is None:
            transform=self.default_transform
        if inv_transform is not None:
            self.inverse_transform = inv_transform  # MUST HAVE THIS FOR MARK DATASET TO WORK
        train=(split=="train")
        super().__init__(root,train,transform,target_transform,download)
        self.split=split
        self.classes=[i for i in range(10)]
        N = super(MNIST, self).__len__()

        if not train:
            if (os.path.isfile(f"datasets/{self.name}_val_ids")and os.path.isfile(f"datasets/{self.name}_test_ids")):
                self.val_ids=torch.load(f"datasets/{self.name}_val_ids")
                self.test_ids=torch.load(f"datasets/{self.name}_test_ids")
            else:
                torch.manual_seed(42)  # THIS SHOULD NOT BE CHANGED BETWEEN TRAIN TIME AND TEST TIME
                perm = torch.randperm(N)
                self.val_ids = torch.tensor([i for i in perm[:validation_size]])
                self.test_ids = torch.tensor([i for i in perm[validation_size:]])
                torch.save(self.val_ids, f'datasets/{self.name}_val_ids')
                torch.save(self.test_ids, f'datasets/{self.name}_test_ids')


            print("Validation ids:")
            print(self.val_ids)
            print("Test ids:")
            print(self.test_ids)
            self.test_targets=torch.tensor(self.targets)[self.test_ids]


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
            return super(MNIST, self).__len__()
        elif self.split=="val":
            return len(self.val_ids)
        else:
            return len(self.test_ids)

class FashionMNIST(MNIST):
    default_class_groups = [[i] for i in range(10)]
    name = 'FashionMNIST'
    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]
    resources = [
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
    ]
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    class_labels = ["t-shirt", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "boot"]
    default_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    inverse_transform = transforms.Compose([transforms.Normalize(mean=(0.,),
                                                                 std=(1 / 0.3530,)),
                                            transforms.Normalize(mean=(-0.2860,),
                                                                 std=(1.,)),
                                            ])