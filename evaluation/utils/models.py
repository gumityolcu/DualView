import torch.utils.data
import torchvision
from models import BasicConvModel, CIFARResNet
from models.resnet import resnet18
import tqdm

def load_model(model_name, dataset_name, num_classes):
    bias = not ('homo' in model_name)
    params={
        'MNIST':
            {
                'convs': {
                        'num': 3,
                        'padding': 0,
                        'kernel': 3,
                        'stride': 1,
                        'features': [5, 10, 5]
                    },

                'fc' : {
                    'num': 2,
                    'features': [500, 100]
                },

                'input_shape':(1,28,28)
            },
        'FashionMNIST':
            {
                'convs': {
                    'num': 3,
                    'padding': 0,
                    'kernel': 3,
                    'stride': 1,
                    'features': [5, 10, 5]
                },

                'fc': {
                    'num': 2,
                    'features': [500, 100]
                },

                'input_shape': (1, 28, 28)
            },
        'CIFAR':
            {
            'convs': {
            'num': 5,
            'padding': 0,
            'kernel': 3,
            'stride': 1,
            'features': [5, 10, 10, 10, 5]
        },
            'fc': {
                'num': 2,
                'features': [500, 100]
            },
            'input_shape': (3, 32, 32)
        }
    }

    params=params[dataset_name]
    return BasicConvModel(input_shape=params['input_shape'], convs=params['convs'], fc=params['fc'], num_classes=num_classes, bias=bias)

def load_cifar_model(model_path,dataset_type,num_classes,device):
    model=resnet18()
    model.fc = torch.nn.Linear(in_features=model.fc.in_features, out_features=num_classes, bias=False)
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint={key[6:]: value for key,value in checkpoint['state_dict'].items()}

    model.load_state_dict(checkpoint)
    model.eval()
    return CIFARResNet(model,device=device)

def compute_accuracy(model, test, device):
    loader=torch.utils.data.DataLoader(test, 64, shuffle=False)
    acc = 0.
    model.eval()
    index = 0
    fault_list = []

    for x, y in tqdm.tqdm(loader):
        x=x.to(device)
        if isinstance(y,list):
            y=y[1]
        y=y.to(device)
        real_out = torch.argmax(model(x), dim=1)
        preds = (y == real_out)
        racc = torch.sum(preds)
        if racc != 64:
            for j in range(x.size(0)):
                if not preds[j]:
                    fault_list.append(j + index)
        acc = acc + racc
        index += 64
    NNN = len(test)
    acc=acc/NNN
    return acc, fault_list
