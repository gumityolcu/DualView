from time import time as time
from abc import ABC, abstractmethod
import numpy as np
from torch.utils.data import DataLoader
import torch
import os
import json
from zennit.composites import EpsilonPlus, EpsilonAlpha2Beta1
from zennit.attribution import Gradient
from zennit.image import imgify
import itertools
from zennit.canonizers import AttributeCanonizer, CompositeCanonizer
from models.resnet import BasicBlock, Bottleneck
from zennit.torchvision import SequentialMergeBatchNorm, Sum


# These canonizers are directly copied from zennit and incorporated with our CIFAR resnet model (https://github.com/chr5tphr/zennit)
class CIFARResNetBottleneckCanonizer(AttributeCanonizer):
    '''Canonizer specifically for Bottlenecks of torchvision.models.resnet* type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        '''Create a forward function and a Sum module to overload as new attributes for module.

        Parameters
        ----------
        name : string
            Name by which the module is identified.
        module : obj:`torch.nn.Module`
            Instance of a module. If this is a Bottleneck layer, the appropriate attributes to overload are returned.

        Returns
        -------
        None or dict
            None if `module` is not an instance of Bottleneck, otherwise the appropriate attributes to overload onto
            the module instance.
        '''
        if isinstance(module, Bottleneck):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        '''Modified Bottleneck forward for ResNet.'''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.stack([identity, out], dim=-1)
        out = self.canonizer_sum(out)

        out = self.relu(out)

        return out


class CIFARResNetBasicBlockCanonizer(AttributeCanonizer):
    '''Canonizer specifically for BasicBlocks of torchvision.models.resnet* type models.'''

    def __init__(self):
        super().__init__(self._attribute_map)

    @classmethod
    def _attribute_map(cls, name, module):
        '''Create a forward function and a Sum module to overload as new attributes for module.

        Parameters
        ----------
        name : string
            Name by which the module is identified.
        module : obj:`torch.nn.Module`
            Instance of a module. If this is a BasicBlock layer, the appropriate attributes to overload are returned.

        Returns
        -------
        None or dict
            None if `module` is not an instance of BasicBlock, otherwise the appropriate attributes to overload onto
            the module instance.
        '''
        if isinstance(module, BasicBlock):
            attributes = {
                'forward': cls.forward.__get__(module),
                'canonizer_sum': Sum(),
            }
            return attributes
        return None

    @staticmethod
    def forward(self, x):
        '''Modified BasicBlock forward for ResNet.'''
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = torch.stack([identity, out], dim=-1)
        out = self.canonizer_sum(out)

        out = self.relu(out)

        return out


class CIFARResNetCanonizer(CompositeCanonizer):
    def __init__(self):
        super().__init__((
            SequentialMergeBatchNorm(),
            CIFARResNetBottleneckCanonizer(),
            CIFARResNetBasicBlockCanonizer(),
        ))


def zennit_inner_product_explanation(model, train, test, composite_cls=EpsilonAlpha2Beta1, canonizer=None,
                                     mode="train",cmap_name="bwr"):
    with torch.no_grad():
        train_features = model.features(train)
        test_features = model.features(test)
        init = test_features if mode == "train" else train_features
        # init=init/test_features.abs().max()
    if canonizer is not None:
        attributor = Gradient(model.features, composite_cls(canonizers=[canonizer]))
    else:
        attributor = Gradient(model.features, composite_cls())

    input = train if mode == "train" else test

    output, attr = attributor(input, init)
    attr = attr.cpu()
    img = imgify(attr.sum(1), symmetric=True, cmap=cmap_name)

    sensible_palette = {img.palette.colors[color]: torch.tensor(color, dtype=torch.float32) for color in
                        img.palette.colors.keys()}

    tensor_img = torch.tensor(img.getdata()).resize(1, img.size[0], img.size[1])
    rgb_img = torch.empty((tensor_img.shape[1], tensor_img.shape[2], 3), dtype=torch.float32)
    for i in range(tensor_img.shape[1]):
        for j in range(tensor_img.shape[2]):
            rgb_img[i, j] = sensible_palette[int(tensor_img[0, i, j])]
    # return img, attr.sum(1)
    return rgb_img / 255., attr.sum(1)


def xplain(model, train, test, device, explainer_cls, batch_size, kwargs, num_batches_per_file, save_dir,
           start_file, num_files):
    explainer = explainer_cls(model=model, dataset=train, device=device, **kwargs)
    explainer.train()
    explainer.save_coefs(save_dir)

    test_ld = DataLoader(test, batch_size=batch_size, shuffle=False)
    explanations = torch.empty((0, len(train)), device=device)
    i = 0
    j = start_file
    print(f"Starting file {j}")
    file_indices = torch.zeros(int(len(test) / batch_size) + 1, dtype=torch.int)
    file_indices[start_file * num_batches_per_file:(start_file + num_files) * num_batches_per_file] = 1
    iter_loader = itertools.compress(test_ld, file_indices)
    for u, (x, y) in enumerate(iter_loader):
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)
            preds = torch.argmax(model(x), dim=1)
        xpl = explainer.explain(x=x, preds=preds, targets=y)
        explanations = torch.cat((explanations, xpl), dim=0)
        i = i + 1
        if i == num_batches_per_file:
            i = 0
            torch.save(explanations, os.path.join(save_dir, f"{explainer_cls.name}_{j}"))
            explanations = torch.empty((0, len(train)), device=device)
            print(f"Finished file {j}")
            j = j + 1
            print(f"Starting file {j}")

    if not i == 0:
        torch.save(explanations, os.path.join(save_dir, f"{explainer_cls.name}_{j}"))
        print(f"Finished file {j}")

    return explanations


def xplain_to_compute_time(model, train, test, device, explainer_clses, kwargses,
                           save_dirs, err, start_page, num_pages, page_size, skip=0):
    explainers = [explainer_clses[i](model=model, dataset=train, device=device, **(kwargses[i])) for i in
                  range(len(kwargses))]
    test_ld = DataLoader(test, batch_size=1, shuffle=False)
    i = 0
    for page in range(num_pages):
        xpl_timeses = [[] for t in range(len(explainers))]
        reses = [{} for t in range(len(explainers))]
        xpl_instance_ids = [skip + start_page * page_size + page * page_size + i for i in range(page_size)]
        selection_tensor = torch.zeros(len(test_ld))
        selection_tensor[xpl_instance_ids] = 1.
        iter_loader = itertools.compress(test_ld, selection_tensor)
        for u, (x, y) in enumerate(iter_loader):
            with torch.no_grad():
                x = x.to(device)
                y = y.to(device)
                preds = torch.argmax(model(x), dim=1)
            for t in range(len(explainers)):
                explainer = explainers[t]
                save_dir = save_dirs[t]
                if u == 0:
                    train_time = explainer.train()
                    reses[t] = {'training': train_time}
                print(f"Starting page {page} with page_size={page_size}")
                last_time = time()
                xpl = explainer.explain(x=x, preds=preds, targets=y)
                xpl_timeses[t].append(time() - last_time)
                # print(xpl_times)
                # explanations=torch.cat((explanations, xpl), dim=0)
                i = i + 1
                reses[t]["xpl"] = xpl_timeses[t]
                with open(os.path.join(save_dir, f"resources_page_{page}"), 'w', encoding='utf-8') as f:
                    json.dump(reses[t], f, ensure_ascii=False, indent=4)
            print(f"Ended page {page} with page_size={page_size}")


class Metric(ABC):
    name = "BaseMetricClass"

    @abstractmethod
    def __init__(self, train, test):
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_result(self, dir):
        pass

    def write_result(self, resdict, dir, file_name):
        with open(f"{dir}/{file_name}", 'w', encoding='utf-8') as f:
            json.dump(self.to_float(resdict), f, ensure_ascii=False, indent=4)
        print(resdict)

    @staticmethod
    def to_float(results):
        if isinstance(results, dict):
            return {key: Metric.to_float(r) for key, r in results.items()}
        elif isinstance(results, str):
            return results
        else:
            return np.array(results).astype(float).tolist()
