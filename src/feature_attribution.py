import argparse
import matplotlib.pyplot as plt
import torch
from skimage import feature, filters
import torchvision.models.resnet
from torchvision.transforms.functional import gaussian_blur
import zennit
from matplotlib.gridspec import GridSpec
from utils import zennit_inner_product_explanation
from utils.data import load_datasets
from utils.models import compute_accuracy, load_model, load_cifar_model
import yaml
import logging
from tqdm import tqdm
import os
from metrics import *
from zennit.image import imgify
from zennit.composites import EpsilonPlus, EpsilonAlpha2Beta1
from zennit.attribution import Gradient
from models.resnet import BasicBlock,Bottleneck
from zennit import canonizers as zcanon
from zennit.canonizers import AttributeCanonizer
from zennit.torchvision import SequentialMergeBatchNorm,Sum

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

class CIFARResNetCanonizer(zcanon.CompositeCanonizer):
    '''Canonizer for torchvision.models.resnet* type models. This applies SequentialMergeBatchNorm, as well as
    add a Sum module to the Bottleneck modules and overload their forward method to use the Sum module instead of
    simply adding two tensors, such that forward and backward hooks may be applied.'''
    def __init__(self):
        super().__init__((
            SequentialMergeBatchNorm(),
            CIFARResNetBottleneckCanonizer(),
            CIFARResNetBasicBlockCanonizer(),
        ))

def evaluate(model_name, model_path, device, class_groups,
             dataset_name, dataset_type,
             data_root, xpl_root,
             save_dir, validation_size, num_classes,imagenet_class_ids, testsplit, pages):
    if not torch.cuda.is_available():
        device="cpu"
    ds_kwargs = {
        'data_root': data_root,
        'class_groups': class_groups,
        'image_set': "test",
        'validation_size': validation_size,
        'only_train': False,
        'imagenet_class_ids':imagenet_class_ids,
        'testsplit':testsplit
    }
    train, test = load_datasets(dataset_name, dataset_type, **ds_kwargs)
    canonizer=None
    if dataset_name == "CIFAR":
        model = load_cifar_model(model_path, dataset_type, num_classes, device)
        canonizer=CIFARResNetCanonizer()
    else:
        model = load_model(model_name, dataset_name, num_classes).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    page_size=4
    offset=7

    fname_root=f"{dataset_name}_{dataset_type}_{xpl_root.split('/')[-1]}"
    acc, all_indices = compute_accuracy(model, test, device=device)
    all_indices=all_indices[:100]
    file_list = [f for f in os.listdir(xpl_root) if ("tgz" not in f) and ("csv" not in f) and ("coefs" not in f) and ("tensor" not in f) and (".shark" not in f)]
    file_root = file_list[0].split('_')[0]
    cur_index = 0
    cumul_xpl = torch.empty(0, len(train), dtype=torch.float32)
    for i in range(len(file_list)):
        file_name = os.path.join(xpl_root, f"{file_root}_{i}")
        xpl = torch.load(file_name, map_location=torch.device("cpu"))
        xpl = (1 / xpl.abs().max(dim=-1)[0][:, None]) * xpl
        len_xpl = xpl.shape[0]
        for j in range(len_xpl):
            if cur_index + j in all_indices:
                cumul_xpl = torch.concat((cumul_xpl, xpl[None, j]))
        cur_index = cur_index + len_xpl
    T=4
    for p in range(pages):
        start_ind=offset+p*page_size
        indices=all_indices[start_ind:start_ind+page_size]
        #indices=[indices[4]]
        #cumul_xpl=cumul_xpl[[4]]
        fname=f"{fname_root}_{p}"
        if dataset_name=="MNIST":
            generate_all_pixel_attributions_MNIST(model, train, test, cumul_xpl[start_ind:start_ind+T], indices, save_dir, fname, T, device, composite='EpsilonPlus', canonizer=canonizer, modes=["heatmap"], start_ind=start_ind-offset, include_test=True)
        else:
            generate_all_pixel_attributions(model, train, test, cumul_xpl[start_ind:start_ind+T], indices, save_dir, fname, T, device, composite='EpsilonPlus', canonizer=canonizer, modes=["heatmap"], start_ind=start_ind-offset, include_test=True)



def mix(x,xpl):
    return .5*(x+xpl)


def max_norm(rel, stabilize=1e-10):
    return rel / (rel.max() + stabilize)

@torch.no_grad()
def vis_opaque_img(data_batch, heatmaps, alpha=0.3, vis_th=0.2, crop_th=0.1, kernel_size=19):
    """
    Draws reference images. The function lowers the opacity in regions with relevance lower than max(relevance)*vis_th.
    In addition, the reference image can be cropped where relevance is less than max(relevance)*crop_th by setting 'rf' to True.

    Parameters:
    ----------
    data_batch: torch.Tensor
        original images from dataset without FeatureVisualization.preprocess() applied to it
    heatmaps: torch.Tensor
        ouput heatmap tensor of the CondAttribution call
    rf: boolean
        Computes the CRP heatmap for a single neuron and hence restricts the heatmap to the receptive field.
        The amount of cropping is further specified by the 'crop_th' argument.
    alpha: between [0 and 1]
        Regulates the transparency in low relevance regions.
    vis_th: between [0 and 1)
        Visualization Threshold: Increases transparency in regions where relevance is smaller than max(relevance)*vis_th.
    crop_th: between [0 and 1)
        Cropping Threshold: Crops the image in regions where relevance is smaller than max(relevance)*crop_th. 
        Cropping is only applied, if receptive field 'rf' is set to True.
    kernel_size: scalar
        Parameter of the torchvision.transforms.functional.gaussian_blur function used to smooth the CRP heatmap.

    Returns:
    --------
    image: list of PIL.Image objects
        If 'rf' is True, reference images have different shapes.

    """

    if alpha > 1 or alpha < 0:
        raise ValueError("'alpha' must be between [0, 1]")
    if vis_th >= 1 or vis_th < 0:
        raise ValueError("'vis_th' must be between [0, 1)")
    if crop_th >= 1 or crop_th < 0:
        raise ValueError("'crop_th' must be between [0, 1)")

    imgs = []
    for i in range(len(data_batch)):

        img = data_batch[i]

        filtered_heat = max_norm(gaussian_blur(heatmaps[i].unsqueeze(0), kernel_size=kernel_size)[0])
        vis_mask = filtered_heat > vis_th


        inv_mask = ~vis_mask
        img = img * vis_mask + torch.ones_like(img) * inv_mask * 0.33
        img = zennit.image.imgify(img.detach().cpu())

        imgs.append(img)

    return imgs


def generate_sample_explanations(model, train, test, cumul_xpl, indices, save_dir, fname, T, device, start_ind):
    x, y = test[0]
    samples = torch.empty((0, x.shape[0], x.shape[1], x.shape[2]))
    labels = torch.empty((0,), dtype=torch.int)
    #N = 5  # number of explanations
    #T = 5  # number of train samples
    for ind in indices:
        x, y = test[ind]
        if isinstance(y,tuple):
            y=y[1]
        y = torch.tensor([y], dtype=torch.int)
        samples = torch.cat((samples, torch.unsqueeze(x, dim=0)), dim=0)
        labels = torch.cat((labels, y), dim=0)
    with torch.no_grad():
        preds = torch.argmax(model(samples.to(device)), dim=1)

    N = len(indices)
    fig = plt.figure(figsize=((T + 2) * 1.5, 3.5 * N))
    fig.tight_layout()
    gs = GridSpec(nrows=N * 2, ncols=T + 2)
    # ax0 = fig.add_subplot(gs[0, 0])
    # ax0.plot(time, height)
    # ax1 = fig.add_subplot(gs[1, 0])
    # ax1.plot(time, weight)
    # ax2 = fig.add_subplot(gs[:, 1])
    # ax2.plot(time, score)
    # ax3 = fig.add_axes([0.6, 0.6, 0.2, 0.2])
    # ax3.hist(distribution)

    if samples.shape[1]==1:
        samples = torch.transpose(samples, 1, 2)
        samples = torch.transpose(samples, 2, 3)
    for i in range(N):
        influence_sort_ids = torch.argsort(cumul_xpl[i])
        ax = fig.add_subplot(gs[2 * i:2 * i + 2, 0:2])
        ax.set_title(f'Prediction:{train.class_labels[preds[i]]}')
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,
            left=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False,
        )
        sample_img=train.inverse_transform(samples[i])
        if samples.shape[-1]==1:
            sample_img=torch.concat((sample_img,sample_img,sample_img),dim=-1)
        sample_img=torch.clip(sample_img, min=0., max=1.)
        #assert sample_img.min() >= 0. and sample_img.max() <= 1.
        if sample_img.shape[0]==3:
            sample_img=sample_img.transpose(0,1)
            sample_img=sample_img.transpose(1,2)
        ax.imshow(sample_img)

        ax.set_ylabel(f'Test image {start_ind+i+1}: {train.class_labels[labels[i]]}')
        for j in range(T):
            x, y = train[influence_sort_ids[j]]
            x = train.inverse_transform(x)
            x = torch.transpose(x, 0, 1)
            x = torch.transpose(x, 1, 2)
            if x.shape[-1]==1:
                x=torch.concat((x,x,x),dim=-1)

            ax = fig.add_subplot(gs[2 * i, j + 2])
            # ax.set_ylabel('Negative Influence')
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            # axes[1 + 3 * i, 0].set_ylabel('Positive Influence')
            if j == T - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel('Negative')
            x=torch.clip(x,min=0.,max=1.)
            ax.imshow(x)
            ax.set_xlabel(f"{cumul_xpl[i, influence_sort_ids[j]]:.2f},  {train.class_labels[y]}")
            x, y = train[influence_sort_ids[-(j + 1)]]
            x = train.inverse_transform(x)
            if x.shape[0]==1:
                x = torch.transpose(x, 0, 1)
                x = torch.transpose(x, 1, 2)
                x=torch.concat((x,x,x),dim=-1)

            ax = fig.add_subplot(gs[2 * i + 1, j + 2])
            if j == T - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel('Positive')
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            x=torch.clip(x,min=0.,max=1.)
            assert x.min() >= 0. and x.max() <= 1.0
            if x.shape[0]==3:
                x=x.transpose(0,1)
                x=x.transpose(1,2)
            ax.imshow(x)
            ax.set_xlabel(f"{cumul_xpl[i, influence_sort_ids[-(j + 1)]]:.2f},  {train.class_labels[y]}")

    #fig.suptitle(fname, fontsize=20)
    fname = fname + ".png"
    fig.savefig(os.path.join(save_dir, fname))

def generate_sample_pixel_attributions(model, train, test, cumul_xpl, indices, save_dir, fname, T, device, composite, canonizer, modes=["overlay"], xpl_test=False,start_ind=0):
    composite_cls_dict={
        'EpsilonPlus':EpsilonPlus,
        'EpsilonAlpha2Beta1': EpsilonAlpha2Beta1
    }
    composite_cls=composite_cls_dict[composite]
    #acc, indices = compute_accuracy(model, test, device=device)
    #indices=indices[:min(len(indices),100)]
    xpl_mode = "test" if xpl_test else "train"
    #file_list = [f for f in os.listdir(xpl_root) if ("tgz" not in f) and ("csv" not in f) and ("coefs" not in f) and (".shark" not in f)]
    #file_root = file_list[0].split('_')[0]
    #cur_index = 0
    #cumul_xpl = torch.empty(0, len(train), dtype=torch.float32)
    #for i in tqdm(range(len(file_list))):
    #    file_name = os.path.join(xpl_root, f"{file_root}_{i}")
    #    xpl = torch.load(file_name, map_location=torch.device("cpu"))
    #    for j in range(xpl.shape[0]):
    #        if cur_index + j in indices:
    #            cumul_xpl = torch.concat((cumul_xpl, xpl[None, j]))
    #    len_xpl = xpl.shape[0]
    #    cur_index = cur_index + len_xpl
    x, y = test[0]
    samples = torch.empty((0, x.shape[0], x.shape[1], x.shape[2]))
    labels = torch.empty((0,), dtype=torch.int)
    #N = 10  # number of explanations
    #T = 5  # number of train samples
    N=len(indices)
    for ind in indices:
        x, y = test[ind]
        if isinstance(y,tuple):
            y=y[1]
        y = torch.tensor([y], dtype=torch.int)
        samples = torch.cat((samples, torch.unsqueeze(x, dim=0)), dim=0)
        labels = torch.cat((labels, y), dim=0)
    with torch.no_grad():
        preds = torch.argmax(model(samples.to(device)), dim=1)

    raw_samples=samples
    if samples.shape[1]==1:
        samples = torch.transpose(samples, 1, 2)
        samples = torch.transpose(samples, 2, 3)
    for mode in modes:
        fig = plt.figure(figsize=((T + 2) * 1.5, 3.5 * N))
        fig.tight_layout()
        gs = GridSpec(nrows=N * 2, ncols=T + 2)
        # ax0 = fig.add_subplot(gs[0, 0])
        # ax0.plot(time, height)
        # ax1 = fig.add_subplot(gs[1, 0])
        # ax1.plot(time, weight)
        # ax2 = fig.add_subplot(gs[:, 1])
        # ax2.plot(time, score)
        # ax3 = fig.add_axes([0.6, 0.6, 0.2, 0.2])
        # ax3.hist(distribution)
        for i in range(N):
            influence_sort_ids = torch.argsort(cumul_xpl[i])
            ax = fig.add_subplot(gs[2 * i:2 * i + 2, 0:2])
            ax.set_title(f'Prediction:{train.class_labels[preds[i]]}')
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            sample_img = train.inverse_transform(samples[i])
            if samples.shape[-1] == 1:
                sample_img = torch.concat((sample_img, sample_img, sample_img), dim=-1)
            sample_img = torch.clip(sample_img, min=0., max=1.)
            # assert sample_img.min() >= 0. and sample_img.max() <= 1.
            if sample_img.shape[0] == 3:
                sample_img = sample_img.transpose(0, 1)
                sample_img = sample_img.transpose(1, 2)

            ax.imshow(sample_img, cmap="jet")

            ax.set_ylabel(f'Test image {start_ind+i+1}: {train.class_labels[labels[i]]}')
            for j in range(T):
                if mode=="positive":
                    raw_x, y = train[influence_sort_ids[-(j + 1)]]
                else:
                    raw_x, y = train[influence_sort_ids[j]]
                x = train.inverse_transform(raw_x)
                x = torch.transpose(x, 0, 1)
                x = torch.transpose(x, 1, 2)
                if x.shape[-1] == 1:
                    x = torch.concat((x, x, x), dim=-1)

                ax = fig.add_subplot(gs[2 * i, j + 2])
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,
                    left=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                if j == T - 1:
                    if mode=="positive":
                        ax.yaxis.set_label_position("right")
                        ax.set_ylabel('Positive')
                    else:
                        ax.yaxis.set_label_position("right")
                        ax.set_ylabel('Negative')

                if mode=="positive":
                    ax.set_xlabel(f"{cumul_xpl[i, influence_sort_ids[-(j+1)]]:.2f},  {train.class_labels[y]}")
                else:
                    ax.set_xlabel(f"{cumul_xpl[i, influence_sort_ids[j]]:.2f},  {train.class_labels[y]}")
                raw_x=raw_x.to(device)
                raw_samples=raw_samples.to(device)

                if mode=="overlay":
                    xpl,_ = zennit_inner_product_explanation(model=model, train=raw_x[None, ...],
                                                           test=raw_samples[i:i + 1, ...], composite_cls=composite_cls, canonizer=canonizer, mode="train")
                    x=mix(x,xpl)
                x=torch.clip(x,min=0.,max=1.)
                ax.imshow(x)
                ax = fig.add_subplot(gs[2 * i + 1, j + 2])
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,
                    left=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                if not mode=="overlay":
                    xpl,_ = zennit_inner_product_explanation(model=model, train=raw_x[None, ...],
                                                           test=raw_samples[i:i + 1, ...], composite_cls=composite_cls, canonizer=canonizer, mode=xpl_mode)
                    xpl=torch.clip(xpl,min=0.,max=1.)
                    ax.imshow(xpl, cmap="jet")
                else:
                    raw_x, y = train[influence_sort_ids[-(j + 1)]]
                    raw_x=raw_x.to(device)
                    raw_samples=raw_samples.to(device)
                    x = train.inverse_transform(raw_x)
                    x = torch.transpose(x, 0, 1)
                    x = torch.transpose(x, 1, 2)
                    if j == T - 1:
                        ax.yaxis.set_label_position("right")
                        ax.set_ylabel('Positive')

                    ax.set_xlabel(f"{cumul_xpl[i, influence_sort_ids[-(j+1)]]:.2f},  {train.class_labels[y]}")
                    xpl,_ = zennit_inner_product_explanation(model=model, train=raw_x[None, ...],
                                                           test=raw_samples[i:i + 1, ...], mode="train")
                    xpl=xpl.to(device)
                    xpl=torch.clip(xpl,min=0.,max=1.)
                    mix_img=mix(x,xpl).to("cpu")
                    ax.imshow(mix_img, cmap="jet")

        method=""
        if "mcsvm" in fname:
            method="Dual Variables"
        elif "representer" in fname:
            method="Representer Points"
        elif "similarity" in fname:
            method="Similarity Score"
        elif "influence" in fname:
            method="Influence Functions"

        dstype=""
        if "std" in fname:
            dstype="Vanilla"
        elif "mark" in fname:
            dstype= "Watermarked"
        elif "corrupt" in fname:
            dstype = "Corrupted Labels"
        elif "group" in fname:
            dstype= "Superlabels"
        title=f"Misclassified images in {dstype} type dataset with {method} method"
        #fig.suptitle(title, fontsize=20)
        fig.savefig(os.path.join(save_dir, f"{fname}_{composite}_{mode}{'_test' if xpl_test else ''}.png"))

#in below function, modes will be "heatmap" or "img"
def generate_all_pixel_attributions(model, train, test, cumul_xpl, indices, save_dir, fname, T, device, composite, canonizer, modes=["heatmap"],include_test=False,start_ind=0):
    alpha=0.2
    vis_th=0.35
    fontsize=12
    composite_cls_dict={
        'EpsilonPlus':EpsilonPlus,
        'EpsilonAlpha2Beta1': EpsilonAlpha2Beta1
    }
    composite_cls=composite_cls_dict[composite]
    #acc, indices = compute_accuracy(model, test, device=device)
    #indices=indices[:min(len(indices),100)]
    #file_list = [f for f in os.listdir(xpl_root) if ("tgz" not in f) and ("csv" not in f) and ("coefs" not in f) and (".shark" not in f)]
    #file_root = file_list[0].split('_')[0]
    #cur_index = 0
    #cumul_xpl = torch.empty(0, len(train), dtype=torch.float32)
    #for i in tqdm(range(len(file_list))):
    #    file_name = os.path.join(xpl_root, f"{file_root}_{i}")
    #    xpl = torch.load(file_name, map_location=torch.device("cpu"))
    #    for j in range(xpl.shape[0]):
    #        if cur_index + j in indices:
    #            cumul_xpl = torch.concat((cumul_xpl, xpl[None, j]))
    #    len_xpl = xpl.shape[0]
    #    cur_index = cur_index + len_xpl
    x, y = test[0]
    samples = torch.empty((0, x.shape[0], x.shape[1], x.shape[2]))
    labels = torch.empty((0,), dtype=torch.int)
    #N = 10  # number of explanations
    #T = 5  # number of train samples
    N=len(indices)
    for ind in indices:
        x, y = test[ind]
        if isinstance(y,tuple):
            y=y[1]
        y = torch.tensor([y], dtype=torch.int)
        samples = torch.cat((samples, torch.unsqueeze(x, dim=0)), dim=0)
        labels = torch.cat((labels, y), dim=0)
    with torch.no_grad():
        preds = torch.argmax(model(samples.to(device)), dim=1)

    raw_samples=samples
    if samples.shape[1]==1:
        samples = torch.transpose(samples, 1, 2)
        samples = torch.transpose(samples, 2, 3)
    for mode in modes:
        if include_test:
            rows_per_img=3
            fig = plt.figure(figsize=((T + 2) * 1.5, 5 * N))
        else:
            rows_per_img=2
            fig = plt.figure(figsize=((T + 2) * 1.5, 3.5 * N))
        fig.tight_layout()

        gs = GridSpec(nrows=N * rows_per_img, ncols=T + 2)
        # ax0 = fig.add_subplot(gs[0, 0])
        # ax0.plot(time, height)
        # ax1 = fig.add_subplot(gs[1, 0])
        # ax1.plot(time, weight)
        # ax2 = fig.add_subplot(gs[:, 1])
        # ax2.plot(time, score)
        # ax3 = fig.add_axes([0.6, 0.6, 0.2, 0.2])
        # ax3.hist(distribution)
        for i in range(N):
            influence_sort_ids = torch.argsort(cumul_xpl[i])
            ax = fig.add_subplot(gs[rows_per_img *i:rows_per_img *i + 2, 0:2])
            ax.set_title(f'Prediction:{train.class_labels[preds[i]]}',fontdict={"size":fontsize})
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            sample_img = train.inverse_transform(samples[i])
            if samples.shape[-1] == 1:
                sample_img = torch.concat((sample_img, sample_img, sample_img), dim=-1)
            sample_img = torch.clip(sample_img, min=0., max=1.)
            # assert sample_img.min() >= 0. and sample_img.max() <= 1.
            if sample_img.shape[0] == 3:
                sample_img = sample_img.transpose(0, 1)
                sample_img = sample_img.transpose(1, 2)
            ax.imshow(1-sample_img)

            ax.set_ylabel(f'Test image: {train.class_labels[labels[i]]}',fontdict={"size":fontsize})
            for j in range(T):
                raw_x, y = train[influence_sort_ids[-(j + 1)]]
                x = train.inverse_transform(raw_x)
                #if x.shape[0]==1:
                #    x=1-x
                x = torch.transpose(x, 0, 1)
                x = torch.transpose(x, 1, 2)
                if x.shape[-1] == 1:
                    x = torch.concat((x, x, x), dim=-1)

                ax = fig.add_subplot(gs[rows_per_img *i, j + 2])
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,
                    left=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False,
                )

                ax.set_xlabel(f"{cumul_xpl[i, influence_sort_ids[-(j+1)]]:.2f},  {train.class_labels[y]}",fontdict={"size":fontsize})
                raw_x=raw_x.to(device)
                raw_samples=raw_samples.to(device)
                x=torch.clip(x,min=0.,max=1.)
                ax.imshow(1-x)
                ax = fig.add_subplot(gs[rows_per_img *i + 1, j + 2])
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,
                    left=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                if j == T - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel('Training\nHeatmap',fontdict={"size":fontsize})
                xpl,attr = zennit_inner_product_explanation(model=model, train=raw_x[None, ...],
                                                         test=raw_samples[i:i + 1, ...], composite_cls=composite_cls, canonizer=canonizer, mode="train")
                #xpl=torch.clip(xpl,min=0.,max=1.)
                if mode=="heatmap":
                    ax.imshow(xpl)
                else:
                    train_x = torch.transpose(x, 1, 2)
                    train_x = torch.transpose(train_x, 0, 1)
                    img = vis_opaque_img(train_x[None], attr[None].to(device), alpha=alpha, vis_th=vis_th)[0]
                    ax.imshow(img)
                if include_test:
                    ax = fig.add_subplot(gs[3 * i + 2, j + 2])
                    ax.tick_params(
                        axis='both',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom=False,
                        left=False,  # ticks along the bottom edge are off
                        top=False,  # ticks along the top edge are off
                        labelbottom=False,
                        labelleft=False,
                    )
                    if j == T - 1:
                        ax.yaxis.set_label_position("right")
                        ax.set_ylabel('Test\nHeatmap',fontdict={"size":fontsize})
                    xpl, attr = zennit_inner_product_explanation(model=model, train=raw_x[None, ...],
                                                              test=raw_samples[i:i + 1, ...], composite_cls=composite_cls,
                                                              canonizer=canonizer, mode="test")

                    if mode=="heatmap":
                        xpl = torch.clip(xpl, min=0., max=1.)
                        ax.imshow(xpl)
                    else:
                        test_x = train.inverse_transform(raw_samples[i:i + 1, ...])
                        if test_x.shape[1] == 1:
                            test_x = torch.concat((test_x, test_x, test_x), dim=1)
                        img = vis_opaque_img(test_x, attr.to(device), alpha=alpha, vis_th=vis_th)[0]
                        ax.imshow(img)

        method=""
        if "mcsvm" in fname:
            method="Dual Variables"
        elif "representer" in fname:
            method="Representer Points"
        elif "similarity" in fname:
            method="Similarity Score"
        elif "influence" in fname:
            method="Influence Functions"

        dstype=""
        if "std" in fname:
            dstype="Vanilla"
        elif "mark" in fname:
            dstype= "Watermarked"
        elif "corrupt" in fname:
            dstype = "Corrupted Labels"
        elif "group" in fname:
            dstype= "Superlabels"
        fig.savefig(os.path.join(save_dir, f"{fname}_{composite}_all_{mode}.pdf"))
        os.system(f"pdfcrop {os.path.join(save_dir, f'{fname}_{composite}_all_{mode}.pdf')}")
        os.system(f"rm {os.path.join(save_dir, f'{fname}_{composite}_all_{mode}.pdf')}")
        os.system(f"mv {os.path.join(save_dir, f'{fname}_{composite}_all_{mode}-crop.pdf')} {os.path.join(save_dir, f'{fname}_{composite}_all_{mode}.pdf')}")
def upscale(img,factor=4):
    newimg=torch.zeros((factor*img.shape[0],factor*img.shape[1],3))
    for i in range(28):
        for j in range(28):
                newimg[i*factor:(i+1)*factor,j*factor:(j+1)*factor]=img[i,j]
    return newimg

def edge_det(img):
    edges= feature.canny(img.sum(-1).numpy(),sigma=3.)
    edges=torch.tensor(edges)
    croppixels=3
    edges[img.shape[0]-croppixels:,:]=0.
    edges[0:croppixels,:]=0.
    edges[:,img.shape[1]-croppixels:]=0.
    edges[:,0:croppixels]=0.
    return edges
def generate_all_pixel_attributions_MNIST(model, train, test, cumul_xpl, indices, save_dir, fname, T, device, composite, canonizer, modes=["heatmap"],include_test=False,start_ind=0):
    alpha=0.2
    vis_th=0.35
    fontsize=12
    composite_cls_dict={
        'EpsilonPlus':EpsilonPlus,
        'EpsilonAlpha2Beta1': EpsilonAlpha2Beta1
    }
    composite_cls=composite_cls_dict[composite]
    #acc, indices = compute_accuracy(model, test, device=device)
    #indices=indices[:min(len(indices),100)]
    #file_list = [f for f in os.listdir(xpl_root) if ("tgz" not in f) and ("csv" not in f) and ("coefs" not in f) and (".shark" not in f)]
    #file_root = file_list[0].split('_')[0]
    #cur_index = 0
    #cumul_xpl = torch.empty(0, len(train), dtype=torch.float32)
    #for i in tqdm(range(len(file_list))):
    #    file_name = os.path.join(xpl_root, f"{file_root}_{i}")
    #    xpl = torch.load(file_name, map_location=torch.device("cpu"))
    #    for j in range(xpl.shape[0]):
    #        if cur_index + j in indices:
    #            cumul_xpl = torch.concat((cumul_xpl, xpl[None, j]))
    #    len_xpl = xpl.shape[0]
    #    cur_index = cur_index + len_xpl
    x, y = test[0]
    samples = torch.empty((0, x.shape[0], x.shape[1], x.shape[2]))
    labels = torch.empty((0,), dtype=torch.int)
    #N = 10  # number of explanations
    #T = 5  # number of train samples
    N=len(indices)
    for ind in indices:
        x, y = test[ind]
        if isinstance(y,tuple):
            y=y[1]
        y = torch.tensor([y], dtype=torch.int)
        samples = torch.cat((samples, torch.unsqueeze(x, dim=0)), dim=0)
        labels = torch.cat((labels, y), dim=0)
    with torch.no_grad():
        preds = torch.argmax(model(samples.to(device)), dim=1)

    raw_samples=samples
    if samples.shape[1]==1:
        samples = torch.transpose(samples, 1, 2)
        samples = torch.transpose(samples, 2, 3)
    for mode in modes:
        if include_test:
            rows_per_img=3
            fig = plt.figure(figsize=((T + 2) * 1.5, 5 * N))
        else:
            rows_per_img=2
            fig = plt.figure(figsize=((T + 2) * 1.5, 3.5 * N))
        fig.tight_layout()

        gs = GridSpec(nrows=N * rows_per_img, ncols=T + 2)
        # ax0 = fig.add_subplot(gs[0, 0])
        # ax0.plot(time, height)
        # ax1 = fig.add_subplot(gs[1, 0])
        # ax1.plot(time, weight)
        # ax2 = fig.add_subplot(gs[:, 1])
        # ax2.plot(time, score)
        # ax3 = fig.add_axes([0.6, 0.6, 0.2, 0.2])
        # ax3.hist(distribution)
        for i in range(N):
            influence_sort_ids = torch.argsort(cumul_xpl[i])
            ax = fig.add_subplot(gs[rows_per_img *i:rows_per_img *i + 2, 0:2])
            ax.set_title(f'Prediction:{train.class_labels[preds[i]]}',fontdict={"size":fontsize})
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            sample_img = train.inverse_transform(samples[i])
            if samples.shape[-1] == 1:
                sample_img = torch.concat((sample_img, sample_img, sample_img), dim=-1)
            sample_img = torch.clip(sample_img, min=0., max=1.)
            # assert sample_img.min() >= 0. and sample_img.max() <= 1.
            if sample_img.shape[0] == 3:
                sample_img = sample_img.transpose(0, 1)
                sample_img = sample_img.transpose(1, 2)
            testoutline=edge_det(upscale(1-sample_img))
            ax.imshow(1-sample_img)

            ax.set_ylabel(f'Test image: {train.class_labels[labels[i]]}',fontdict={"size":fontsize})
            for j in range(T):
                raw_x, y = train[influence_sort_ids[-(j + 1)]]
                x = train.inverse_transform(raw_x)
                #if x.shape[0]==1:
                #    x=1-x
                x = torch.transpose(x, 0, 1)
                x = torch.transpose(x, 1, 2)
                if x.shape[-1] == 1:
                    x = torch.concat((x, x, x), dim=-1)

                ax = fig.add_subplot(gs[rows_per_img *i, j + 2])
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,
                    left=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False,
                )

                ax.set_xlabel(f"{cumul_xpl[i, influence_sort_ids[-(j+1)]]:.2f},  {train.class_labels[y]}",fontdict={"size":fontsize})
                raw_x=raw_x.to(device)
                raw_samples=raw_samples.to(device)
                x=torch.clip(x,min=0.,max=1.)
                outline=edge_det(upscale(1-x))
                ax.imshow(1-x)
                ax = fig.add_subplot(gs[rows_per_img *i + 1, j + 2])
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,
                    left=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                if j == T - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel('Training\nHeatmap',fontdict={"size":fontsize})
                xpl,attr = zennit_inner_product_explanation(model=model, train=raw_x[None, ...],
                                                         test=raw_samples[i:i + 1, ...], composite_cls=composite_cls, canonizer=canonizer, mode="train")

                #xpl=torch.clip(xpl,min=0.,max=1.)
                if mode=="heatmap":
                    xpl=upscale(xpl)
                    xpl[outline]=torch.tensor([0.,0.,0.])
                    ax.imshow(xpl)
                else:
                    train_x = torch.transpose(x, 1, 2)
                    train_x = torch.transpose(train_x, 0, 1)
                    img = vis_opaque_img(train_x[None], attr[None].to(device), alpha=alpha, vis_th=vis_th)[0]
                    ax.imshow(img)
                if include_test:
                    ax = fig.add_subplot(gs[3 * i + 2, j + 2])
                    ax.tick_params(
                        axis='both',  # changes apply to the x-axis
                        which='both',  # both major and minor ticks are affected
                        bottom=False,
                        left=False,  # ticks along the bottom edge are off
                        top=False,  # ticks along the top edge are off
                        labelbottom=False,
                        labelleft=False,
                    )
                    if j == T - 1:
                        ax.yaxis.set_label_position("right")
                        ax.set_ylabel('Test\nHeatmap',fontdict={"size":fontsize})
                    xpl, attr = zennit_inner_product_explanation(model=model, train=raw_x[None, ...],
                                                              test=raw_samples[i:i + 1, ...], composite_cls=composite_cls,
                                                              canonizer=canonizer, mode="test")

                    if mode=="heatmap":
                        xpl = torch.clip(xpl, min=0., max=1.)
                        xpl=upscale(xpl)
                        xpl[testoutline]=torch.tensor([0.,0.,0.])
                        ax.imshow(xpl)
                    else:
                        test_x = train.inverse_transform(raw_samples[i:i + 1, ...])
                        if test_x.shape[1] == 1:
                            test_x = torch.concat((test_x, test_x, test_x), dim=1)
                        img = vis_opaque_img(test_x, attr.to(device), alpha=alpha, vis_th=vis_th)[0]
                        ax.imshow(img)

        method=""
        if "mcsvm" in fname:
            method="Dual Variables"
        elif "representer" in fname:
            method="Representer Points"
        elif "similarity" in fname:
            method="Similarity Score"
        elif "influence" in fname:
            method="Influence Functions"

        dstype=""
        if "std" in fname:
            dstype="Vanilla"
        elif "mark" in fname:
            dstype= "Watermarked"
        elif "corrupt" in fname:
            dstype = "Corrupted Labels"
        elif "group" in fname:
            dstype= "Superlabels"
        fig.savefig(os.path.join(save_dir, f"{fname}_{composite}_all_{mode}.pdf"))
        os.system(f"pdfcrop {os.path.join(save_dir, f'{fname}_{composite}_all_{mode}.pdf')}")
        os.system(f"rm {os.path.join(save_dir, f'{fname}_{composite}_all_{mode}.pdf')}")
        os.system(f"mv {os.path.join(save_dir, f'{fname}_{composite}_all_{mode}-crop.pdf')} {os.path.join(save_dir, f'{fname}_{composite}_all_{mode}.pdf')}")


if __name__ == "__main__":
    # current = os.path.dirname(os.path.realpath(__file__))
    # parent_directory = os.path.dirname(current)
    # sys.path.append(current)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--pages', type=int)
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, "r") as stream:
        try:
            train_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.info(exc)

    save_dir = f"{train_config['save_dir']}/{os.path.basename(config_file)[:-5]}"

    evaluate(model_name=train_config.get('model_name', None),
             model_path=train_config.get('model_path', None),
             device=train_config.get('device', 'cuda'),
             class_groups=train_config.get('class_groups', None),
             dataset_name=train_config.get('dataset_name', None),
             dataset_type=train_config.get('dataset_type', 'std'),
             data_root=train_config.get('data_root', None),
             xpl_root=train_config.get('xpl_root', None),
             #coef_root=train_config.get('coef_root', None),
             save_dir=train_config.get('save_dir', None),
             validation_size=train_config.get('validation_size', 2000),
             num_classes=train_config.get('num_classes'),
             imagenet_class_ids=train_config.get('imagenet_class_ids', [i for i in range(397)]),
             testsplit=train_config.get('testsplit', "test"),
             pages=args.pages
             )
