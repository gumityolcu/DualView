import argparse
import yaml
from matplotlib.gridspec import GridSpec
from utils.data import load_datasets
from utils.models import compute_accuracy, load_model, load_cifar_model
import logging
from tqdm import tqdm
import os
from metrics import *


def evaluate(model_name, model_path, device, class_groups,
             dataset_name, dataset_type,
             data_root, xpl_roots, method_names,
             save_dir, validation_size, num_classes, imagenet_class_ids, testsplit, pages):
    if not torch.cuda.is_available():
        device = "cpu"
    ds_kwargs = {
        'data_root': data_root,
        'class_groups': class_groups,
        'image_set': "test",
        'validation_size': validation_size,
        'only_train': False,
        'imagenet_class_ids': imagenet_class_ids,
        'testsplit': testsplit
    }
    train, test = load_datasets(dataset_name, dataset_type, **ds_kwargs)
    canonizer = None
    if dataset_name == "CIFAR":
        model = load_cifar_model(model_path, dataset_type, num_classes, device)
    elif "ImageNet" in dataset_name:
        model = load_imagenet_model(device)
        if imagenet_class_ids is not None:
            temp = torch.nn.Linear(in_features=model.classifier.in_features,
                                   out_features=len(imagenet_class_ids), bias=True)
            temp.weight = torch.nn.Parameter(model.classifier.weight[imagenet_class_ids])
            temp.bias = torch.nn.Parameter(model.classifier.bias[imagenet_class_ids])
            model.classifier = temp
    else:
        model = load_model(model_name, dataset_name, num_classes).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    page_size = 5
    offset = 0
    if dataset_name == "CIFAR":
        offset = 5

    fname_roots = [f"{dataset_name}_{dataset_type}_{xpl_root.split('/')[-1]}" for xpl_root in xpl_roots]
    acc, all_indices = compute_accuracy(model, test, device=device)
    all_indices = all_indices[:100]
    if dataset_name == "CIFAR":
        all_indices = [all_indices[i] for i in [12, 13, 15, 16, 18, 25, 30, 36, 39, 47]]
    file_lists = [[f for f in os.listdir(xpl_root) if
                   ("tgz" not in f) and ("csv" not in f) and ("coefs" not in f) and ("tensor" not in f)] for xpl_root in
                  xpl_roots]
    # file_root = file_list[0].split('_')[0]
    cumul_xpl = [torch.empty(0, len(train), dtype=torch.float32) for _ in xpl_roots]
    for i in range(len(cumul_xpl)):
        cur_index = 0
        for j in range(len(file_lists[i])):
            file_name = os.path.join(xpl_roots[i], f"{file_lists[i][0].split('_')[0]}_{j}")
            xpl = torch.load(file_name, map_location=torch.device("cpu"))
            xpl = (1 / xpl.abs().max(dim=-1)[0][:, None]) * xpl
            len_xpl = xpl.shape[0]
            for k in range(len_xpl):
                if cur_index + k in all_indices:
                    cumul_xpl[i] = torch.concat((cumul_xpl[i], xpl[None, k]))
            cur_index = cur_index + len_xpl
    T = 3
    for p in range(pages):
        start_ind = offset + p * page_size
        indices = all_indices[start_ind:start_ind + page_size]
        xpl_tensors = [c[start_ind:start_ind + page_size] for c in cumul_xpl]
        fname = f"xpl_comparison_{p}"
        # generate_comparison_explanations_horizontal_with_spaces(model, train, test, xpl_tensors, method_names, indices, save_dir, f"H_{fname}", T, device, start_ind=start_ind-offset)
        generate_comparison_explanations_horizontal_with_small_spaces(model, train, test, xpl_tensors, method_names,
                                                                      indices, save_dir, f"H_{fname}", T, device,
                                                                      start_ind=start_ind - offset)
        # generate_comparison_explanations_vertical(model, train, test, xpl_tensors, method_names, indices, save_dir, f"V_{fname}", T, device, start_ind=start_ind-offset)
        # composite='EpsilonPlus', modes=["positive","overlay"])


def mix(x, xpl):
    return .5 * (x + xpl)


def generate_comparison_explanations_horizontal(model, train, test, xpl_tensors, method_names, indices, save_dir,
                                                base_fname, T, device, start_ind):
    x, y = test[0]
    samples = torch.empty((0, x.shape[0], x.shape[1], x.shape[2]))
    labels = torch.empty((0,), dtype=torch.int)
    # plt.tight_layout()
    # N = 5  # number of explanations
    # T = 5  # number of train samples
    for ind in indices:
        x, y = test[ind]
        if isinstance(y, tuple):
            y = y[1]
        y = torch.tensor([y], dtype=torch.int)
        samples = torch.cat((samples, torch.unsqueeze(x, dim=0)), dim=0)
        labels = torch.cat((labels, y), dim=0)
    with torch.no_grad():
        preds = torch.argmax(model(samples.to(device)), dim=1)

    N = len(indices)

    # ax0 = fig.add_subplot(gs[0, 0])
    # ax0.plot(time, height)
    # ax1 = fig.add_subplot(gs[1, 0])
    # ax1.plot(time, weight)
    # ax2 = fig.add_subplot(gs[:, 1])
    # ax2.plot(time, score)
    # ax3 = fig.add_axes([0.6, 0.6, 0.2, 0.2])
    # ax3.hist(distribution)

    if samples.shape[1] == 1:
        samples = torch.transpose(samples, 1, 2)
        samples = torch.transpose(samples, 2, 3)
    for i in range(N):
        fig = plt.figure(figsize=(((T) * len(xpl_tensors) + 1) * 1.7, 3.7))
        gs = GridSpec(nrows=2, ncols=len(xpl_tensors) * (T) + 1)
        gs.tight_layout(fig)
        influence_sort_ids = [torch.argsort(cumul_xpl[i]) for cumul_xpl in xpl_tensors]
        ax = fig.add_subplot(gs[0:1, 0:1])
        ax.set_title(f'Pred.: {train.class_labels[preds[i]]}')
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
        ax.imshow(sample_img)
        ax.set_ylabel(f'Label: {train.class_labels[labels[i]]}')
        for k in range(len(xpl_tensors)):
            ax = fig.add_subplot(gs[0:2, 1 + k * (T):T + k * (T) + 1])
            # ax.yaxis.set_label_position("right")
            ax.set_title(method_names[k], fontdict={'size': 16}, )
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            for j in range(T):
                x, y = train[influence_sort_ids[k][j]]
                x = train.inverse_transform(x)
                x = torch.transpose(x, 0, 1)
                x = torch.transpose(x, 1, 2)
                if x.shape[-1] == 1:
                    x = torch.concat((x, x, x), dim=-1)

                ax = fig.add_subplot(gs[0, (T) * k + j + 1])
                if j == T - 1 and k == len(xpl_tensors) - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel('Negative')
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
                # if j == T - 1:
                #    ax.yaxis.set_label_position("right")
                #    ax.set_ylabel('Negative')
                x = torch.clip(x, min=0., max=1.)
                ax.imshow(x)
                ax.set_xlabel(f"{xpl_tensors[k][i, influence_sort_ids[k][j]]:.2f},  {train.class_labels[y]}")
                x, y = train[influence_sort_ids[k][-(j + 1)]]
                x = train.inverse_transform(x)
                if x.shape[0] == 1:
                    x = torch.transpose(x, 0, 1)
                    x = torch.transpose(x, 1, 2)
                    x = torch.concat((x, x, x), dim=-1)

                ax = fig.add_subplot(gs[1, (T) * k + j + 1])
                if j == T - 1 and k == len(xpl_tensors) - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel('Positive')
                # if j == T - 1:
                #    ax.yaxis.set_label_position("right")
                #    ax.set_ylabel('Positive')
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,
                    left=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                x = torch.clip(x, min=0., max=1.)
                assert x.min() >= 0. and x.max() <= 1.0
                if x.shape[0] == 3:
                    x = x.transpose(0, 1)
                    x = x.transpose(1, 2)
                ax.imshow(x)
                ax.set_xlabel(f"{xpl_tensors[k][i, influence_sort_ids[k][-(j + 1)]]:.2f},  {train.class_labels[y]}")
        fname = f"{base_fname}-{i}"

        fig.savefig(f"{os.path.join(save_dir, fname)}.pdf")
        os.system(f"pdfcrop {os.path.join(save_dir, fname)}.pdf")
        os.system(f"rm {os.path.join(save_dir, fname)}.pdf")
        os.system(f"mv {os.path.join(save_dir, fname)}-crop.pdf {os.path.join(save_dir, fname)}.pdf")


def generate_comparison_explanations_horizontal_with_spaces(model, train, test, xpl_tensors, method_names, indices,
                                                            save_dir, base_fname, T, device, start_ind):
    fontsize = 15
    x, y = test[0]
    samples = torch.empty((0, x.shape[0], x.shape[1], x.shape[2]))
    labels = torch.empty((0,), dtype=torch.int)
    # plt.tight_layout()
    # N = 5  # number of explanations
    # T = 5  # number of train samples
    for ind in indices:
        x, y = test[ind]
        if isinstance(y, tuple):
            y = y[1]
        y = torch.tensor([y], dtype=torch.int)
        samples = torch.cat((samples, torch.unsqueeze(x, dim=0)), dim=0)
        labels = torch.cat((labels, y), dim=0)
    with torch.no_grad():
        preds = torch.argmax(model(samples.to(device)), dim=1)

    N = len(indices)

    # ax0 = fig.add_subplot(gs[0, 0])
    # ax0.plot(time, height)
    # ax1 = fig.add_subplot(gs[1, 0])
    # ax1.plot(time, weight)
    # ax2 = fig.add_subplot(gs[:, 1])
    # ax2.plot(time, score)
    # ax3 = fig.add_axes([0.6, 0.6, 0.2, 0.2])
    # ax3.hist(distribution)

    if samples.shape[1] == 1:
        samples = torch.transpose(samples, 1, 2)
        samples = torch.transpose(samples, 2, 3)
    for i in range(N):
        fig = plt.figure(figsize=(((T + 1) * len(xpl_tensors)) * 1.7, 3.7))
        gs = GridSpec(nrows=2, ncols=len(xpl_tensors) * (T + 1))
        gs.tight_layout(fig)
        influence_sort_ids = [torch.argsort(cumul_xpl[i]) for cumul_xpl in xpl_tensors]
        ax = fig.add_subplot(gs[0:1, 0:1])
        ax.set_title(f'Pred.: {train.class_labels[preds[i]]}', fontdict={"size": fontsize})
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
        ax.imshow(sample_img)
        ax.set_ylabel(f'Label: {train.class_labels[labels[i]]}', fontdict={'size': fontsize})
        for k in range(len(xpl_tensors)):
            ax = fig.add_subplot(gs[0:2, 1 + k * (T + 1):T + k * (T + 1) + 1])
            # ax.yaxis.set_label_position("right")
            ax.set_title(method_names[k], fontdict={"size": fontsize})
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            for j in range(T):
                x, y = train[influence_sort_ids[k][j]]
                x = train.inverse_transform(x)
                x = torch.transpose(x, 0, 1)
                x = torch.transpose(x, 1, 2)
                if x.shape[-1] == 1:
                    x = torch.concat((x, x, x), dim=-1)

                ax = fig.add_subplot(gs[0, (T + 1) * k + j + 1])
                if j == T - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel('Negative', fontdict={"size": fontsize})
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
                # if j == T - 1:
                #    ax.yaxis.set_label_position("right")
                #    ax.set_ylabel('Negative')
                x = torch.clip(x, min=0., max=1.)
                ax.imshow(x)
                ax.set_xlabel(f"{xpl_tensors[k][i, influence_sort_ids[k][j]]:.2f},{train.class_labels[y]}",
                              fontdict={"size": fontsize})
                x, y = train[influence_sort_ids[k][-(j + 1)]]
                x = train.inverse_transform(x)
                if x.shape[0] == 1:
                    x = torch.transpose(x, 0, 1)
                    x = torch.transpose(x, 1, 2)
                    x = torch.concat((x, x, x), dim=-1)

                ax = fig.add_subplot(gs[1, (T + 1) * k + j + 1])
                if j == T - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel('Positive', fontdict={"size": fontsize})
                # if j == T - 1:
                #    ax.yaxis.set_label_position("right")
                #    ax.set_ylabel('Positive')
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,
                    left=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                x = torch.clip(x, min=0., max=1.)
                assert x.min() >= 0. and x.max() <= 1.0
                if x.shape[0] == 3:
                    x = x.transpose(0, 1)
                    x = x.transpose(1, 2)
                ax.imshow(x)
                ax.set_xlabel(f"{xpl_tensors[k][i, influence_sort_ids[k][-(j + 1)]]:.2f},  {train.class_labels[y]}",
                              fontdict={"size": fontsize})
        fname = f"{base_fname}-{i}"

        fig.savefig(f"{os.path.join(save_dir, fname)}.pdf")
        os.system(f"pdfcrop {os.path.join(save_dir, fname)}.pdf")
        os.system(f"rm {os.path.join(save_dir, fname)}.pdf")
        os.system(f"mv {os.path.join(save_dir, fname)}-crop.pdf {os.path.join(save_dir, fname)}.pdf")


def generate_comparison_explanations_horizontal_with_small_spaces(model, train, test, xpl_tensors, method_names,
                                                                  indices, save_dir, base_fname, T, device, start_ind):
    fontsize = 12
    buffer = 3
    x, y = test[0]
    samples = torch.empty((0, x.shape[0], x.shape[1], x.shape[2]))
    labels = torch.empty((0,), dtype=torch.int)
    # plt.tight_layout()
    # N = 5  # number of explanations
    # T = 5  # number of train samples
    for ind in indices:
        x, y = test[ind]
        if isinstance(y, tuple):
            y = y[1]
        y = torch.tensor([y], dtype=torch.int)
        samples = torch.cat((samples, torch.unsqueeze(x, dim=0)), dim=0)
        labels = torch.cat((labels, y), dim=0)
    with torch.no_grad():
        preds = torch.argmax(model(samples.to(device)), dim=1)

    N = len(indices)

    # ax0 = fig.add_subplot(gs[0, 0])
    # ax0.plot(time, height)
    # ax1 = fig.add_subplot(gs[1, 0])
    # ax1.plot(time, weight)
    # ax2 = fig.add_subplot(gs[:, 1])
    # ax2.plot(time, score)
    # ax3 = fig.add_axes([0.6, 0.6, 0.2, 0.2])
    # ax3.hist(distribution)
    persquare = 5
    space = 2
    if samples.shape[1] == 1:
        samples = torch.transpose(samples, 1, 2)
        samples = torch.transpose(samples, 2, 3)
    for i in range(N):
        fig = plt.figure(figsize=(((T + 1) * len(xpl_tensors)) * 1.3, 3.7))
        gs = GridSpec(nrows=2 * persquare + buffer,
                      ncols=(persquare + len(xpl_tensors) * T * persquare + (len(xpl_tensors) - 1) * space))
        gs.tight_layout(fig)
        influence_sort_ids = [torch.argsort(cumul_xpl[i]) for cumul_xpl in xpl_tensors]
        ax = fig.add_subplot(gs[0:persquare, 0:persquare])
        ax.set_title(f'Pred.: {train.class_labels[preds[i]]}')
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
        ax.imshow(sample_img)
        ax.set_ylabel(f'Label: {train.class_labels[labels[i]]}')
        for k in range(len(xpl_tensors)):
            ax = fig.add_subplot(gs[0:2 * persquare + buffer,
                                 persquare + k * (T * persquare + space):T * persquare + k * (
                                             T * persquare + space) + persquare])
            # ax.yaxis.set_label_position("right")
            ax.set_title(method_names[k])
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            for j in range(T):
                x, y = train[influence_sort_ids[k][j]]
                x = train.inverse_transform(x)
                x = torch.transpose(x, 0, 1)
                x = torch.transpose(x, 1, 2)
                if x.shape[-1] == 1:
                    x = torch.concat((x, x, x), dim=-1)

                ax = fig.add_subplot(gs[0:persquare, (T * persquare + space) * k + j * persquare + persquare:(
                                                                                                                         T * persquare + space) * k + j * persquare + 2 * persquare])
                if j == T - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel('Negative')
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
                # if j == T - 1:
                #    ax.yaxis.set_label_position("right")
                #    ax.set_ylabel('Negative')
                x = torch.clip(x, min=0., max=1.)
                ax.imshow(x)
                ax.set_xlabel(f"{xpl_tensors[k][i, influence_sort_ids[k][j]]:.2f},  {train.class_labels[y]}")
                x, y = train[influence_sort_ids[k][-(j + 1)]]
                x = train.inverse_transform(x)
                if x.shape[0] == 1:
                    x = torch.transpose(x, 0, 1)
                    x = torch.transpose(x, 1, 2)
                    x = torch.concat((x, x, x), dim=-1)

                ax = fig.add_subplot(gs[persquare + buffer:2 * persquare + buffer,
                                     (T * persquare + space) * k + j * persquare + persquare:(
                                                                                                         T * persquare + space) * k + j * persquare + 2 * persquare])
                if j == T - 1:
                    ax.yaxis.set_label_position("right")
                    ax.set_ylabel('Positive')
                # if j == T - 1:
                #    ax.yaxis.set_label_position("right")
                #    ax.set_ylabel('Positive')
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,
                    left=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                x = torch.clip(x, min=0., max=1.)
                assert x.min() >= 0. and x.max() <= 1.0
                if x.shape[0] == 3:
                    x = x.transpose(0, 1)
                    x = x.transpose(1, 2)
                ax.imshow(x)
                ax.set_xlabel(f"{xpl_tensors[k][i, influence_sort_ids[k][-(j + 1)]]:.2f},  {train.class_labels[y]}")
        fname = f"{base_fname}-{i}"

        fig.savefig(f"{os.path.join(save_dir, fname)}.png")


#        fig.savefig(f"{os.path.join(save_dir, fname)}.pdf")
#        os.system(f"pdfcrop {os.path.join(save_dir, fname)}.pdf")
#        os.system(f"rm {os.path.join(save_dir, fname)}.pdf")
#        os.system(f"mv {os.path.join(save_dir, fname)}-crop.pdf {os.path.join(save_dir, fname)}.pdf")

def generate_comparison_explanations_vertical(model, train, test, xpl_tensors, method_names, indices, save_dir,
                                              base_fname, T, device, start_ind):
    x, y = test[0]
    samples = torch.empty((0, x.shape[0], x.shape[1], x.shape[2]))
    labels = torch.empty((0,), dtype=torch.int)
    # plt.tight_layout()
    # plt.tight_layout()
    # N = 5  # number of explanations
    # T = 5  # number of train samples
    for ind in indices:
        x, y = test[ind]
        if isinstance(y, tuple):
            y = y[1]
        y = torch.tensor([y], dtype=torch.int)
        samples = torch.cat((samples, torch.unsqueeze(x, dim=0)), dim=0)
        labels = torch.cat((labels, y), dim=0)
    with torch.no_grad():
        preds = torch.argmax(model(samples.to(device)), dim=1)

    N = len(indices)

    # ax0 = fig.add_subplot(gs[0, 0])
    # ax0.plot(time, height)
    # ax1 = fig.add_subplot(gs[1, 0])
    # ax1.plot(time, weight)
    # ax2 = fig.add_subplot(gs[:, 1])
    # ax2.plot(time, score)
    # ax3 = fig.add_axes([0.6, 0.6, 0.2, 0.2])
    # ax3.hist(distribution)

    if samples.shape[1] == 1:
        samples = torch.transpose(samples, 1, 2)
        samples = torch.transpose(samples, 2, 3)
    for i in range(N):
        fig = plt.figure(figsize=((T + 1) * 1.7, 3.5 * len(xpl_tensors)))
        gs = GridSpec(nrows=2 * len(xpl_tensors), ncols=T + 1)
        gs.tight_layout(fig)
        influence_sort_ids = [torch.argsort(cumul_xpl[i]) for cumul_xpl in xpl_tensors]
        ax = fig.add_subplot(gs[0:1, 0:1])
        ax.set_title(f'Pred.: {train.class_labels[preds[i]]}')
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
        ax.imshow(sample_img)
        ax.set_ylabel(f'Label: {train.class_labels[labels[i]]}')
        for k in range(len(xpl_tensors)):
            ax = fig.add_subplot(gs[2 * k:2 * k + 2, 1:T + 1])
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(method_names[k], fontdict={'size': 14})
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            for j in range(T):
                x, y = train[influence_sort_ids[k][j]]
                x = train.inverse_transform(x)
                x = torch.transpose(x, 0, 1)
                x = torch.transpose(x, 1, 2)
                if x.shape[-1] == 1:
                    x = torch.concat((x, x, x), dim=-1)

                ax = fig.add_subplot(gs[2 * k, j + 1])
                if j == 0:
                    ax.set_ylabel('Negative')
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
                # if j == T - 1:
                #    ax.yaxis.set_label_position("right")
                #    ax.set_ylabel('Negative')
                x = torch.clip(x, min=0., max=1.)
                ax.imshow(x)
                ax.set_xlabel(f"{xpl_tensors[k][i, influence_sort_ids[k][j]]:.2f},  {train.class_labels[y]}")
                x, y = train[influence_sort_ids[k][-(j + 1)]]
                x = train.inverse_transform(x)
                if x.shape[0] == 1:
                    x = torch.transpose(x, 0, 1)
                    x = torch.transpose(x, 1, 2)
                    x = torch.concat((x, x, x), dim=-1)

                ax = fig.add_subplot(gs[2 * k + 1, j + 1])
                if j == 0:
                    ax.set_ylabel('Positive')
                # if j == T - 1:
                #    ax.yaxis.set_label_position("right")
                #    ax.set_ylabel('Positive')
                ax.tick_params(
                    axis='both',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,
                    left=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False,
                )
                x = torch.clip(x, min=0., max=1.)
                assert x.min() >= 0. and x.max() <= 1.0
                if x.shape[0] == 3:
                    x = x.transpose(0, 1)
                    x = x.transpose(1, 2)
                ax.imshow(x)
                ax.set_xlabel(f"{xpl_tensors[k][i, influence_sort_ids[k][-(j + 1)]]:.2f},  {train.class_labels[y]}")
        fname = f"{base_fname}-{i}"
        fig.savefig(f"{os.path.join(save_dir, fname)}.pdf")
        os.system(f"pdfcrop {os.path.join(save_dir, fname)}.pdf")
        os.system(f"rm {os.path.join(save_dir, fname)}.pdf")
        os.system(f"mv {os.path.join(save_dir, fname)}-crop.pdf {os.path.join(save_dir, fname)}.pdf")


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

    evaluate(
        model_name=train_config.get('model_name', None),
        model_path=train_config.get('model_path', None),
        device=train_config.get('device', 'cuda'),
        class_groups=train_config.get('class_groups', None),
        dataset_name=train_config.get('dataset_name', None),
        dataset_type=train_config.get('dataset_type', 'std'),
        data_root=train_config.get('data_root', None),
        xpl_roots=train_config.get('xpl_roots', None),
        method_names=train_config.get('method_names', None),
        # coef_root=train_config.get('coef_root', None),
        save_dir=train_config.get('save_dir', None),
        validation_size=train_config.get('validation_size', 2000),
        num_classes=train_config.get('num_classes'),
        imagenet_class_ids=train_config.get('imagenet_class_ids', [i for i in range(397)]),
        testsplit=train_config.get('testsplit', "test"),
        pages=args.pages
    )
