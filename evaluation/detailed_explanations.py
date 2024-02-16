import torch
import matplotlib.pyplot as plt
from explainers import DualView
from utils.models import load_imagenet_model
from utils.data import load_datasets_reduced, RestrictedDataset
from utils.models import compute_accuracy
from matplotlib.gridspec import GridSpec
from zennit.composites import EpsilonPlus
from zennit.attribution import Gradient
from utils import zennit_inner_product_explanation, CIFARResNetCanonizer
from crp.image import vis_opaque_img


def compute_indices(model, test, max, device):
    loader = torch.utils.data.DataLoader(test, 32, shuffle=False)
    model.eval()
    index = 0
    fault_list = []
    loader = iter(loader)
    while (len(fault_list) < max):
        x, y = loader.__next__()
        x = x.to(device)
        if isinstance(y, list):
            y = y[1]
        y = y.to(device)
        real_out = torch.argmax(model(x), dim=1)
        preds = (y == real_out)
        racc = torch.sum(preds)
        if racc != 32:
            for j in range(x.size(0)):
                if not preds[j]:
                    fault_list.append(j + index)
        index += 32
    return fault_list


device = "cuda"
dataset_name = "ImageNet1em3"
save_dir = "/home/fe/yolcu/Documents/Code/THESIS/explanations/ImageNet1em3/std/resnet_std/mcsvm"
dataset_type = "std"
data_root = "/home/fe/yolcu/Documents/Code/Datasets/imagenet"
C_margin = 1e-3
alpha_th = 0.0
vis_th = 0.3
#indices = [0]
indices = [0, 1, 6, 7, 8, 9]
model = load_imagenet_model(device)
imagenet_class_ids = range(397)


def generate_with_test(device, dataset_name, save_dir, dataset_type, data_root, C_margin, alpha_th, vis_th, model,
                       imagenet_class_ids, indices):
    ds_kwargs = {
        'data_root': data_root,
        'class_groups': None,
        'image_set': "test",
        'validation_size': 0,
        "only_train": False,
        'imagenet_class_ids': imagenet_class_ids,
        'testsplit': "test"
    }

    train, test = load_datasets_reduced(dataset_name, dataset_type, ds_kwargs)
    plt.rcParams['text.usetex'] = True

    if imagenet_class_ids is not None:
        temp = torch.nn.Linear(in_features=model.classifier.in_features,
                               out_features=len(imagenet_class_ids), bias=True)
        temp.weight = torch.nn.Parameter(model.classifier.weight[imagenet_class_ids])
        temp.bias = torch.nn.Parameter(model.classifier.bias[imagenet_class_ids])
        model.classifier = temp
    model.to(device)
    err = None
    # if accuracy:
    #    acc, err = compute_accuracy(model, test,device)
    #    print(f"Accuracy: {acc}")
    explainer_cls, kwargs = DualView, {"sanity_check": False, "dir": save_dir}
    if C_margin is not None:
        kwargs["C"] = C_margin
    print(f"Generating explanations with {explainer_cls.name}")
    explainer = explainer_cls(model=model, dataset=train, device=device, **kwargs)
    restr_indices = compute_indices(model, test, 10, device)
    # print(f"Accuracy: {acc}")
    test = RestrictedDataset(test, restr_indices)

    explainer.train()
    for i in indices:
        fig = plt.figure(figsize=(7 * 1.5, 3.5 * 6))
        fig.tight_layout()
        gs = GridSpec(nrows=6, ncols=11)
        ax = fig.add_subplot(gs[0:2, 0:2])
        dp, y = test[i]
        img = test.dataset.inverse_transform(dp)
        img = img.transpose(0, 1).transpose(1, 2)
        dp = dp[None, ...].to(device)
        pred = torch.argmax(model(dp))
        ax.set_title(f'Pred.: {test.dataset.class_labels[pred]}')
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,
            left=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False,
        )
        ax.set_ylabel(f'Image {1 + i}: {test.dataset.class_labels[y]}')
        ax.imshow(img.to("cpu"))
        y = torch.tensor(y, device=device)
        pred_xpl = explainer.explain(dp, preds=pred[None, ...], targets=None)
        pred_xpl = pred_xpl / pred_xpl.abs().max()
        pred_indices = torch.argsort(pred_xpl, descending=True)
        target_xpl = explainer.explain(dp, preds=y[None, ...], targets=None)
        target_xpl = target_xpl / target_xpl.abs().max()
        target_indices = torch.argsort(target_xpl, descending=True)
        ax = fig.add_subplot(gs[0:2, 2:])
        ax.set_title(f"Evidence for prediction class {test.dataset.class_labels[pred]}")
        ax.axis("off")
        ax = fig.add_subplot(gs[3:6, 2:])
        ax.set_title(f"Evidence for true class {test.dataset.class_labels[y]}")
        ax.axis("off")
        for j in range(3):
            #### TOP-LEFT FIGURE
            # // TR IMG
            ax = fig.add_subplot(gs[j, 2])
            tr_dp, tr_y = train[pred_indices[j]]
            img = test.dataset.inverse_transform(tr_dp)
            img = img.transpose(0, 1).transpose(1, 2)
            ax.imshow(img.to("cpu"))

            tr_dp = tr_dp[None, ...].to(device)

            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            # // HEATMAP
            heatmap, attr = zennit_inner_product_explanation(model=model, train=tr_dp, test=dp,
                                                             composite_cls=EpsilonPlus,
                                                             canonizer=CIFARResNetCanonizer(), mode="train")
            ax = fig.add_subplot(gs[j, 3])
            ax.set_xlabel(f"{pred_xpl[pred_indices[j]]:.2f}, {test.dataset.class_labels[tr_y]}")
            ax.imshow(heatmap.to("cpu"))
            if j == 0:
                ax.set_title("\\textbf{Positive Evidence}")
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            # // HIGHLIGHT
            ax = fig.add_subplot(gs[j, 4])
            img = vis_opaque_img(tr_dp, attr.to(device), alpha=alpha_th, vis_th=vis_th)[0]
            ax.imshow(img)
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            # // HIGHLIGHT 2
            ax = fig.add_subplot(gs[j, 5])
            heatmap, attr = zennit_inner_product_explanation(model=model, train=tr_dp, test=dp,
                                                             composite_cls=EpsilonPlus,
                                                             canonizer=CIFARResNetCanonizer(), mode="test")
            img = vis_opaque_img(dp, attr.to(device), alpha=alpha_th, vis_th=vis_th)[0]
            ax.imshow(img)
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            #### TOP RIGHT FIGURE
            # //TR IMG
            ax = fig.add_subplot(gs[j, 2 + 4 + 1])
            tr_dp, tr_y = train[pred_indices[-j - 1]]
            img = test.dataset.inverse_transform(tr_dp)
            img = img.transpose(0, 1).transpose(1, 2)
            tr_dp = tr_dp[None, ...]
            tr_dp = tr_dp.to(device)
            tr_y = torch.tensor(tr_y, device=device)

            ax.imshow(img.to("cpu"))
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            # // HEATMAP
            heatmap, attr = zennit_inner_product_explanation(model=model, train=tr_dp, test=dp,
                                                             composite_cls=EpsilonPlus,
                                                             canonizer=CIFARResNetCanonizer(), mode="train")
            ax = fig.add_subplot(gs[j, 2 + 1 + 4 + 1])
            ax.imshow(heatmap.to("cpu"))
            ax.set_xlabel(f"{pred_xpl[pred_indices[-j - 1]]:.2f}, {test.dataset.class_labels[tr_y]}")
            if j == 0:
                ax.set_title("\\textbf{Negative Evidence}")
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            # // HIGHLIGHT
            ax = fig.add_subplot(gs[j, 2 + 2 + 4 + 1])
            img = vis_opaque_img(tr_dp, attr.to(device), alpha=alpha_th, vis_th=vis_th)[0]
            ax.imshow(img)
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            # // HIGHLIGHT 2
            ax = fig.add_subplot(gs[j, 2 + 3 + 4 + 1])
            heatmap, attr = zennit_inner_product_explanation(model=model, train=tr_dp, test=dp,
                                                             composite_cls=EpsilonPlus,
                                                             canonizer=CIFARResNetCanonizer(), mode="test")

            img = vis_opaque_img(dp, attr.to(device), alpha=alpha_th, vis_th=vis_th)[0]
            ax.imshow(img)
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            #### BOTTOM-LEFT FIGURE
            # // TR IMG
            ax = fig.add_subplot(gs[j + 3, 2])
            tr_dp, tr_y = train[target_indices[j]]
            img = test.dataset.inverse_transform(tr_dp)
            img = img.transpose(0, 1).transpose(1, 2)
            ax.imshow(img.to("cpu"))

            tr_dp = tr_dp[None, ...].to(device)

            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            # // HEATMAP
            heatmap, attr = zennit_inner_product_explanation(model=model, train=tr_dp, test=dp,
                                                             composite_cls=EpsilonPlus,
                                                             canonizer=CIFARResNetCanonizer(), mode="train")
            ax = fig.add_subplot(gs[j + 3, 3])
            ax.set_xlabel(f"{target_xpl[target_indices[j]]:.2f}, {test.dataset.class_labels[tr_y]}")
            ax.imshow(heatmap.to("cpu"))
            if j == 0:
                ax.set_title("\\textbf{Positive Evidence}")
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            # // HIGHLIGHT
            ax = fig.add_subplot(gs[j + 3, 4])
            img = vis_opaque_img(tr_dp, attr.to(device), alpha=alpha_th, vis_th=vis_th)[0]
            ax.imshow(img)
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            # // HIGHLIGHT 2
            ax = fig.add_subplot(gs[j + 3, 5])
            heatmap, attr = zennit_inner_product_explanation(model=model, train=tr_dp, test=dp,
                                                             composite_cls=EpsilonPlus,
                                                             canonizer=CIFARResNetCanonizer(), mode="test")
            img = vis_opaque_img(dp, attr.to(device), alpha=alpha_th, vis_th=vis_th)[0]
            ax.imshow(img)
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            #### BOTTOM RIGHT FIGURE
            # //TR IMG
            ax = fig.add_subplot(gs[j + 3, 2 + 4+1])
            tr_dp, tr_y = train[target_indices[-j - 1]]
            img = test.dataset.inverse_transform(tr_dp)
            img = img.transpose(0, 1).transpose(1, 2)
            tr_dp = tr_dp[None, ...]
            tr_dp = tr_dp.to(device)
            tr_y = torch.tensor(tr_y, device=device)

            ax.imshow(img.to("cpu"))
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            # // HEATMAP
            heatmap, attr = zennit_inner_product_explanation(model=model, train=tr_dp, test=dp,
                                                             composite_cls=EpsilonPlus,
                                                             canonizer=CIFARResNetCanonizer(), mode="train")
            ax = fig.add_subplot(gs[j + 3, 2 + 1 + 4+1])
            ax.imshow(heatmap.to("cpu"))
            ax.set_xlabel(f"{target_xpl[target_indices[-j - 1]]:.2f}, {test.dataset.class_labels[tr_y]}")
            if j == 0:
                ax.set_title("\\textbf{Negative Evidence}")
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            # // HIGHLIGHT
            ax = fig.add_subplot(gs[j + 3, 2 + 2 + 4+1])
            img = vis_opaque_img(tr_dp, attr.to(device), alpha=alpha_th, vis_th=vis_th)[0]
            ax.imshow(img)
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            # // HIGHLIGHT 2
            ax = fig.add_subplot(gs[j + 3, 2 + 3 + 4 + 1])
            heatmap, attr = zennit_inner_product_explanation(model=model, train=tr_dp, test=dp,
                                                             composite_cls=EpsilonPlus,
                                                             canonizer=CIFARResNetCanonizer(), mode="test")

            img = vis_opaque_img(dp, attr.to(device), alpha=alpha_th, vis_th=vis_th)[0]
            ax.imshow(img)
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
        # plt.show()
        plt.savefig(f"../test_output/imagenet_deep_explanation_w_{i}")


def generate_without_test(device, dataset_name, save_dir, dataset_type, data_root, C_margin, alpha_th, vis_th, model,
                          imagenet_class_ids, indices):
    ds_kwargs = {
        'data_root': data_root,
        'class_groups': None,
        'image_set': "test",
        'validation_size': 0,
        "only_train": False,
        'imagenet_class_ids': imagenet_class_ids,
        'testsplit': "test"
    }

    train, test = load_datasets_reduced(dataset_name, dataset_type, ds_kwargs)
    plt.rcParams['text.usetex'] = True

    if imagenet_class_ids is not None:
        temp = torch.nn.Linear(in_features=model.classifier.in_features,
                               out_features=len(imagenet_class_ids), bias=True)
        temp.weight = torch.nn.Parameter(model.classifier.weight[imagenet_class_ids])
        temp.bias = torch.nn.Parameter(model.classifier.bias[imagenet_class_ids])
        model.classifier = temp
    model.to(device)
    err = None
    # if accuracy:
    #    acc, err = compute_accuracy(model, test,device)
    #    print(f"Accuracy: {acc}")
    explainer_cls, kwargs = DualView, {"sanity_check": False, "dir": save_dir}
    if C_margin is not None:
        kwargs["C"] = C_margin
    print(f"Generating explanations with {explainer_cls.name}")
    explainer = explainer_cls(model=model, dataset=train, device=device, **kwargs)
    restr_indices = compute_indices(model, test, 10, device)
    # print(f"Accuracy: {acc}")
    test = RestrictedDataset(test, restr_indices)

    explainer.train()
    for i in indices:
        fig = plt.figure(figsize=(7 * 1.5, 3.5 * 5))
        fig.tight_layout()
        gs = GridSpec(nrows=6, ncols=9)
        ax = fig.add_subplot(gs[0:2, 0:2])
        dp, y = test[i]
        img = test.dataset.inverse_transform(dp)
        img = img.transpose(0, 1).transpose(1, 2)
        dp = dp[None, ...].to(device)
        pred = torch.argmax(model(dp))
        ax.set_title(f'Pred.: {test.dataset.class_labels[pred]}')
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,
            left=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False,
            labelleft=False,
        )
        ax.set_ylabel(f'Image {1 + i}: {test.dataset.class_labels[y]}')
        ax.imshow(img.to("cpu"))
        y = torch.tensor(y, device=device)
        pred_xpl = explainer.explain(dp, preds=pred[None, ...], targets=None)
        pred_xpl = pred_xpl / pred_xpl.abs().max()
        pred_indices = torch.argsort(pred_xpl, descending=True)
        target_xpl = explainer.explain(dp, preds=y[None, ...], targets=None)
        target_xpl = target_xpl / target_xpl.abs().max()
        target_indices = torch.argsort(target_xpl, descending=True)
        ax = fig.add_subplot(gs[0:2, 2:8])
        ax.set_title(f"Evidence for prediction class {test.dataset.class_labels[pred]}")
        ax.axis("off")
        ax = fig.add_subplot(gs[3:6, 2:8])
        ax.set_title(f"Evidence for true class {test.dataset.class_labels[y]}")
        ax.axis("off")
        for j in range(3):
            #### TOP-LEFT FIGURE
            # // TR IMG
            ax = fig.add_subplot(gs[j, 2])
            tr_dp, tr_y = train[pred_indices[j]]
            img = test.dataset.inverse_transform(tr_dp)
            img = img.transpose(0, 1).transpose(1, 2)
            ax.imshow(img.to("cpu"))

            tr_dp = tr_dp[None, ...].to(device)

            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            # // HEATMAP
            heatmap, attr = zennit_inner_product_explanation(model=model, train=tr_dp, test=dp,
                                                             composite_cls=EpsilonPlus,
                                                             canonizer=CIFARResNetCanonizer(), mode="train")
            ax = fig.add_subplot(gs[j, 3])
            ax.set_xlabel(f"{pred_xpl[pred_indices[j]]:.2f}, {test.dataset.class_labels[tr_y]}")
            ax.imshow(heatmap.to("cpu"))
            if j == 0:
                ax.set_title("\\textbf{Positive Evidence}")
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            # // HIGHLIGHT
            ax = fig.add_subplot(gs[j, 4])
            img = vis_opaque_img(tr_dp, attr.to(device), alpha=alpha_th, vis_th=vis_th)[0]
            ax.imshow(img)
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            #### TOP RIGHT FIGURE
            # //TR IMG
            ax = fig.add_subplot(gs[j, 2 + 4])
            tr_dp, tr_y = train[pred_indices[-j - 1]]
            img = test.dataset.inverse_transform(tr_dp)
            img = img.transpose(0, 1).transpose(1, 2)
            tr_dp = tr_dp[None, ...]
            tr_dp = tr_dp.to(device)
            tr_y = torch.tensor(tr_y, device=device)

            ax.imshow(img.to("cpu"))
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            # // HEATMAP
            heatmap, attr = zennit_inner_product_explanation(model=model, train=tr_dp, test=dp,
                                                             composite_cls=EpsilonPlus,
                                                             canonizer=CIFARResNetCanonizer(), mode="train")
            ax = fig.add_subplot(gs[j, 2 + 1 + 4])
            ax.imshow(heatmap.to("cpu"))
            ax.set_xlabel(f"{pred_xpl[pred_indices[-j - 1]]:.2f}, {test.dataset.class_labels[tr_y]}")
            if j == 0:
                ax.set_title("\\textbf{Negative Evidence}")
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            # // HIGHLIGHT
            ax = fig.add_subplot(gs[j, 2 + 2 + 4])
            img = vis_opaque_img(tr_dp, attr.to(device), alpha=alpha_th, vis_th=vis_th)[0]
            ax.imshow(img)
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            #### BOTTOM-LEFT FIGURE
            # // TR IMG
            ax = fig.add_subplot(gs[j + 3, 2])
            tr_dp, tr_y = train[target_indices[j]]
            img = test.dataset.inverse_transform(tr_dp)
            img = img.transpose(0, 1).transpose(1, 2)
            ax.imshow(img.to("cpu"))

            tr_dp = tr_dp[None, ...].to(device)

            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            # // HEATMAP
            heatmap, attr = zennit_inner_product_explanation(model=model, train=tr_dp, test=dp,
                                                             composite_cls=EpsilonPlus,
                                                             canonizer=CIFARResNetCanonizer(), mode="train")
            ax = fig.add_subplot(gs[j + 3, 3])
            ax.set_xlabel(f"{target_xpl[target_indices[j]]:.2f}, {test.dataset.class_labels[tr_y]}")
            ax.imshow(heatmap.to("cpu"))
            if j == 0:
                ax.set_title("\\textbf{Positive Evidence}")
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            # // HIGHLIGHT
            ax = fig.add_subplot(gs[j + 3, 4])
            img = vis_opaque_img(tr_dp, attr.to(device), alpha=alpha_th, vis_th=vis_th)[0]
            ax.imshow(img)
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            #### BOTTOM RIGHT FIGURE
            # //TR IMG
            ax = fig.add_subplot(gs[j + 3, 2 + 4])
            tr_dp, tr_y = train[target_indices[-j - 1]]
            img = test.dataset.inverse_transform(tr_dp)
            img = img.transpose(0, 1).transpose(1, 2)
            tr_dp = tr_dp[None, ...]
            tr_dp = tr_dp.to(device)
            tr_y = torch.tensor(tr_y, device=device)

            ax.imshow(img.to("cpu"))
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            # // HEATMAP
            heatmap, attr = zennit_inner_product_explanation(model=model, train=tr_dp, test=dp,
                                                             composite_cls=EpsilonPlus,
                                                             canonizer=CIFARResNetCanonizer(), mode="train")
            ax = fig.add_subplot(gs[j + 3, 2 + 1 + 4])
            ax.imshow(heatmap.to("cpu"))
            ax.set_xlabel(f"{target_xpl[target_indices[-j - 1]]:.2f}, {test.dataset.class_labels[tr_y]}")
            if j == 0:
                ax.set_title("\\textbf{Negative Evidence}")
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
            # // HIGHLIGHT
            ax = fig.add_subplot(gs[j + 3, 2 + 2 + 4])
            img = vis_opaque_img(tr_dp, attr.to(device), alpha=alpha_th, vis_th=vis_th)[0]
            ax.imshow(img)
            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                bottom=False,
                left=False,  # ticks along the bottom edge are off
                top=False,  # ticks along the top edge are off
                labelbottom=False,
                labelleft=False,
            )
        # plt.show()
        plt.savefig(f"../test_output/imagenet_deep_explanation_wo_{i}")


generate_without_test(device, dataset_name, save_dir, dataset_type, data_root, C_margin, alpha_th, vis_th, model,
                      imagenet_class_ids,indices)
generate_with_test(device, dataset_name, save_dir, dataset_type, data_root, C_margin, alpha_th, vis_th, model,
                   imagenet_class_ids,indices)
