import argparse
import math
import logging
from models import BasicConvModel, BasicFCModel
import os
import json
import sys
import yaml
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from utils.models import load_model
from utils.data import ReduceLabelDataset, FeatureDataset, GroupLabelDataset, CorruptLabelDataset
import torch
import matplotlib.pyplot as plt
from datasets.MNIST import MNIST
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.tensorboard import SummaryWriter
from utils.data import load_datasets_reduced


def parse_report(rep, num_classes):
    print(rep)
    ret = dict()
    rep = rep.split('\n')
    keys = rep[0].strip().split(' ')
    i = 0
    while i < len(keys):
        k = keys[i]
        if k == '':
            if i != len(keys) - 1:
                keys = keys[:i] + keys[i + 1:]
                i = i - 1
            else:
                keys = keys[:-1]
        i = i + 1
    keys = keys[:-1]
    for k in keys:
        ret[k] = dict()
    rep = rep[num_classes + 3:-1]
    for line in rep:
        for i, key in enumerate(keys):
            spl = line.strip().split('    ')
            ret[key][spl[0].strip().replace(' ', '_')] = float(spl[i + 1].strip())
    return ret


def get_validation_loss(model, ds, loss, device):
    model.eval()
    loader = DataLoader(ds, batch_size=64)
    l = torch.tensor(0.0)
    # count = 0
    for inputs, targets in tqdm(iter(loader)):
        inputs = inputs.to(torch.device(device))
        targets = targets.to(torch.device(device))
        with torch.no_grad():
            y = model(inputs)
            l = l + loss(y, targets)
        # count = count + inputs.shape[0]
    # l = l / count
    model.train()
    return l

def load_scheduler(name, optimizer):
    return ConstantLR(optimizer=optimizer, last_epoch=-1 ) 

def load_optimizer(name, model, lr):
    return SGD(model.parameters(), lr=lr, momentum=0.9)

def load_augmentation(name):
    return lambda x:x

def load_loss():
    return CrossEntropyLoss()

def start_training(model_name, device, num_classes, class_groups, data_root, epochs,
                   batch_size, lr, save_dir, save_each, model_path, base_epoch,
                   dataset_name, dataset_type, num_batches_eval, validation_size,
                   augmentation, optimizer, scheduler, loss):
    if not torch.cuda.is_available():
        device="cpu"
    if dataset_type=="group":
        num_classes=len(class_groups)
    model = load_model(model_name, dataset_name, num_classes).to(device)
    tensorboarddir = f"{model_name}_{lr}_{scheduler}_{optimizer}{f'_aug' if augmentation is not None else ''}"
    tensorboarddir = os.path.join(save_dir, tensorboarddir)
    writer = SummaryWriter(tensorboarddir)

    learning_rates=[]
    train_losses = []
    validation_losses = []
    validation_epochs = []
    val_acc = []
    train_acc = []
    loss=load_loss()
    optimizer = load_optimizer(optimizer, model, lr)
    scheduler = load_scheduler(scheduler, optimizer)
    if augmentation is not None:
        augmentation = load_augmentation(augmentation)

    kwargs = {
        'data_root': data_root,
        'class_groups': class_groups,
        'image_set': "val",
        'validation_size': validation_size,
        'only_train': True
    }
    corrupt = (dataset_type == "corrupt")
    group = (dataset_type == "group")
    ds, valds = load_datasets_reduced(dataset_name, dataset_type, kwargs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    saved_files = []

    if model_path is not None:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        train_losses = checkpoint["train_losses"]
        validation_losses = checkpoint["validation_losses"]
        validation_epochs = checkpoint["validation_epochs"]
        val_acc = checkpoint["validation_accuracy"]
        train_acc = checkpoint["train_accuracy"]

    for i,r in enumerate(learning_rates):
        writer.add_scalar('Metric/lr', r, i)
    for i, r in enumerate(train_acc):
        writer.add_scalar('Metric/train_acc', r, i)
    for i, r in enumerate(val_acc):
        writer.add_scalar('Metric/val_acc', r, validation_epochs[i])
    for i, l in enumerate(train_losses):
        writer.add_scalar('Loss/train', l, i)
    for i, l in enumerate(validation_losses):
        writer.add_scalar('Loss/val', l, validation_epochs[i])

    model.train()
    best_model_yet = model_path
    best_loss_yet = None

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir,exist_ok=True)
    for e in range(epochs):
        y_true = torch.empty(0, device=device)
        y_out = torch.empty((0, num_classes), device=device)
        cum_loss = 0
        cnt = 0
        for inputs, targets in tqdm(iter(loader)):
            inputs = inputs.to(device)
            targets = targets.to(device)
        
            if augmentation is not None:
                inputs=augmentation(inputs)

            y_true = torch.cat((y_true, targets), 0)

            optimizer.zero_grad()
            logits = model(inputs)
            l = loss(logits, targets)
            y_out = torch.cat((y_out, logits.detach().clone()), 0)
            if math.isnan(l):
                if not os.path.isdir("./broken_model"):
                    os.mkdir("broken_model")
                save_dict = {
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'epoch': base_epoch + e,
                    'learning_rates': learning_rates,
                    'train_losses': train_losses,
                    'validation_losses': validation_losses,
                    'validation_epochs': validation_epochs,
                    'validation_accuracy': val_acc,
                    'ds_type':dataset_type,
                    'train_accuracy': train_acc
                }
                path = os.path.join("./broken_model", f"{dataset_name}_{model_name}_{base_epoch + e}")
                torch.save(save_dict, path)
                print("NaN loss")
                exit()
            l.backward()
            optimizer.step()
            cum_loss = cum_loss + l
            cnt = cnt + inputs.shape[0]
        #y_out = torch.softmax(y_out, dim=1)
        y_pred = torch.argmax(y_out, dim=1)
        # y_true = y_true.cpu().numpy()
        # y_out = y_out.cpu().numpy()
        # y_pred = y_pred.cpu().numpy()
        train_loss = cum_loss.detach().cpu()
        acc = (y_true == y_pred).sum() / y_out.shape[0]
        train_acc.append(acc)
        print(f"train accuracy: {acc}")
        writer.add_scalar('Metric/train_acc', acc, base_epoch + e)
        #writer.add_scalar('Metric/learning_rates', 0.95, base_epoch + e)
        train_losses.append(train_loss)
        writer.add_scalar('Loss/train', train_loss, base_epoch + e)
        print(f"Epoch {e + 1}/{epochs} loss: {cum_loss}")  # / cnt}")
        print("\n==============\n")
        learning_rates.append(scheduler.get_lr())
        scheduler.step()
        if (e + 1) % save_each == 0:
            validation_loss = get_validation_loss(model, valds, loss, device)
            validation_losses.append(validation_loss.detach().cpu())
            validation_epochs.append(e)
            save_dict = {
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'epoch': base_epoch + e,
                'train_losses': train_losses,
                'validation_losses': validation_losses,
                'validation_epochs': validation_epochs,
                'validation_accuracy': val_acc,
                'train_accuracy': train_acc
            }
            if corrupt:
                save_dict["corrupt_samples"] = ds.dataset.corrupt_samples
                save_dict["corrupt_labels"] = ds.dataset.corrupt_labels
            if group:
                save_dict["classes"] = ds.dataset.classes
            save_id = f"{dataset_name}_{model_name}_{base_epoch + e}"
            path = os.path.join(save_dir, save_id)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            torch.save(save_dict, path)
            saved_files.append((path, save_id))

            print(f"\n\nValidation loss: {validation_loss}\n\n")
            writer.add_scalar('Loss/val', validation_loss, base_epoch + e)
            valeval = evaluate_model(model_name=model_name, device=device, num_classes=num_classes,
                                     data_root=data_root,
                                     batch_size=batch_size, num_batches_to_process=num_batches_eval,
                                     load_path=best_model_yet, dataset_name=dataset_name, dataset_type=dataset_type,
                                     validation_size=validation_size,
                                     image_set="val", class_groups=class_groups
                                     )
            print(f"validation accuracy: {valeval}")
            writer.add_scalar('Metric/val_acc', valeval, base_epoch + e)
            val_acc.append(valeval)
            if best_loss_yet is None or best_loss_yet > validation_loss:
                best_loss_yet = validation_loss
                path = os.path.join(save_dir, f"best_val_score_{dataset_name}_{model_name}_{base_epoch + e}")
                torch.save(save_dict, path)
                if best_model_yet is not None:
                    os.remove(best_model_yet)
                best_model_yet = path
        writer.flush()

        # Save train and validation loss figures
        # plt.subplot(2, 1, 1)
        # plt.title("Training Loss")
        # plt.plot(base_epoch + np.asarray(range(epochs)), np.asarray(train_losses))
        # plt.subplot(2, 1, 2)
        # plt.title("Validation Loss")
        # plt.plot(base_epoch + np.asarray(vaidation_epochs), np.asarray(validation_losses))
        # plt.savefig(os.path.join(save_dir, f"{dataset_name}_{model_name}_{base_epoch + epochs}_losses.png"))
    writer.close()
    save_id = os.path.basename(best_model_yet)


def evaluate_model(model_name, device, num_classes, class_groups, data_root, batch_size,
                   num_batches_to_process, load_path, dataset_name, dataset_type, validation_size, image_set):
    model = load_model(model_name, dataset_name, num_classes).to(device)

    kwparams = {
        'data_root': data_root,
        'image_set': image_set,
        'class_groups': class_groups,
        'validation_size': validation_size,
        'only_train': True
    }
    _, ds = load_datasets_reduced(dataset_name=dataset_name, dataset_type=dataset_type, kwparams=kwparams)
    if not len(ds) > 0:
        return 0.0, 0.0
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    if load_path is not None:
        checkpoint = torch.load(load_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    # classes = ds.all_classes
    model.eval()
    y_true = torch.empty(0, device=device)
    y_out = torch.empty((0, num_classes), device=device)

    for i, (inputs, targets) in enumerate(tqdm(iter(loader), total=min(num_batches_to_process, len(loader)))):
        inputs = inputs.to(device)
        targets = targets.to(device)
        y_true = torch.cat((y_true, targets), 0)
        with torch.no_grad():
            logits = model(inputs)
        y_out = torch.cat((y_out, logits), 0)

    results = dict()
    results["tpr"] = dict()
    results["fpr"] = dict()
    results["roc_auc"] = dict()

    y_out = torch.softmax(y_out, dim=1)
    y_pred = torch.argmax(y_out, dim=1)
    # y_true = y_true.cpu().numpy()
    # y_out = y_out.cpu().numpy()
    # y_pred = y_pred.cpu().numpy()
    model.train()
    return (y_true == y_pred).sum() / y_out.shape[0]

def test_all_models(model_root_path, device, num_classes, class_groups, data_root, batch_size,
                   num_batches_to_process, dataset_name, dataset_type, validation_size, save_dir):
    model_root_path=os.path.join(model_root_path, dataset_name)
    dirlist=[direc for direc in os.listdir(model_root_path) if ((os.path.isdir(os.path.join(model_root_path, direc))) and (direc != "results_all"))]
    results_dict={}
    for dir_name in dirlist:
        print(f"Testing: {dir_name}")       
        results_dict[dir_name]={}
        spl=dir_name.split("_")
        model_name=f"{spl[0]}_{spl[1]}"
        model_path=os.path.join(model_root_path, dir_name)
        model_path=os.path.join(model_path,f"{dataset_name}_{model_name}")
        train_acc=evaluate_model(
            model_name=model_name,
            device=device,
            num_classes=num_classes,
            class_groups=class_groups,
            data_root=data_root,
            batch_size=batch_size,
            num_batches_to_process=num_batches_to_process,
            load_path=model_path,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            validation_size=validation_size,
            image_set="train"
        )
        test_acc = evaluate_model(
            model_name=model_name,
            device=device,
            num_classes=num_classes,
            class_groups=class_groups,
            data_root=data_root,
            batch_size=batch_size,
            num_batches_to_process=num_batches_to_process,
            load_path=model_path,
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            validation_size=validation_size,
            image_set="test"
        )
        print(f"train: {train_acc} - test: {test_acc}")
        results_dict[dir_name]["train"]=train_acc
        results_dict[dir_name]["test"]=test_acc
    save_dir=os.path.join(save_dir,"results.json")
    with open(save_dir, 'w') as file:
        json.dump(results_dict, file)
    return results_dict



if __name__ == "__main__":
    current = os.path.dirname(os.path.realpath(__file__))
    parent_directory = os.path.dirname(current)
    sys.path.append(current)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, "r") as stream:
        try:
            train_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.info(exc)

    save_dir = f"{train_config['save_dir']}/{os.path.basename(config_file)[:-5]}"

    #test_all_models(
    #    model_root_path="/home/fe/yolcu/Documents/Code/THESIS/checkpoints",
    #    device=train_config.get('device', 'cuda'),
    #    num_classes=train_config.get('num_classes', None),
    #    class_groups=train_config.get('class_groups', None),
    #    dataset_name=train_config.get('dataset_name', None),
    #    dataset_type=train_config.get('dataset_type', 'std'),
    #    data_root=train_config.get('data_root', None),
    #    batch_size=train_config.get('batch_size', None),
    #    save_dir=train_config.get('save_dir', None),
    #    num_batches_to_process=train_config.get('num_batches_eval', None),
    #    validation_size=train_config.get('validation_size', 2000)
    #)
    #exit()

    start_training(model_name=train_config.get('model_name', None),
                   model_path=train_config.get('model_path', None),
                   base_epoch=train_config.get('base_epoch', 0),
                   device=train_config.get('device', 'cuda'),
                   num_classes=train_config.get('num_classes', None),
                   class_groups=train_config.get('class_groups', None),
                   dataset_name=train_config.get('dataset_name', None),
                   dataset_type=train_config.get('dataset_type', 'std'),
                   data_root=train_config.get('data_root', None),
                   epochs=train_config.get('epochs', None),
                   batch_size=train_config.get('batch_size', None),
                   lr=train_config.get('lr', 0.1),
                   augmentation=train_config.get('augmentation', None),
                   loss=train_config.get('loss', None),
                   optimizer=train_config.get('optimizer', None),
                   save_dir=train_config.get('save_dir', None),
                   save_each=train_config.get('save_each', 100),
                   num_batches_eval=train_config.get('num_batches_eval', None),
                   validation_size=train_config.get('validation_size', 2000),
                   scheduler=train_config.get('scheduler', None)
                   )
