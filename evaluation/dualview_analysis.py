import argparse
import torch
from utils.explainers import GradientProductExplainer
from utils.data import FeatureDataset
from explainers import DualView, RepresenterPointsExplainer, SimilarityExplainer, TracInExplainer, InfluenceFunctionExplainer
from utils.data import load_datasets_reduced
from utils.models import compute_accuracy, load_model, load_cifar_model
import yaml
import logging
import os
import time
import json

def load_explainer(xai_method, model_path, save_dir, dataset_name):
    explainers = {
              'representer':(RepresenterPointsExplainer, {}),
              'similarity':(SimilarityExplainer,{}),
              'tracin':(TracInExplainer,{"ckpt_dir":os.path.dirname(model_path)}),
              'mcsvm':(DualView, {"sanity_check":True, "dir": save_dir}),
              'gradprod':(GradientProductExplainer,{}),
              'influence': (InfluenceFunctionExplainer, {'recursion_depth':50, 'r':1200} if dataset_name=="MNIST" else {'recursion_depth':50, 'r':1000})
    }
    return explainers[xai_method]


def explain_model(model_name, model_path, device,
                     class_groups, dataset_name, dataset_type,
                     data_root, batch_size, save_dir,
                     validation_size, num_batches_per_file,
                     start_file, num_files, xai_method,
                     accuracy, num_classes, C_margin
                     ):
    # (explainer_class, kwargs)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not torch.cuda.is_available():
        device="cpu"
    ds_kwargs = {
        'data_root': data_root,
        'class_groups': class_groups,
        'image_set': "test",
        'validation_size': validation_size,
        "only_train": False
    }

    train, test = load_datasets_reduced(dataset_name, dataset_type, ds_kwargs)
    if dataset_name=="CIFAR":
        model=load_cifar_model(model_path,dataset_type,num_classes,device)
    else:
        model = load_model(model_name, dataset_name, num_classes).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    test_features=FeatureDataset(model,test,device)
    #err=None
    #if accuracy:
    #    acc, err = compute_accuracy(model, test,device)
    #    print(f"Accuracy: {acc}")
    explainer_cls, kwargs=load_explainer(xai_method, model_path, save_dir, dataset_name)

    C_list=[10e-7, 10e-5, 10e-3, 0.1, 1.0, 10., 100.,]
    C_list=[1e-5,1e-3]
    times=[]
    tr_accs=[]
    ts_accs=[]
    svs=[]
    sanity_errs=[]

    for c in range(len(C_list)):
        kwargs["C"]=C_list[c]
        print(f"Generating explanations with {explainer_cls.name}")
        explainer = explainer_cls(model=model, dataset=train, device=device, **kwargs)
        train_time = time.time()
        explainer.train()
        train_time=time.time()-train_time
        #explainer.read_variables()
        #explainer.compute_coefficients()
        explainer.save_coefs(save_dir)
        coefs=explainer.coefficients
        sv_nums = []
        for i in range(coefs.shape[-1]):
            ids = torch.nonzero(coefs[:, i]).squeeze()
            l = [int(j) for j in ids]
            sv_nums.append(len(l))
        tr_acc=torch.sum(1.0*(torch.argmax(torch.matmul(explainer.learned_weight,explainer.samples.T),dim=0)==explainer.labels))/explainer.samples.shape[0]
        ts_acc=torch.sum(1.0*(torch.argmax(torch.matmul(explainer.learned_weight,test_features.samples.T),dim=0)==test_features.labels))/test_features.samples.shape[0]
        tr_accs.append(float(tr_acc))
        ts_accs.append(float(ts_acc))
        sanity_errs.append(float(explainer.sanity_err))
        svs.append(sv_nums)
        times.append(train_time)
        res={
            "train_times":times,
            "train_accuracies": tr_accs,
            "test_accuracies": ts_accs,
            "sanity_errors": sanity_errs,
            "num_sv":svs,
            "C":C_list
        }
        with open(f"test_output/{dataset_name}_{dataset_type}", 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    #current = os.path.dirname(os.path.realpath(__file__))
    #parent_directory = os.path.dirname(current)
    #sys.path.append(current)
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

    explain_model(model_name=train_config.get('model_name', None),
                     model_path=train_config.get('model_path', None),
                     device=train_config.get('device', 'cuda'),
                     class_groups=train_config.get('class_groups', None),
                     dataset_name=train_config.get('dataset_name', None),
                     dataset_type=train_config.get('dataset_type', 'std'),
                     data_root=train_config.get('data_root', None),
                     batch_size=train_config.get('batch_size', None),
                     save_dir=train_config.get('save_dir', None),
                     validation_size=train_config.get('validation_size', 2000),
                     accuracy=train_config.get('accuracy', False),
                     num_batches_per_file=train_config.get('num_batches_per_file', 10),
                     start_file=train_config.get('start_file', 0),
                     num_files=train_config.get('num_files', 100),
                     xai_method=train_config.get('xai_method', None),
                     num_classes=train_config.get('num_classes'),
                    C_margin=train_config.get('C',None)
                     )
