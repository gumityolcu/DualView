import argparse
import torch
from time import time
from utils import xplain_to_compute_time
from utils.explainers import GradientProductExplainer
from explainers import DualView, RepresenterPointsExplainer, SimilarityExplainer, InfluenceFunctionExplainer, RPSimilarityExplainer
from utils.data import load_datasets_reduced
from utils.models import load_model, load_cifar_model
import yaml
import logging
import os

def load_explainer(xai_method, model_path, save_dir, dataset_name):
    explainers = {
              'representer':(RepresenterPointsExplainer, {}),
              'similarity':(SimilarityExplainer,{}),
              'rp_similarity': (RPSimilarityExplainer, {"dir": save_dir, 'dimensions': 128}),
              'mcsvm':(DualView, {"sanity_check":True, "dir": save_dir}),
              'gradprod':(GradientProductExplainer,{}),
              'influence': (InfluenceFunctionExplainer, {'recursion_depth':50, 'r':1200} if dataset_name=="MNIST" else {'recursion_depth':50, 'r':1000})
    }
    return explainers[xai_method]




if __name__ == "__main__":
    #current = os.path.dirname(os.path.realpath(__file__))
    #parent_directory = os.path.dirname(current)
    #sys.path.append(current)
    total_time=time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds_name', type=str)
    parser.add_argument('--page_size', type=int)
    parser.add_argument('--start_page', type=int)
    parser.add_argument('--num_pages', type=int)
    parser.add_argument('--mode', type=str, default="b")
    parser.add_argument('--skip', type=int, default=0)
    args = parser.parse_args()
    ds_name = args.ds_name
    mode = args.mode
    base_path= "../config_files/local/time"
    config_names={
        "MNIST_t": ["basic_conv_std_mcsvm.yaml", "basic_conv_std_representer.yaml", "basic_conv_std_rp_similarity.yaml"],
        "MNIST_g": ["basic_conv_std_influence.yaml", "basic_conv_std_similarity.yaml"],
        "MNIST_b": ["basic_conv_std_mcsvm.yaml", "basic_conv_std_representer.yaml", "basic_conv_std_rp_similarity.yaml","basic_conv_std_influence.yaml","basic_conv_std_similarity.yaml"],
        "CIFAR_t": ["resnet_std_mcsvm.yaml", "resnet_std_representer.yaml", "resnet_std_rp_similarity.yaml"],
        "CIFAR_g": ["resnet_std_influence.yaml"],
        "CIFAR_b": ["resnet_std_mcsvm.yaml", "resnet_std_representer.yaml", "resnet_std_rp_similarity.yaml","resnet_std_influence.yaml"]
        #"CIFAR_b": ["resnet_std_rp_similarity.yaml","resnet_std_influence.yaml"]
    }
    first_config=True
    explainer_clses=[]
    save_dirs=[]
    kwargses=[]
    for fname in config_names[f"{ds_name}_{mode}"]:
        config_file=os.path.join(base_path,ds_name,fname)
        with open(config_file, "r") as stream:
            try:
                train_config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logging.info(exc)

        save_dir = f"{train_config['save_dir']}/{os.path.basename(config_file)[:-5]}"

        if first_config:
            first_config=False
            model_name=train_config.get('model_name', None)
            model_path=train_config.get('model_path', None)
            device=train_config.get('device', 'cuda')
            class_groups=train_config.get('class_groups', None)
            dataset_name=train_config.get('dataset_name', None)
            dataset_type=train_config.get('dataset_type', 'std')
            data_root=train_config.get('data_root', None)
            validation_size=train_config.get('validation_size', 2000)
            accuracy=train_config.get('accuracy', False)
            num_classes=train_config.get('num_classes')
            imagenet_class_ids=train_config.get('imagenet_class_ids',[i for i in range(397)])
            testsplit=train_config.get('testsplit',"test")
            if not torch.cuda.is_available():
                device = "cpu"
            ds_kwargs = {
                'data_root': data_root,
                'class_groups': class_groups,
                'image_set': "test",
                'validation_size': validation_size,
                "only_train": False,
                'imagenet_class_ids': imagenet_class_ids,
                'testsplit': testsplit
            }

            train, test = load_datasets_reduced(dataset_name, dataset_type, ds_kwargs)
            if dataset_name == "CIFAR":
                model = load_cifar_model(model_path, dataset_type, num_classes, device)
            else:
                model = load_model(model_name, dataset_name, num_classes).to(device)
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint["model_state"])
            model.to(device)

        save_dir=train_config.get('save_dir', None)
        save_dirs.append(save_dir)
        C_margin=train_config.get('C',None)
        xai_method=train_config.get('xai_method', None)
        page_size=args.page_size
        start_page=args.start_page
        num_pages=args.num_pages
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        err=None
        explainer_cls, kwargs=load_explainer(xai_method, model_path, save_dir, dataset_name)
        explainer_clses.append(explainer_cls)
        if C_margin is not None:
            kwargs["C"]=C_margin
        kwargses.append(kwargs)

    xplain_to_compute_time(
        model=model,
        train=train,
        test=test,
        device=device,
        explainer_clses=explainer_clses,
        kwargses=kwargses,
        err=err,
        save_dirs=save_dirs,
        start_page=start_page,
        num_pages=num_pages,
        page_size=page_size,
        skip=args.skip
    )
    print(f"jobtime: {time()-total_time}")