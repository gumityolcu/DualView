import argparse
from utils.data import load_datasets
from utils.models import load_model, load_cifar_model
import yaml
import logging
from metrics import *


def load_metric(dataset_type, train, test, device, coef_root, model):
    ret_dict = {"std": SameClassMetric, "group": SameSubclassMetric, "corrupt": CorruptLabelMetric,
                "mark": MarkImageMetric}
    if dataset_type not in ret_dict.keys():
        return SameClassMetric(train, test, device)
    metric_cls = ret_dict[dataset_type]
    if dataset_type == "corrupt":
        ret = metric_cls(train, test, coef_root, device)
    elif dataset_type == "mark":
        ret = metric_cls(train, test, model, device)
    else:
        ret = metric_cls(train, test, device=device)
    return ret


def evaluate(model_name, model_path, device, class_groups,
             dataset_name, dataset_type,
             data_root, xpl_root, coef_root,
             save_dir, validation_size, num_classes):
    if not torch.cuda.is_available():
        device="cpu"
    ds_kwargs = {
        'data_root': data_root,
        'class_groups': class_groups,
        'image_set': "test",
        'validation_size': validation_size,
        'only_train': False
    }
    train, test = load_datasets(dataset_name, dataset_type, **ds_kwargs)
    if dataset_name == "CIFAR":
        model = load_cifar_model(model_path, dataset_type, num_classes, device)
    else:
        model = load_model(model_name, dataset_name, num_classes).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    metric = load_metric(dataset_type, train, test, device, coef_root, model)
    print(f"Computing metric {metric.name}")
    fname=f"{dataset_name}_{dataset_type}_{xpl_root.split('/')[-1]}"
    if not os.path.isdir(xpl_root):
        raise Exception(f"Can not find explanation directory {xpl_root}")
    file_list = [f for f in os.listdir(xpl_root) if ("tgz" not in f) and ("csv" not in f) and ("coefs" not in f) and ("_tensor" not in f)]
    file_root = file_list[0].split('_')[0]
    cur_index = 0
    num_files=len(file_list)
    file_sizes=[]
    if f"{file_root}_init" in file_list:
        fname = os.path.join(xpl_root, f"{file_root}_init")
        xpl = torch.load(fname, map_location=torch.device(device))
        xpl.to(device)
        file_sizes.append(xpl.shape[0])
        assert not torch.any(xpl.isnan())
        len_xpl = xpl.shape[0]
        metric(xpl, cur_index)
        cur_index = cur_index + len_xpl
        num_files=num_files-1
    for i in range(num_files):
        fname = os.path.join(xpl_root, f"{file_root}_{i}")
        xpl = torch.load(fname, map_location=torch.device(device))
        xpl.to(device)
        # limit explanations to 4k test samples
        if cur_index+xpl.shape[0]>4000:
            xpl=xpl[:4000-cur_index,...]
        file_sizes.append(xpl.shape[0])
        assert not torch.any(xpl.isnan())
        len_xpl = xpl.shape[0]
        metric(xpl, cur_index)
        cur_index = cur_index + len_xpl

    metric.get_result(save_dir, f"{dataset_name}_{dataset_type}_{xpl_root.split('/')[-1]}_eval_results.json")




if __name__ == "__main__":
    # current = os.path.dirname(os.path.realpath(__file__))
    # parent_directory = os.path.dirname(current)
    # sys.path.append(current)
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

    evaluate(model_name=train_config.get('model_name', None),
             model_path=train_config.get('model_path', None),
             device=train_config.get('device', 'cuda'),
             class_groups=train_config.get('class_groups', None),
             dataset_name=train_config.get('dataset_name', None),
             dataset_type=train_config.get('dataset_type', 'std'),
             data_root=train_config.get('data_root', None),
             xpl_root=train_config.get('xpl_root', None),
             coef_root=train_config.get('coef_root', None),
             save_dir=train_config.get('save_dir', None),
             validation_size=train_config.get('validation_size', 2000),
             num_classes=train_config.get('num_classes')
             )
