import os 
import json
import torch

def compute_mean_std(tensor):
    mean=tensor.sum()/tensor.shape[0]
    centered=tensor-mean
    mean_squared=(centered*centered).sum()/tensor.shape[0]
    return mean, torch.sqrt(mean_squared)

N=50
if __name__=="__main__":
    ds="MNIST"
    method="mcsvm"
    dir=f"/home/fe/yolcu/Documents/Code/THESIS/explanations/{ds}/time/{method}"

    files=[]
    cumul_times=[]
    for f in os.listdir(dir):
        if "resources_page" in f:
            with open(os.path.join(dir,f),"r") as file:
                res=json.load(file)
            cumul_times+=res['xpl']
    #if N>0:
    #    cumul_times=cumul_times[:N]
    mean,std = compute_mean_std(torch.tensor(cumul_times))
    print(f"mean: {mean} std:{std} over N={N} samples")


