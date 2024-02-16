import os.path
from tqdm import tqdm
import torch
from time import time
from math import sqrt
from utils.data import ReduceLabelDataset, CorruptLabelDataset, GroupLabelDataset, MarkDataset
from utils.explainers import GradientProductExplainer, Explainer

class SimilarityExplainer(GradientProductExplainer):
    name = "SimilarityExplainer"

    def __init__(self,model,dataset,device):
        super().__init__(model,dataset,device,loss=None)

    def train(self):
        return 0.

    def explain(self, x, preds, targets=None):
        xpl=super().explain(x=x, preds=preds)
        return xpl#/self.norms[None,:]

class RPSimilarityExplainer(Explainer):
    name="RPSimilarityExplainer"
    def __init__(self,model,dataset,dir,dimensions,device):
        super().__init__(model,dataset,device)
        self.number_of_params=0
        for p in list(self.model.sim_parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            self.number_of_params += nn
        # USE get_param_grad instead of grad_ds = GradientDataset(self.model, dataset)
        self.dataset = dataset
        self.norms=torch.ones(len(self.dataset),device=self.device)

        self.dir=dir
        self.dimensions=dimensions

    def train(self):
        ds=self.dataset
        if isinstance(self.images,ReduceLabelDataset):
            ds=ds.dataset

        d = {CorruptLabelDataset: "corrupt", GroupLabelDataset: "group", MarkDataset:"mark"}
        t = type(ds)
        if t in d.keys():
            ds_type=d[t]
            ds_name=ds.dataset.name
        else:
            ds_type="std"
            ds_name=ds.name
        t0=time()
        file_path=os.path.join(self.dir,"random_matrix_tensor")
        if os.path.isfile(file_path):
            self.random_matrix=torch.load(file_path,map_location=self.device)
        else:
            self.random_matrix=self.make_random_matrix()
            torch.save(self.random_matrix,file_path)

        file_path=os.path.join(self.dir,"train_grads_tensor")
        if os.path.isfile(file_path):
            self.train_grads=torch.load(file_path,map_location=self.device)
        else:
            self.train_grads=self.make_train_grads()
            torch.save(self.train_grads,file_path)
        return time()-t0

    def make_random_matrix(self):
        unitvar = torch.randn((self.dimensions,self.number_of_params),device=self.device)
        return unitvar/sqrt(self.dimensions)

    def make_train_grads(self):
        train_grads=torch.empty(len(self.dataset),self.dimensions,device=self.device)
        for i,(x,y) in tqdm(enumerate(self.dataset)):
            grad=self.get_param_grad(x,y)
            train_grads[i]=grad/grad.norm()
        return train_grads

    def explain(self, x, preds, targets):
        xpl=torch.empty(x.shape[0],len(self.dataset),device=self.device)
        for i in tqdm(range(x.shape[0])):
            test_grad=self.get_param_grad(x[i],preds[i])
            xpl[i]=torch.matmul(self.train_grads,test_grad)
        return xpl

    def get_param_grad(self, x, index):
        x = x.to(self.device)
        out = self.model(x[None, :, :])
        self.model.zero_grad()
        out[0][index].backward()
        cumul = torch.empty(0, device=self.device)
        for par in self.model.sim_parameters():
            grad = par.grad.flatten()
            cumul = torch.cat((cumul, grad), 0)
        grads=torch.matmul(self.random_matrix,cumul)
        return grads
