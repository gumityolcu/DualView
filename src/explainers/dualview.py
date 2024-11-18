import torch
import os
import subprocess
import time
from copy import deepcopy
from utils.csv_io import read_matrix, write_data
from utils.explainers import FeatureKernelExplainer
from sklearn.svm import LinearSVC ##modified LIBLINEAR MCSVM_CS_Solver which returns the dual variables


class DualView(FeatureKernelExplainer):
    name = "DualViewExplainer"
    def __init__(self, model, dataset, device, dir, C=1.0, max_iter=50000, normalize=False):
        super().__init__(model, dataset, device, normalize=normalize)
        self.C=C
        if dir[-1]=="\\":
            dir=dir[:-1]
        self.dir=dir
        self.features_dir=dir
        self.max_iter=max_iter
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)

    def read_variables(self):
        self.learned_weight = torch.load(os.path.join(self.dir,"weights"), map_location=self.device).to(torch.float)
        self.coefficients=torch.load(os.path.join(self.dir,"coefficients"), map_location=self.device).to(torch.float)
        self.train_time=torch.load(os.path.join(self.dir,"train_time"), map_location=self.device).to(torch.float)

    def train(self):
        tstart = time.time()
        
        if not os.path.isfile(os.path.join(self.features_dir, "samples")):
            torch.save(self.normalized_samples,os.path.join(self.features_dir, "samples"))
        if not os.path.isfile(os.path.join(self.features_dir, "labels")):
            torch.save(self.labels,os.path.join(self.features_dir, "labels"))
        
        if os.path.isfile(os.path.join(self.dir,'weights')) and os.path.isdir(os.path.join(self.dir,'coefficients')):
            self.read_variables()
        else:
            model = LinearSVC(multi_class="crammer_singer", max_iter=self.max_iter, C=self.C)
            model.fit(self.normalized_samples.cpu(),self.labels.cpu())

            self.coefficients=torch.tensor(model.alpha_.T,dtype=torch.float,device=self.device)
            self.learned_weight=torch.tensor(model.coef_,dtype=torch.float, device=self.device)
            self.train_time = torch.tensor(time.time() - tstart)

            torch.save(self.train_time,os.path.join(self.dir,'train_time'))
            torch.save(self.learned_weights,os.path.join(self.dir,'weights'))
            torch.save(self.coefficients,os.path.join(self.dir,'coefficients'))
            print(f"Training took {self.train_time} seconds")
        return self.train_time

    def self_influences(self, only_coefs=False):
        self_coefs=super().self_influences()
        if only_coefs:
            return self_coefs
        else:
            return self.normalized_samples.norm(dim=-1)*self_coefs

class DualView_Shark(FeatureKernelExplainer):
    name = "DualViewExplainer"
    def __init__(self, model, dataset, device, dir, C=1.0, normalize=True):
        if os.path.isfile(os.path.join(dir,"labels_tensor")) and os.path.isfile(os.path.join(dir,"samples_tensor")):
            super().__init__(model, dataset, device, dir, normalize=normalize)
        else:
            super().__init__(model, dataset, device, os.path.join(dir, "data.csv"), normalize=normalize)
        self.C=C
        if dir[-1]=="\\":
            dir=dir[:-1]
        self.dir=dir

    def compute_coefficients(self):
        indices = range(self.dual_variables.shape[0])
        self.coefficients = deepcopy(self.dual_variables)
        temp_positive_coefficients = torch.sum(self.dual_variables, dim=1) - self.coefficients[indices, self.labels[indices]]
        self.coefficients=-1.*self.coefficients
        self.coefficients[indices, self.labels[indices]] = temp_positive_coefficients

    def read_variables(self):
        self.learned_weight = torch.tensor(read_matrix(os.path.join(self.dir,'weights.csv')),dtype=torch.float, device=self.device)
        if os.path.isfile(os.path.join(self.dir,"dualvars_tensor")):
            self.dual_variables=torch.load(os.path.join(self.dir,"dualvars_tensor"), map_location=self.device)
        else:
            self.dual_variables = read_matrix(os.path.join(self.dir, "dualvars.csv"))
            torch.save(self.dual_variables, os.path.join(self.dir,"dualvars_tensor"))
        self.dual_variables = torch.tensor(self.dual_variables[:, :-1],device=self.device)

    def train(self):
        tstart = time.time()
        normalized_samples=self.normalize_features(self.samples)
        if not os.path.isfile(os.path.join(self.dir, "samples_tensor")):
            torch.save(normalized_samples,os.path.join(self.dir, "samples_tensor"))
        if not os.path.isfile(os.path.join(self.dir, "labels_tensor")):
            torch.save(self.labels,os.path.join(self.dir, "labels_tensor"))
        if not os.path.isfile(os.path.join(self.dir, "data.csv")):
            write_data(normalized_samples, self.labels, os.path.join(self.dir, "data.csv"))
        if not os.path.isfile(os.path.join(self.dir, "weights.csv")):
            subprocess.run(['solvers/interfaces/Shark-4.0.0/solver', self.dir, str(self.C)])
        elapsed_time = time.time() - tstart
        print(f"Training took {elapsed_time} seconds")
        self.read_variables()
        self.compute_coefficients()
        return elapsed_time

