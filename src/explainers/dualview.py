import torch
import os
import subprocess
import time
from copy import deepcopy
from utils.csv_io import read_matrix, write_data
from utils.explainers import FeatureKernelExplainer
from struct import pack
from tqdm import tqdm

class DualView(FeatureKernelExplainer):
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
        
        if not os.path.isfile(os.path.join(self.dir,"data.shark")):
           with open(os.path.join(self.dir,"data.shark"),"wb+") as fdata:
                for dp in tqdm(normalized_samples):
                    fdata.write(pack(f'{len(dp)}f', *dp))
        with open(os.path.join(self.dir,"labels.shark"),"wb+") as flabels:
            print("Writing labels")
            flabels.write(pack('I', self.normalized_samples.shape[1])) #first line is the number of features
            flabels.write(pack(f'{len(self.labels)}I', *self.labels))

        #if not os.path.isfile(os.path.join(self.dir, "data.csv")):
            #write_data(normalized_samples, self.labels, os.path.join(self.dir, "data.csv"))
        if not os.path.isfile(os.path.join(self.dir, "weights.csv")):
            subprocess.run(['solvers/interfaces/Shark-4.0.0/solver', self.dir, str(self.C)])
        elapsed_time = time.time() - tstart
        print(f"Training took {elapsed_time} seconds")
        self.read_variables()
        self.compute_coefficients()
        return elapsed_time

